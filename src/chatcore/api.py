"""HTTP API (docs/engine-core-plan.md Phase 5).

    POST /v1/bots/{bot}/sessions/{session}/messages   {"text": "..."} -> bot messages
    POST /v1/bots/{bot}/nlu/parse                     {"text": "..."} -> NLU result only
    GET  /healthz
    /v1/builder/...                                    drafts, training jobs, test builds, deploy (admin)

Run: `make serve` (uv run python -m chatcore.api --bot bots/sample).
Errors never carry exception text: 404 unknown bot, 409 the session is mid-turn,
422 malformed request, 503 storage unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import sys
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from chatcore.bot.loader import BotError, document_version
from chatcore.bot.repository import RevisionConflict, VersionNotFound
from chatcore.bot.schema import BotDocument
from chatcore.dialogue.state import fold
from chatcore.dialogue.turn import MAX_INPUT_CHARS, TurnResult
from chatcore.host import BelowBar, BotHost, Host, NoDraft, NoEarlierDeployment, NotTrained
from chatcore.jobs import JobFailed, JobRunner
from chatcore.nlu.train import TrainingError
from chatcore.runtime import BotRuntime
from chatcore.store.base import SessionBusy

try:  # storage errors that mean "try again later"
    from psycopg import OperationalError as _DbDown

    UNAVAILABLE: tuple[type[BaseException], ...] = (OSError, _DbDown)
except ImportError:  # pragma: no cover
    UNAVAILABLE = (OSError,)

log = logging.getLogger("chatcore.api")
turn_log = logging.getLogger("chatcore.turn")  # one JSON line per turn

# bot (<= 64) + ":" + session (<= 63) fits the stores' 128-character session key
SESSION_ID = PathParam(pattern=r"^[A-Za-z0-9_-]{1,63}$")
BOT_ID = PathParam(pattern=r"^[a-z][a-z0-9_]{0,63}$")
VERSION = PathParam(pattern=r"^[0-9a-f]{12}$")
JOB_ID = PathParam(pattern=r"^[0-9a-f]{32}$")
TEST_SESSION_SEP = "~"  # test-build sessions are "<bot>~<session>": never "<bot>:...", so no stats or misunderstood


class MessageIn(BaseModel):
    text: str = Field(max_length=10_000)  # the engine itself answers input_invalid above 1,000 characters


class MessageOut(BaseModel):
    response: str
    text: str
    quick_replies: list[str]


class TurnOut(BaseModel):
    bot: str
    session: str
    messages: list[MessageOut]
    nlu: dict[str, Any] | None = None


class DeployIn(BaseModel):
    version: str = Field(pattern=r"^[0-9a-f]{12}$")
    note: str = Field("", max_length=100)  # what changed, shown in the deploy history


class ParseOut(BaseModel):
    bot: str
    text: str
    intents: list[tuple[str, float]]  # top 3: (intent, probability)
    similarity: float  # cosine to the nearest example of any intent
    matched_by: Literal["exact", "classifier"]
    entities: list[dict[str, Any]]


class MisunderstoodItem(BaseModel):
    session: str
    text: str
    rule: str
    intents: list[tuple[str, float]]


class MisunderstoodOut(BaseModel):
    bot: str
    items: list[MisunderstoodItem]


class BotSummaryOut(BaseModel):
    id: str
    label: str
    live: str | None  # deployed version
    live_number: int | None
    serving: str | None  # version this server answers with (None: not loaded here)
    draft_revision: int | None
    undeployed_changes: bool  # the draft differs from what is live


class BotListOut(BaseModel):
    bots: list[BotSummaryOut]


class DraftSavedOut(BaseModel):
    bot: str
    version: str | None  # the draft's content hash; None when it does not parse
    revision: int
    valid: bool
    errors: list[str]
    warnings: list[str]


class JobOut(BaseModel):
    id: str
    kind: str  # "version" | "test_build"
    bot: str
    status: Literal["queued", "running", "done", "failed"]
    result: dict[str, Any] | None  # version: {version, number, macro_f1, ...}; test_build: {build}
    error: dict[str, Any] | None  # {kind, detail, ...}
    created_at: str
    finished_at: str | None


class ActionOut(BaseModel):
    scenario: str
    action: str
    ok: bool


class FrameOut(BaseModel):
    scenario: str
    step: int
    awaiting: str | None
    slots: dict[str, Any]


class TestTurnOut(BaseModel):
    bot: str
    build: str
    session: str
    messages: list[MessageOut]
    nlu: dict[str, Any] | None
    actions: list[ActionOut]
    stack: list[FrameOut]


class VersionOut(BaseModel):
    version: str
    number: int
    created_at: str
    macro_f1: float | None
    live: bool


class DeploymentOut(BaseModel):
    version: str
    number: int
    note: str
    deployed_at: str


class VersionsOut(BaseModel):
    bot: str
    live: str | None
    versions: list[VersionOut]
    history: list[DeploymentOut]


class LiveOut(BaseModel):
    bot: str
    live: str


# The draft is stored as written (even when invalid), so it is described with the bot
# document's schema but not validated against it.
DOCUMENT_SCHEMA = {"$ref": "#/components/schemas/BotDocument"}
DRAFT_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {"model": BotDocument, "description": "the draft as stored (may be invalid); ETag header"}}
DRAFT_BODY: dict[str, Any] = {
    "requestBody": {"required": True, "content": {"application/json": {"schema": DOCUMENT_SCHEMA}}}}

MISUNDERSTOOD_RULES = ("fallback", "clarify")


def create_app(host: Host, admin_token: str | None = None, jobs: JobRunner | None = None) -> FastAPI:
    """`admin_token`: when set, builder endpoints require `Authorization: Bearer <token>`.
    `jobs`: where builder training runs (a new JobRunner when the host is a BotHost)."""
    app = FastAPI(title="chatcore", version="0.0.1")

    def runtime(bot: str) -> BotRuntime:
        rt = host.runtime(bot)
        if rt is None:
            raise HTTPException(404, f"unknown bot {bot!r}")
        return rt

    expected = f"Bearer {admin_token}".encode() if admin_token is not None else None

    def admin(authorization: str | None = Header(default=None)) -> None:
        given = (authorization or "").encode("utf-8", "surrogateescape")
        if expected is not None and not hmac.compare_digest(given, expected):  # constant time
            raise HTTPException(401, "builder endpoints need the admin token")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "bots": host.bots()}

    @app.post("/v1/bots/{bot}/sessions/{session}/messages", response_model=TurnOut)
    def message(body: MessageIn, bot: str = BOT_ID, session: str = SESSION_ID, debug: bool = False) -> Any:
        key = f"{bot}:{session}"
        try:
            rt = host.runtime_for_session(bot, key)
        except UNAVAILABLE:
            log.exception("storage unavailable for %s/%s", bot, session)
            return JSONResponse({"detail": "temporarily unavailable, please retry"}, status_code=503)
        except Exception:
            log.exception("could not pick a version for %s/%s", bot, session)
            return JSONResponse({"detail": "internal error"}, status_code=500)
        if rt is None:
            raise HTTPException(404, f"unknown bot {bot!r}")
        started = time.perf_counter()
        result = run_turn(rt, key, body.text)
        if isinstance(result, JSONResponse):
            return result
        nlu = next((e.nlu for e in reversed(result.events) if e.type == "commands_received"), None)
        turn_log.info(json.dumps({
            "bot": bot, "version": rt.bot.version, "session": session,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "rule": (nlu or {}).get("rule"), "intent": ((nlu or {}).get("intents") or [[None]])[0][0],
            "responses": [m.response for m in result.messages],
        }, ensure_ascii=False))
        return TurnOut(bot=bot, session=session, messages=messages_out(result), nlu=nlu if debug else None)

    @app.post("/v1/bots/{bot}/nlu/parse", response_model=ParseOut)
    def parse(body: MessageIn, bot: str = BOT_ID) -> Any:
        rt = runtime(bot)
        text = " ".join(body.text.split())  # what a turn would see
        if not text or len(text) > MAX_INPUT_CHARS:
            raise HTTPException(422, f"text must be 1-{MAX_INPUT_CHARS} characters")
        try:
            result = rt.model.parse(text)
        except Exception:
            log.exception("parse failed for %s", bot)
            return JSONResponse({"detail": "internal error"}, status_code=500)
        return {"bot": bot, "text": text, **result.to_log()}

    @app.get("/v1/builder/bots/{bot}/misunderstood", dependencies=[Depends(admin)], response_model=MisunderstoodOut)
    def misunderstood(bot: str = BOT_ID, limit: int = 50) -> dict[str, Any]:
        """Recent messages the bot could not handle — candidates for new examples."""
        runtime(bot)
        limit = max(1, min(limit, 500))
        items = []
        for session_key, event in host.store.recent(f"{bot}:", "commands_received", limit * 10):
            nlu = getattr(event, "nlu", {}) or {}
            if nlu.get("rule") in MISUNDERSTOOD_RULES and nlu.get("text"):
                items.append({"session": session_key.split(":", 1)[1], "text": nlu["text"], "rule": nlu["rule"],
                              "intents": nlu.get("intents", [])})
            if len(items) >= limit:
                break
        return {"bot": bot, "items": items}

    if isinstance(host, BotHost):
        runner = jobs or JobRunner()
        _builder_routes(app, host, admin, runner)
        if jobs is None:  # ours: finish queued jobs on shutdown
            app.router.on_shutdown.append(runner.close)
    return app


def run_turn(rt: BotRuntime, key: str, text: str) -> TurnResult | JSONResponse:
    try:
        return rt.processor.handle(key, text)
    except SessionBusy:
        return JSONResponse({"detail": "this session is still answering the previous message"}, status_code=409)
    except UNAVAILABLE:  # database/network down: worth retrying
        log.exception("storage unavailable for %s", key)
        return JSONResponse({"detail": "temporarily unavailable, please retry"}, status_code=503)
    except Exception:  # a bug (e.g. an inconsistent event log): retrying will not help
        log.exception("turn failed for %s", key)
        return JSONResponse({"detail": "internal error"}, status_code=500)


def messages_out(result: TurnResult) -> list[MessageOut]:
    return [MessageOut(response=m.response, text=m.text, quick_replies=list(m.quick_replies)) for m in result.messages]


def etag(revision: int) -> str:
    return f'"{revision}"'


def parse_if_match(value: str | None) -> int | None:
    """The revision in an If-Match header (`"3"` or `W/"3"`), None when absent; 400 when malformed."""
    if value is None:
        return None
    tag = value.strip().removeprefix("W/").strip('"')
    if not tag.isdigit():
        raise HTTPException(400, "If-Match must be the draft's ETag")
    return int(tag)


def _builder_routes(app: FastAPI, host: BotHost, admin: Any, jobs: JobRunner) -> None:
    deps = [Depends(admin)]

    def known(bot: str) -> None:
        if bot not in host.repository.bots():
            raise HTTPException(404, f"unknown bot {bot!r}")

    @app.get("/v1/builder/bots", dependencies=deps, response_model=BotListOut)
    def list_bots() -> Any:
        """Every bot in the repository, with or without a deployment (the builder's bot list)."""
        live = host.bots()
        items = []
        for bot in host.repository.bots():
            draft = host.repository.draft(bot)
            history = host.repository.history(bot)
            current = history[0] if history else None
            label = (draft.document.get("label") if draft is not None else None) or (
                host.repository.get_version(bot, current.version).get("label") if current is not None else None)
            try:
                draft_version = document_version(draft.document) if draft is not None else None
            except BotError:  # not even plain JSON: the draft cannot become a version as it is
                draft_version = None
            items.append({
                "id": bot, "label": label or "",
                "live": current.version if current is not None else None,
                "live_number": current.number if current is not None else None,
                "serving": live.get(bot),  # what this server answers with (None: not loaded here)
                "draft_revision": draft.revision if draft is not None else None,
                # the draft differs from what is live: there is something to deploy
                "undeployed_changes": draft is not None and draft_version != (current.version if current else None),
            })
        return {"bots": items}

    @app.get("/v1/builder/bots/{bot}/draft", dependencies=deps,
             responses=DRAFT_RESPONSE)
    def get_draft(response: Response, bot: str = BOT_ID) -> Any:
        known(bot)
        draft = host.repository.draft(bot)
        if draft is None:
            raise HTTPException(404, f"{bot!r} has no draft")
        response.headers["ETag"] = etag(draft.revision)
        return draft.document

    @app.put("/v1/builder/bots/{bot}/draft", dependencies=deps, response_model=DraftSavedOut,
             openapi_extra=DRAFT_BODY)
    def put_draft(document: dict[str, Any], response: Response, bot: str = BOT_ID,
                  if_match: str | None = Header(default=None)) -> Any:
        """Saving over an existing draft needs its ETag in If-Match (428 without, 412 when it moved on)."""
        expected = parse_if_match(if_match)
        if expected is None:
            if host.repository.draft(bot) is not None:
                raise HTTPException(428, "send the draft's ETag in If-Match")
            expected = 0  # create: must still not exist when we write
        try:
            check = host.save_draft(bot, document, expected)
        except BotError as exc:
            raise HTTPException(422, exc.problems) from exc
        except RevisionConflict as exc:
            return JSONResponse({"detail": "the draft was changed elsewhere; reload it", "revision": exc.current},
                                status_code=412)
        response.headers["ETag"] = etag(check.revision)
        return {"bot": bot, "version": check.version, "revision": check.revision, "valid": not check.errors,
                "errors": check.errors, "warnings": check.warnings}

    @app.post("/v1/builder/bots/{bot}/versions", dependencies=deps, status_code=202, response_model=JobOut)
    def train_version(bot: str = BOT_ID) -> Any:
        """Start training the draft as a deployable version; follow it at /v1/builder/jobs/{id}."""
        known(bot)

        def run() -> dict[str, Any]:
            try:
                info, report = host.train_draft(bot)
            except BotError as exc:
                raise JobFailed("invalid_draft", {"detail": "the draft is invalid", "errors": exc.problems}) from exc
            except BelowBar as exc:
                raise JobFailed("below_bar", {"detail": str(exc), "per_intent_f1": exc.report.per_intent_f1}) from exc
            except TrainingError as exc:  # e.g. an intent with too few examples to evaluate
                raise JobFailed("training", {"detail": str(exc)}) from exc
            except NoDraft as exc:
                raise JobFailed("no_draft", {"detail": f"{bot!r} has no draft"}) from exc
            return {"version": info.version, "number": info.number, "macro_f1": report.macro_f1,
                    "clarify_rate": report.clarify_rate, "per_intent_f1": report.per_intent_f1,
                    "warnings": report.warnings}

        return jobs.submit("version", bot, run).to_dict()

    @app.post("/v1/builder/bots/{bot}/test-builds", dependencies=deps, status_code=202, response_model=JobOut)
    def test_build(bot: str = BOT_ID) -> Any:
        """Fit the draft for the test chat (no evaluation, not deployable); follow the job."""
        known(bot)

        def run() -> dict[str, Any]:
            try:
                return {"build": host.build_test(bot)}
            except BotError as exc:
                raise JobFailed("invalid_draft", {"detail": "the draft is invalid", "errors": exc.problems}) from exc
            except TrainingError as exc:  # fewer than two intents, nothing in scope
                raise JobFailed("training", {"detail": str(exc)}) from exc
            except NoDraft as exc:
                raise JobFailed("no_draft", {"detail": f"{bot!r} has no draft"}) from exc

        return jobs.submit("test_build", bot, run, urgent=True).to_dict()  # someone waits at the test chat

    @app.get("/v1/builder/jobs/{job_id}", dependencies=deps, response_model=JobOut)
    def get_job(job_id: str = JOB_ID) -> Any:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job (jobs are forgotten when the server restarts)")
        return job

    @app.get("/v1/builder/jobs/{job_id}/events", dependencies=deps)
    async def job_events(request: Request, job_id: str = JOB_ID, wait: float = 15.0) -> Any:
        """Server-sent events: the job as JSON on every change, until it is done or failed.
        A comment line every `wait` seconds keeps proxies from closing an idle stream. The
        stream stops within a second of the client going away (no thread is held meanwhile)."""
        if jobs.get(job_id) is None:
            raise HTTPException(404, "unknown job")
        wait = max(0.1, min(wait, 60.0))

        async def stream() -> AsyncIterator[str]:
            seen = -1
            deadline = last = time.monotonic()
            deadline += 600
            while time.monotonic() < deadline and not await request.is_disconnected():
                got = await asyncio.to_thread(jobs.wait, job_id, seen, min(wait, 1.0))
                if got is None:
                    return
                job, revision = got
                if revision == seen:
                    if time.monotonic() - last >= wait:
                        last = time.monotonic()
                        yield ": still waiting\n\n"
                    continue
                seen, last = revision, time.monotonic()
                yield f"data: {json.dumps(job, ensure_ascii=False)}\n\n"
                if job["status"] in ("done", "failed"):
                    return

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/v1/builder/bots/{bot}/test/{build}/sessions/{session}/messages", dependencies=deps,
              response_model=TestTurnOut)
    def test_message(body: MessageIn, bot: str = BOT_ID, build: str = VERSION, session: str = SESSION_ID) -> Any:
        """Chat with a test build. 410 once it expired or was replaced: never falls back to the live bot."""
        rt = host.test_runtime(bot, build)
        if rt is None:
            raise HTTPException(410, "this test build expired; save the draft again")
        key = f"{bot}{TEST_SESSION_SEP}{session}"
        result = run_turn(rt, key, body.text)
        if isinstance(result, JSONResponse):
            return result
        events = result.events
        state = fold(host.store.events(key))
        return {
            "bot": bot, "build": build, "session": session, "messages": [m.model_dump() for m in messages_out(result)],
            "nlu": next((e.nlu for e in reversed(events) if e.type == "commands_received"), None),
            "actions": [{"scenario": e.scenario, "action": e.action, "ok": e.type == "action_executed"}
                        for e in events if e.type in ("action_executed", "action_failed")],
            "stack": [{"scenario": f.scenario, "step": f.step, "awaiting": f.awaiting, "slots": f.slots}
                      for f in state.stack],
        }

    @app.get("/v1/builder/bots/{bot}/versions", dependencies=deps, response_model=VersionsOut)
    def list_versions(bot: str = BOT_ID) -> Any:
        known(bot)
        history = host.repository.history(bot)
        live = history[0].version if history else None
        return {"bot": bot, "live": live,
                "versions": [{"version": v.version, "number": v.number, "created_at": v.created_at,
                              "macro_f1": v.macro_f1, "live": v.version == live}
                             for v in host.repository.versions(bot)],
                "history": [{"version": d.version, "number": d.number, "note": d.note, "deployed_at": d.deployed_at}
                            for d in history]}

    @app.post("/v1/builder/bots/{bot}/deploy", dependencies=deps, response_model=LiveOut)
    def deploy(body: DeployIn, bot: str = BOT_ID) -> Any:
        known(bot)
        with deploy_errors(body.version):
            host.deploy(bot, body.version, body.note.strip())
        return {"bot": bot, "live": body.version}

    @app.post("/v1/builder/bots/{bot}/rollback", dependencies=deps, response_model=LiveOut)
    def rollback(bot: str = BOT_ID) -> Any:
        known(bot)
        with deploy_errors("the previous version"):
            version = host.rollback(bot)
        return {"bot": bot, "live": version}


@contextmanager
def deploy_errors(what: str) -> Iterator[None]:
    try:
        yield
    except NoEarlierDeployment as exc:
        raise HTTPException(409, "there is no earlier deployment to go back to") from exc
    except VersionNotFound as exc:
        raise HTTPException(404, f"unknown version: {what}") from exc
    except NotTrained as exc:
        raise HTTPException(409, f"{what} has no trained model; train it first") from exc
    except BotError as exc:  # stored, but no longer valid here (e.g. an action was removed)
        raise HTTPException(409, {"detail": f"{what} cannot run here", "errors": exc.problems}) from exc


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - starts a server
    import os

    import uvicorn

    from chatcore.host import BotHost
    from chatcore.nlu.model import Embedder, SentenceEmbedder
    from chatcore.runtime import default_repository, default_store
    from chatcore.serving import BatchedEmbedder

    parser = argparse.ArgumentParser(description="Serve chatcore bots over HTTP.")
    parser.add_argument("--bot", type=Path, action="append", default=[],
                        help="bot directory to import on first start (repeatable); "
                             "later edits go through the builder API")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--embedder", choices=("sentence", "hashing"), default="sentence",
                        help="hashing: a fast stand-in for tests and CI (use a separate --artifacts directory)")
    parser.add_argument("--min-f1", type=float, default=0.90,
                        help="macro-F1 a trained version needs to be deployable (the hashing embedder needs 0)")
    parser.add_argument("--keep-warm-ms", type=float, default=None,
                        help="when idle, run the model this often so the first request after a pause stays fast "
                             "(keeps the GPU busy; see docs/benchmarks/api_latency.md)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.embedder == "hashing":  # tests / CI / demos: instant, no model, NOT semantic
        from chatcore.nlu.hashing import HashingEmbedder

        log.warning("--embedder hashing: matching by character n-grams only; never serve real users with it")
        inner: Embedder = HashingEmbedder()
    else:
        inner = SentenceEmbedder()
    embedder = BatchedEmbedder(inner, keepalive_ms=args.keep_warm_ms)
    host = BotHost(default_repository(), default_store(), embedder, args.artifacts, bulk_embedder=embedder.bulk(),
                   min_f1=args.min_f1)
    host.start()
    for directory in args.bot:
        host.seed(directory)
    if not host.bots():
        parser.error("no deployed bots: pass --bot <directory>")
    for bot_id in host.bots():  # load the model weights before the first user waits for them
        runtime = host.runtime(bot_id)
        assert runtime is not None
        runtime.model.parse("안녕하세요")
    token = os.environ.get("CHATCORE_ADMIN_TOKEN") or None
    if token is None and args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("CHATCORE_ADMIN_TOKEN is not set: anyone who can reach %s can edit and deploy bots "
                    "and read users' messages through /v1/builder", args.host)
    uvicorn.run(create_app(host, token), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
