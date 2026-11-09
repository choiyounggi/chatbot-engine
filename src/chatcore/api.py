"""HTTP API (docs/engine-core-plan.md Phase 5).

    POST /v1/bots/{bot}/sessions/{session}/messages   {"text": "..."} -> bot messages
    POST /v1/bots/{bot}/nlu/parse                     {"text": "..."} -> NLU result only
    GET  /healthz

Run: `make serve` (uv run python -m chatcore.api --bot bots/sample).
Errors never carry exception text: 404 unknown bot, 409 the session is mid-turn,
422 malformed request, 503 storage unavailable.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from chatcore.bot.loader import BotError
from chatcore.bot.repository import VersionNotFound
from chatcore.dialogue.turn import MAX_INPUT_CHARS
from chatcore.host import BelowBar, BotHost, Host, NoEarlierDeployment, NotTrained
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


MISUNDERSTOOD_RULES = ("fallback", "clarify")


def create_app(host: Host, admin_token: str | None = None) -> FastAPI:
    """`admin_token`: when set, builder endpoints require `Authorization: Bearer <token>`."""
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
        try:
            result = rt.processor.handle(key, body.text)
        except SessionBusy:
            return JSONResponse({"detail": "this session is still answering the previous message"}, status_code=409)
        except UNAVAILABLE:  # database/network down: worth retrying
            log.exception("storage unavailable for %s/%s", bot, session)
            return JSONResponse({"detail": "temporarily unavailable, please retry"}, status_code=503)
        except Exception:  # a bug (e.g. an inconsistent event log): retrying will not help
            log.exception("turn failed for %s/%s", bot, session)
            return JSONResponse({"detail": "internal error"}, status_code=500)
        nlu = next((e.nlu for e in reversed(result.events) if e.type == "commands_received"), None)
        turn_log.info(json.dumps({
            "bot": bot, "version": rt.bot.version, "session": session,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "rule": (nlu or {}).get("rule"), "intent": ((nlu or {}).get("intents") or [[None]])[0][0],
            "responses": [m.response for m in result.messages],
        }, ensure_ascii=False))
        return TurnOut(bot=bot, session=session, messages=[MessageOut(response=m.response, text=m.text,
                                                                      quick_replies=list(m.quick_replies))
                                                           for m in result.messages],
                       nlu=nlu if debug else None)

    @app.post("/v1/bots/{bot}/nlu/parse")
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

    @app.get("/v1/builder/bots/{bot}/misunderstood", dependencies=[Depends(admin)])
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
        _builder_routes(app, host, admin)
    return app


def _builder_routes(app: FastAPI, host: BotHost, admin: Any) -> None:
    deps = [Depends(admin)]

    def known(bot: str) -> None:
        if bot not in host.repository.bots():
            raise HTTPException(404, f"unknown bot {bot!r}")

    @app.get("/v1/builder/bots/{bot}/draft", dependencies=deps)
    def get_draft(bot: str = BOT_ID) -> Any:
        known(bot)
        return host.repository.get_draft(bot)

    @app.put("/v1/builder/bots/{bot}/draft", dependencies=deps)
    def put_draft(document: dict[str, Any], bot: str = BOT_ID) -> Any:
        try:
            check = host.save_draft(bot, document)
        except BotError as exc:
            raise HTTPException(422, exc.problems) from exc
        return {"bot": bot, "version": check.version, "valid": not check.errors,
                "errors": check.errors, "warnings": check.warnings}

    @app.post("/v1/builder/bots/{bot}/versions", dependencies=deps)
    def train_version(bot: str = BOT_ID) -> Any:
        known(bot)
        try:
            info, report = host.train_draft(bot)
        except BotError as exc:
            return JSONResponse({"detail": "the draft is invalid", "errors": exc.problems}, status_code=422)
        except BelowBar as exc:
            return JSONResponse({"detail": str(exc), "per_intent_f1": exc.report.per_intent_f1}, status_code=422)
        except TrainingError as exc:  # e.g. an intent with too few examples to evaluate
            return JSONResponse({"detail": str(exc)}, status_code=422)
        return {"bot": bot, "version": info.version, "macro_f1": report.macro_f1, "clarify_rate": report.clarify_rate,
                "per_intent_f1": report.per_intent_f1, "warnings": report.warnings}

    @app.get("/v1/builder/bots/{bot}/versions", dependencies=deps)
    def list_versions(bot: str = BOT_ID) -> Any:
        known(bot)
        history = host.repository.deployments(bot)
        live = history[0] if history else None
        return {"bot": bot, "live": live, "versions": [
            {"version": v.version, "created_at": v.created_at, "macro_f1": v.macro_f1, "live": v.version == live}
            for v in host.repository.versions(bot)]}

    @app.post("/v1/builder/bots/{bot}/deploy", dependencies=deps)
    def deploy(body: DeployIn, bot: str = BOT_ID) -> Any:
        known(bot)
        with deploy_errors(body.version):
            host.deploy(bot, body.version)
        return {"bot": bot, "live": body.version}

    @app.post("/v1/builder/bots/{bot}/rollback", dependencies=deps)
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
    from chatcore.nlu.model import SentenceEmbedder
    from chatcore.runtime import default_repository, default_store
    from chatcore.serving import BatchedEmbedder

    parser = argparse.ArgumentParser(description="Serve chatcore bots over HTTP.")
    parser.add_argument("--bot", type=Path, action="append", default=[],
                        help="bot directory to import on first start (repeatable); "
                             "later edits go through the builder API")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--keep-warm-ms", type=float, default=None,
                        help="when idle, run the model this often so the first request after a pause stays fast "
                             "(keeps the GPU busy; see docs/benchmarks/api_latency.md)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    embedder = BatchedEmbedder(SentenceEmbedder(), keepalive_ms=args.keep_warm_ms)
    host = BotHost(default_repository(), default_store(), embedder, args.artifacts)
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
