"""HTTP API (docs/engine-core-plan.md Phase 5).

    POST /v1/bots/{bot}/sessions/{session}/messages   {"text": "...", "block_id"?: "..."} -> bot messages
    POST /v1/bots/{bot}/sessions/{session}/open       the user opened the chat -> welcome messages
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
import datetime as dt
import hmac
import json
import logging
import re
import sys
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from chatcore.actions.builtin import KST
from chatcore.analytics import ACTIVITY_TYPES, MAX_DAYS, SESSION_GAP, compute_report, start_of
from chatcore.bot.loader import BotError, document_version
from chatcore.bot.repository import RevisionConflict, VersionNotFound
from chatcore.bot.schema import BasicCard, BotDocument, Carousel, ListCard, SimpleImage
from chatcore.dialogue.state import fold
from chatcore.dialogue.turn import MAX_INPUT_CHARS, TurnResult, UnknownBlock
from chatcore.generation import GENERATION_DEPTH, LLM_TIMEOUT, GenerationError, Generator, LlmClient, LlmConfig
from chatcore.host import (
    AlreadyDeleting,
    BadExample,
    BelowBar,
    BotDeleting,
    BotExists,
    BotHost,
    CopySourceNotFound,
    ExampleTaken,
    Host,
    NoDeletionPending,
    NoDraft,
    NoEarlierDeployment,
    NoKnowledgeBase,
    NotTrained,
    unique_sentences,
)
from chatcore.jobs import JobFailed, JobRunner
from chatcore.knowledge import KnowledgeBase, KnowledgeError, extract_text
from chatcore.nlu.train import TrainingError
from chatcore.runtime import BotRuntime
from chatcore.store.base import SessionBusy
from chatcore.training_queue import RuleFilter, StatusFilter, group
from chatcore.training_queue import key as queue_key
from chatcore.web import OriginCache, WebSettings, load, public

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
UTTERANCE_SCAN = 20_000  # user messages the training queue reads per request (newest first)
TEST_SESSION_SEP = "~"  # test-build sessions are "<bot>~<session>": never "<bot>:...", so no stats or misunderstood


class MessageIn(BaseModel):
    text: str = Field(max_length=10_000)  # the engine itself answers input_invalid above 1,000 characters
    # the user pressed a quick reply / button with action "block": start this scenario without understanding
    block_id: str | None = Field(None, pattern=r"^[a-z][a-z0-9_]{0,63}$")


class QuickReplyOut(BaseModel):
    """A quick reply as said (not length-checked: a clarify label or a filled-in template may be long)."""

    label: str
    action: Literal["message", "block"]
    message_text: str | None
    block_id: str | None


class MessageOut(BaseModel):
    response: str
    text: str  # the text; for a rich message a plain-text summary
    quick_replies: list[str]  # labels (quick_reply_items has their actions)
    type: Literal["text", "simple_image", "basic_card", "list_card", "carousel"] = "text"
    simple_image: SimpleImage | None = None
    basic_card: BasicCard | None = None
    list_card: ListCard | None = None
    carousel: Carousel | None = None
    quick_reply_items: list[QuickReplyOut] = Field(default_factory=list)


class TurnOut(BaseModel):
    bot: str
    session: str
    messages: list[MessageOut]
    nlu: dict[str, Any] | None = None


class DeployIn(BaseModel):
    version: str = Field(pattern=r"^[0-9a-f]{12}$")
    note: str = Field("", max_length=100)  # what changed, shown in the deploy history
    # omitted: keep the knowledge base that is live now; null: none
    knowledge_version: str | None = Field(None, pattern=r"^[0-9a-f]{12}$")


class ParseOut(BaseModel):
    bot: str
    text: str
    intents: list[tuple[str, float]]  # top 3: (intent, probability)
    similarity: float  # cosine to the nearest example of any intent
    matched_by: Literal["exact", "keyword", "classifier"]
    entities: list[dict[str, Any]]


class MisunderstoodItem(BaseModel):
    session: str
    text: str
    rule: str
    intents: list[tuple[str, float]]


class MisunderstoodOut(BaseModel):
    bot: str
    items: list[MisunderstoodItem]


class BotDeletionOut(BaseModel):
    purge_after: str  # ISO 8601 UTC: permanently deleted at this time unless restored first


class BotSummaryOut(BaseModel):
    id: str
    label: str
    live: str | None  # deployed version
    live_number: int | None
    serving: str | None  # version this server answers with (None: not loaded here)
    draft_revision: int | None
    undeployed_changes: bool  # the draft differs from what is live
    deletion: BotDeletionOut | None = None  # waiting to be deleted (None: not waiting)


class BotListOut(BaseModel):
    bots: list[BotSummaryOut]


class BotCreateIn(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    label: str = Field(min_length=1, max_length=100)
    copy_from: str | None = Field(None, pattern=r"^[a-z][a-z0-9_]{0,63}$")


class BotCreatedOut(BaseModel):
    bot: str
    revision: int


class BotDeletedOut(BaseModel):
    bot: str
    purge_after: str  # ISO 8601 UTC


class BotRestoredOut(BaseModel):
    bot: str
    live: str | None  # the deployed version now serving again (None: the bot was never deployed)


class DraftSavedOut(BaseModel):
    bot: str
    version: str | None  # the draft's content hash; None when it does not parse
    revision: int
    valid: bool
    errors: list[str]
    warnings: list[str]


class JobOut(BaseModel):
    id: str
    kind: str  # "version" | "test_build" | "knowledge"
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
    knowledge_version: str | None = None


class VersionsOut(BaseModel):
    bot: str
    live: str | None
    versions: list[VersionOut]
    history: list[DeploymentOut]


class KbSavedOut(BaseModel):
    bot: str
    version: str | None  # None when the knowledge base is invalid
    revision: int
    valid: bool
    errors: list[str]


class KbVersionOut(BaseModel):
    version: str
    number: int
    created_at: str
    faq: int
    chunks: int
    live: bool


class KbVersionsOut(BaseModel):
    bot: str
    live: str | None  # the knowledge-base version deployed now
    versions: list[KbVersionOut]


class KbEntryOut(BaseModel):
    id: str
    kind: Literal["faq", "doc"]
    title: str  # FAQ question, or "document - section"
    body: str
    source: str  # the document's name (chunks)


class KbEntriesOut(BaseModel):
    bot: str
    version: str
    entries: list[KbEntryOut]


class KbSearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    version: str | None = Field(None, pattern=r"^[0-9a-f]{12}$")  # default: the newest built version


class KbHitOut(BaseModel):
    id: str
    kind: Literal["faq", "doc"]
    title: str
    cosine: float
    score: float  # the judge's
    passed: bool


class KbReplyOut(BaseModel):
    label: str
    message: str


class KbAnswerOut(BaseModel):
    text: str
    replies: list[KbReplyOut]
    items: list[str]


class KbSearchOut(BaseModel):
    bot: str
    version: str
    judge: str
    threshold: float
    exact: bool  # the query was an entry's title
    answer: KbAnswerOut | None  # what a conversation would say (None: nothing passed)
    candidates: list[KbHitOut]  # best first
    generated: KbGeneratedOut | None = None  # the LLM's attempt, when the settings ask for generated answers


class KbGeneratedOut(BaseModel):
    model: str
    ms: int
    used: list[str]  # entry ids given to the model
    outcome: Literal["generated", "no_answer", "unsupported_number", "too_long", "empty", "timeout", "error"]


class LlmOut(BaseModel):
    configured: bool  # the server was started with --llm-url
    model: str | None = None
    ok: bool | None = None  # a short test answer came back
    ms: int | None = None
    error: str | None = None


class ExtractOut(BaseModel):
    text: str


class AnalyticsTotalsOut(BaseModel):
    users: int
    new_users: int
    returning_users: int
    sessions: int
    messages: int
    block_calls: int
    fallbacks: int
    knowledge_answers: int
    knowledge_rate: float | None


class AnalyticsDayOut(BaseModel):
    date: dt.date
    users: int
    new_users: int
    returning_users: int
    sessions: int
    messages: int
    block_calls: int
    fallbacks: int
    knowledge_answers: int


class AnalyticsBlockOut(BaseModel):
    block: str
    label: str
    calls: int


class AnalyticsRetentionOut(BaseModel):
    date: dt.date
    new_users: int
    d1: float | None
    d7: float | None


class AnalyticsOut(BaseModel):
    bot: str
    from_date: dt.date = Field(alias="from")
    to: dt.date
    timezone: str = "Asia/Seoul"
    totals: AnalyticsTotalsOut
    days: list[AnalyticsDayOut]
    blocks: list[AnalyticsBlockOut]
    retention: list[AnalyticsRetentionOut]


class LiveOut(BaseModel):
    bot: str
    live: str
    knowledge_version: str | None = None  # the knowledge base deployed with it (None: none)


class ChangeOut(BaseModel):
    area: Literal["bot", "settings", "system", "block", "intent", "entity", "response", "skill"]
    id: str
    label: str
    action: Literal["added", "removed", "changed"]
    examples_added: int | None = None  # intents only
    examples_removed: int | None = None


class EditOut(BaseModel):
    revision: int
    saved_at: str
    changes: list[ChangeOut]


class EditsOut(BaseModel):
    bot: str
    items: list[EditOut]  # newest first


class UtteranceOut(BaseModel):
    key: str  # the sentence as grouped; send `text` back to act on it
    text: str
    count: int
    sessions: int
    first_seen: str
    last_seen: str
    rule: str  # how the latest one was handled (fallback, clarify, intent, slot, ...)
    intent: str | None  # the classifier's top intent the latest time
    score: float | None
    status: Literal["pending", "ignored", "added"]
    status_intent: str | None


class UtterancesOut(BaseModel):
    bot: str
    scanned: int  # user messages read
    truncated: bool  # more messages exist in the period than were read (narrow the period)
    items: list[UtteranceOut]


QueueText = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class SentencesIn(BaseModel):
    texts: list[Annotated[str, QueueText]] = Field(min_length=1, max_length=200)


class AddExamplesIn(SentencesIn):
    intent: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)


class MarkedOut(BaseModel):
    bot: str
    marked: int


class MarkOut(BaseModel):
    key: str
    text: str
    status: Literal["pending", "ignored", "added"]
    intent: str | None
    marked_at: str


class MarksOut(BaseModel):
    bot: str
    items: list[MarkOut]  # newest first


class SkillErrorOut(BaseModel):
    at: str
    skill: str
    kind: Literal["blocked", "network", "timeout", "http_status", "invalid_response", "too_large", "secret_missing"]
    status: int | None
    detail: str
    ms: int
    test: bool


class SkillErrorsOut(BaseModel):
    bot: str
    items: list[SkillErrorOut]  # newest first, at most 100


class SkillTestIn(BaseModel):
    params: dict[str, Annotated[str, Field(max_length=MAX_INPUT_CHARS)]] = Field(default_factory=dict, max_length=50)
    utterance: str = Field("", max_length=MAX_INPUT_CHARS)


class SkillTestOut(BaseModel):
    ok: bool
    status: int | None
    ms: int
    result: dict[str, Any] | None
    error: SkillErrorOut | None


class SecretOut(BaseModel):
    name: str
    updated_at: str


class SecretsOut(BaseModel):
    bot: str
    items: list[SecretOut]  # names only: values are never sent back


class WebOut(WebSettings):
    bot: str


class WebPublicOut(BaseModel):
    bot: str
    title: str
    color: str
    placeholder: str


# the chat endpoints a browser may call from an allowed site (docs/engine-core-plan.md "빌더 단계 E-G")
CORS_PATH = re.compile(r"^/v1/bots/([a-z][a-z0-9_]{0,63})/(?:web|sessions/[A-Za-z0-9_-]{1,63}/(?:open|messages))$")
PREFLIGHT_HEADERS = {"Access-Control-Allow-Methods": "GET, POST", "Access-Control-Allow-Headers": "Content-Type",
                     "Access-Control-Max-Age": "600"}


class SecretIn(BaseModel):
    value: str = Field(min_length=1, max_length=4000)


SECRET = PathParam(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
SKILL_ID = PathParam(pattern=r"^[a-z][a-z0-9_]{0,63}$")


# The draft is stored as written (even when invalid), so it is described with the bot
# document's schema but not validated against it.
DOCUMENT_SCHEMA = {"$ref": "#/components/schemas/BotDocument"}
DRAFT_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {"model": BotDocument, "description": "the draft as stored (may be invalid); ETag header"}}
DRAFT_BODY: dict[str, Any] = {
    "requestBody": {"required": True, "content": {"application/json": {"schema": DOCUMENT_SCHEMA}}}}

KB_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {"model": KnowledgeBase, "description": "the knowledge-base draft as stored (may be invalid); ETag header"}}
KB_BODY: dict[str, Any] = {
    "requestBody": {"required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/KnowledgeBase"}}}}}
EXTRACT_BODY: dict[str, Any] = {
    "requestBody": {"required": True, "description": "the file itself (at most 10 MB)",
                    "content": {t: {"schema": {"type": "string", "format": "binary"}}
                                for t in ("application/pdf", "text/plain", "text/markdown")}}}
MAX_UPLOAD = 10 * 1024 * 1024

MISUNDERSTOOD_RULES = ("fallback", "clarify")
EARLIEST, LATEST = dt.date(2000, 1, 1), dt.date(2100, 12, 31)  # analytics periods


def create_app(host: Host, admin_token: str | None = None, jobs: JobRunner | None = None) -> FastAPI:
    """`admin_token`: when set, builder endpoints require `Authorization: Bearer <token>`.
    `jobs`: where builder training runs (a new JobRunner when the host is a BotHost)."""
    app = FastAPI(title="chatcore", version="0.0.1")

    @app.exception_handler(BotDeleting)
    async def bot_deleting(request: Request, exc: BotDeleting) -> JSONResponse:
        # any route or job step that reaches a bot waiting to be deleted (BotHost guards writes, training,
        # deploys and test chats; known() already answers most builder routes the same way)
        return JSONResponse({"detail": "the bot is waiting to be deleted; restore it first"}, status_code=409)

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

    def stored_web(bot: str) -> WebSettings | None:
        repository = getattr(host, "repository", None)  # a StaticHost has none: no site is allowed
        return load(repository.web_settings(bot)) if repository is not None else None

    origins = OriginCache(stored_web)

    @app.middleware("http")
    async def cors(request: Request, call_next: Any) -> Any:
        """Browsers may call a bot's chat endpoints only from the sites its web settings allow."""
        origin = request.headers.get("origin")
        path = request.scope["path"]
        root = request.scope.get("root_path", "")
        if root and path.startswith(root):  # behind a path prefix: match what the router matches
            path = path[len(root):]
        match = CORS_PATH.match(path)
        if origin is None or match is None:
            return await call_next(request)
        bot, vary = match.group(1), {"Vary": "Origin"}
        if host.runtime(bot) is None:  # not served here: no settings read, nothing cached
            return JSONResponse({"detail": f"unknown bot {bot!r}"}, status_code=404, headers=vary)
        try:
            allowed = await run_in_threadpool(origins.allowed, bot, origin)
        except Exception:
            log.exception("could not read the web settings of %s", bot)
            return JSONResponse({"detail": "temporarily unavailable, please retry"}, status_code=503, headers=vary)
        if not allowed:
            return JSONResponse({"detail": "this site may not talk to this bot"}, status_code=403, headers=vary)
        headers = {"Access-Control-Allow-Origin": origin, **vary}
        if request.method == "OPTIONS":
            return Response(status_code=204, headers={**headers, **PREFLIGHT_HEADERS})
        try:
            response = await call_next(request)
        except Exception:  # the widget should see a 500, not an unreadable network error
            log.exception("unhandled error for %s", path)
            return JSONResponse({"detail": "internal error"}, status_code=500, headers=headers)
        response.headers.update(headers)
        return response

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
        result = run_turn(rt, key, body.text, body.block_id)
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

    @app.post("/v1/bots/{bot}/sessions/{session}/open", response_model=TurnOut)
    def open_session(bot: str = BOT_ID, session: str = SESSION_ID) -> Any:
        """The user opened the conversation: a new session answers with the welcome block (if any);
        an existing one with no messages."""
        key = f"{bot}:{session}"
        try:
            rt = host.runtime_for_session(bot, key)
        except UNAVAILABLE:
            log.exception("storage unavailable for %s/%s", bot, session)
            return JSONResponse({"detail": "temporarily unavailable, please retry"}, status_code=503)
        if rt is None:
            raise HTTPException(404, f"unknown bot {bot!r}")
        result = run_turn(rt, key, None)
        if isinstance(result, JSONResponse):
            return result
        return TurnOut(bot=bot, session=session, messages=messages_out(result))

    @app.get("/v1/bots/{bot}/web", response_model=WebPublicOut)
    def web_public(bot: str = BOT_ID) -> Any:
        """How the chat widget looks for this bot (never which sites are allowed)."""
        rt = runtime(bot)
        return public(stored_web(bot) or WebSettings(), bot, rt.bot.label)

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
        if isinstance(host, BotHost) and bot in host.repository.deletions():
            raise HTTPException(409, "the bot is waiting to be deleted; restore it first")
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
        _builder_routes(app, host, admin, runner, origins)
        if jobs is None:  # ours: finish queued jobs on shutdown
            app.router.on_shutdown.append(runner.close)
    return app


def run_turn(rt: BotRuntime, key: str, text: str | None, block: str | None = None) -> TurnResult | JSONResponse:
    """One message (or, with text None, opening the session) with the API's error mapping."""
    try:
        return rt.processor.open(key) if text is None else rt.processor.handle(key, text, block)
    except UnknownBlock:
        return JSONResponse({"detail": "block_id is not a block any response of this bot offers"}, status_code=422)
    except SessionBusy:
        return JSONResponse({"detail": "this session is still answering the previous message"}, status_code=409)
    except UNAVAILABLE:  # database/network down: worth retrying
        log.exception("storage unavailable for %s", key)
        return JSONResponse({"detail": "temporarily unavailable, please retry"}, status_code=503)
    except Exception:  # a bug (e.g. an inconsistent event log): retrying will not help
        log.exception("turn failed for %s", key)
        return JSONResponse({"detail": "internal error"}, status_code=500)


def messages_out(result: TurnResult) -> list[MessageOut]:
    out = []
    for m in result.messages:
        items = [QuickReplyOut(label=r.label, action=r.action, message_text=r.message_text, block_id=r.block_id)
                 for r in m.replies]
        message = MessageOut(response=m.response, text=m.text, quick_replies=list(m.quick_replies),
                             quick_reply_items=items)
        if m.rich is not None:
            message = message.model_copy(update={"type": m.kind, m.kind: m.rich})
        out.append(message)
    return out


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


def _builder_routes(app: FastAPI, host: BotHost, admin: Any, jobs: JobRunner, origins: OriginCache) -> None:
    deps = [Depends(admin)]

    def known(bot: str) -> None:
        if bot in host.repository.deletions():
            raise HTTPException(409, "the bot is waiting to be deleted; restore it first")
        if bot not in host.repository.bots():
            raise HTTPException(404, f"unknown bot {bot!r}")

    @app.get("/v1/builder/bots", dependencies=deps, response_model=BotListOut)
    def list_bots() -> Any:
        """Every bot in the repository, with or without a deployment (the builder's bot list)."""
        live = host.bots()
        deletions = host.repository.deletions()
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
                "deletion": {"purge_after": deletions[bot]} if bot in deletions else None,
            })
        return {"bots": items}

    @app.post("/v1/builder/bots", dependencies=deps, status_code=201, response_model=BotCreatedOut)
    def create_bot(body: BotCreateIn) -> Any:
        """A new draft (not deployed): empty, or (with copy_from) a copy of another bot's draft."""
        try:
            check = host.create_bot(body.id, body.label, body.copy_from)
        except BotExists:
            return JSONResponse({"detail": f"bot {body.id!r} already exists or is waiting to be deleted"},
                                status_code=409)
        except CopySourceNotFound as exc:
            raise HTTPException(404, f"unknown bot {body.copy_from!r}") from exc
        return {"bot": body.id, "revision": check.revision}

    @app.delete("/v1/builder/bots/{bot}", dependencies=deps, status_code=202, response_model=BotDeletedOut)
    def delete_bot(bot: str = BOT_ID) -> Any:
        """Request bot's deletion: out of service now (chat 404s), purged after 7 days unless restored."""
        try:
            purge_after = host.delete_bot(bot)
        except AlreadyDeleting:
            return JSONResponse({"detail": "the bot is already waiting to be deleted"}, status_code=409)
        except KeyError as exc:
            raise HTTPException(404, f"unknown bot {bot!r}") from exc
        return {"bot": bot, "purge_after": purge_after}

    @app.post("/v1/builder/bots/{bot}/restore", dependencies=deps, response_model=BotRestoredOut)
    def restore_bot(bot: str = BOT_ID) -> Any:
        """Cancel a pending deletion and serve the bot's deployed pair again."""
        try:
            key = host.restore_bot(bot)
        except NoDeletionPending as exc:
            raise HTTPException(404, f"no deletion is pending for {bot!r}") from exc
        return {"bot": bot, "live": key[0] if key else None}

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

    @app.get("/v1/builder/bots/{bot}/edits", dependencies=deps, response_model=EditsOut)
    def edits(bot: str = BOT_ID, limit: int = Query(50, ge=1, le=500)) -> Any:
        """작업이력: what each draft save changed, newest first."""
        known(bot)
        return {"bot": bot, "items": [{"revision": e.revision, "saved_at": e.saved_at, "changes": e.changes}
                                      for e in host.repository.edits(bot, limit)]}

    @app.get("/v1/builder/bots/{bot}/utterances", dependencies=deps, response_model=UtterancesOut)
    def utterances(bot: str = BOT_ID, since: dt.datetime | None = None, until: dt.datetime | None = None,
                   rule: RuleFilter = "misunderstood", intent: str | None = Query(None, max_length=64),
                   status: StatusFilter = "pending", limit: int = Query(100, ge=1, le=500)) -> Any:
        """학습대기: what users said in [since, until), grouped by sentence, most frequent first.
        Test-chat sessions are not included. Reads at most the newest UTTERANCE_SCAN messages."""
        known(bot)
        if (since and since.tzinfo is None) or (until and until.tzinfo is None):
            raise HTTPException(422, "since and until need a time zone (e.g. 2026-11-10T00:00:00+09:00)")
        found = host.store.utterances(f"{bot}:", since, until, UTTERANCE_SCAN + 1)
        items = group(found[:UTTERANCE_SCAN], host.repository.latest_marks(bot), rule, intent, status, limit)
        return {"bot": bot, "scanned": min(len(found), UTTERANCE_SCAN), "truncated": len(found) > UTTERANCE_SCAN,
                "items": [asdict(i) for i in items]}

    def mark(bot: str, body: SentencesIn, status: str) -> dict[str, Any]:
        known(bot)
        if any(not queue_key(t) for t in body.texts):
            raise HTTPException(422, "a sentence is empty after trimming spaces and punctuation")
        sentences = unique_sentences(body.texts)
        host.repository.mark_utterances(bot, sentences, status)
        return {"bot": bot, "marked": len(sentences)}

    @app.post("/v1/builder/bots/{bot}/utterances/ignore", dependencies=deps, response_model=MarkedOut)
    def ignore(body: SentencesIn, bot: str = BOT_ID) -> Any:
        """Hide sentences from the queue (they come back with "restore")."""
        return mark(bot, body, "ignored")

    @app.post("/v1/builder/bots/{bot}/utterances/restore", dependencies=deps, response_model=MarkedOut)
    def restore(body: SentencesIn, bot: str = BOT_ID) -> Any:
        return mark(bot, body, "pending")

    @app.post("/v1/builder/bots/{bot}/utterances/add", dependencies=deps, response_model=DraftSavedOut)
    def add_examples(body: AddExamplesIn, response: Response, bot: str = BOT_ID,
                     if_match: str | None = Header(default=None)) -> Any:
        """Add sentences to an intent of the draft as examples, like a save: needs the draft's ETag
        (428 without, 412 when it moved on). 404 unknown intent, 409 already another intent's example."""
        known(bot)
        expected = parse_if_match(if_match)
        if expected is None:
            raise HTTPException(428, "send the draft's ETag in If-Match")
        try:
            check = host.add_examples(bot, body.texts, body.intent, expected)
        except KeyError as exc:
            raise HTTPException(404, f"the draft has no intent {body.intent!r}") from exc
        except ExampleTaken as exc:
            return JSONResponse({"detail": "already examples of another intent", "texts": exc.texts}, status_code=409)
        except BadExample as exc:
            return JSONResponse({"detail": "these cannot be examples (empty, or {braces} of a template)",
                                 "texts": exc.texts}, status_code=422)
        except RevisionConflict as exc:
            return JSONResponse({"detail": "the draft was changed elsewhere; reload it", "revision": exc.current},
                                status_code=412)
        response.headers["ETag"] = etag(check.revision)
        return {"bot": bot, "version": check.version, "revision": check.revision, "valid": not check.errors,
                "errors": check.errors, "warnings": check.warnings}

    @app.get("/v1/builder/bots/{bot}/utterances/marks", dependencies=deps, response_model=MarksOut)
    def marks(bot: str = BOT_ID, limit: int = Query(100, ge=1, le=1000)) -> Any:
        """학습이력: sentences added, ignored or restored, newest first."""
        known(bot)
        return {"bot": bot, "items": [asdict(m) for m in host.repository.utterance_marks(bot, limit)]}

    @app.get("/v1/builder/bots/{bot}/skills/errors", dependencies=deps, response_model=SkillErrorsOut)
    def skill_errors(bot: str = BOT_ID, limit: int = Query(100, ge=1, le=100)) -> Any:
        """스킬 오류 내역: the newest failed skill calls (test runs included), at most 100."""
        known(bot)
        return {"bot": bot, "items": host.repository.skill_errors(bot, limit)}

    @app.post("/v1/builder/bots/{bot}/skills/{skill}/test", dependencies=deps, response_model=SkillTestOut)
    def test_skill(body: SkillTestIn, bot: str = BOT_ID, skill: str = SKILL_ID) -> Any:
        """Call a skill of the saved draft once (its test_url when set) and show what came back."""
        known(bot)
        try:
            outcome = host.test_skill(bot, skill, body.params, body.utterance)
        except NoDraft as exc:
            raise HTTPException(404, f"{bot!r} has no draft") from exc
        except BotError as exc:
            raise HTTPException(422, exc.problems) from exc
        except KeyError as exc:
            raise HTTPException(404, f"the saved draft has no skill {skill!r}") from exc
        return asdict(outcome)

    @app.get("/v1/builder/bots/{bot}/web", dependencies=deps, response_model=WebOut)
    def web_settings(bot: str = BOT_ID) -> Any:
        """The bot's web chat settings: allowed sites and the widget's look (defaults until saved)."""
        known(bot)
        return {"bot": bot, **(load(host.repository.web_settings(bot)) or WebSettings()).model_dump()}

    @app.put("/v1/builder/bots/{bot}/web", dependencies=deps, response_model=WebOut)
    def save_web_settings(body: WebSettings, bot: str = BOT_ID) -> Any:
        """Replace the web chat settings; they apply at once (no deploy)."""
        known(bot)
        host.repository.save_web_settings(bot, body.model_dump())
        origins.forget(bot)
        return {"bot": bot, **body.model_dump()}

    @app.get("/v1/builder/bots/{bot}/secrets", dependencies=deps, response_model=SecretsOut)
    def secrets(bot: str = BOT_ID) -> Any:
        """The bot's secret names (for {{ secret.NAME }} in skill headers). Values are write-only."""
        known(bot)
        return {"bot": bot, "items": [asdict(s) for s in host.repository.secrets(bot)]}

    @app.put("/v1/builder/bots/{bot}/secrets/{name}", dependencies=deps, status_code=204)
    def set_secret(body: SecretIn, bot: str = BOT_ID, name: str = SECRET) -> Response:
        known(bot)
        if any(c in body.value for c in "\r\n\0") or not body.value.isascii():
            raise HTTPException(422, "a secret value must be ASCII without line breaks (it is sent in a header)")
        host.repository.set_secret(bot, name, body.value)
        return Response(status_code=204)

    @app.delete("/v1/builder/bots/{bot}/secrets/{name}", dependencies=deps, status_code=204)
    def delete_secret(bot: str = BOT_ID, name: str = SECRET) -> Response:
        known(bot)
        if not host.repository.delete_secret(bot, name):
            raise HTTPException(404, f"no secret {name!r}")
        return Response(status_code=204)

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
                    "warnings": report.warnings, "thresholds": asdict(report.thresholds),
                    "confusable": [asdict(c) for c in report.confusable]}

        return jobs.submit("version", bot, run).to_dict()

    @app.post("/v1/builder/bots/{bot}/test-builds", dependencies=deps, status_code=202, response_model=JobOut)
    def test_build(bot: str = BOT_ID) -> Any:
        """Fit the draft for the test chat (no evaluation, not deployable); follow the job."""
        known(bot)

        def run() -> dict[str, Any]:
            try:
                build = host.build_test(bot)
                confusable = host.confusable(bot, build)  # of the document that was built
                return {"build": build, "confusable": [asdict(c) for c in confusable]}
            except BotError as exc:
                raise JobFailed("invalid_draft", {"detail": "the draft is invalid", "errors": exc.problems}) from exc
            except TrainingError as exc:  # fewer than two intents, nothing in scope
                raise JobFailed("training", {"detail": str(exc)}) from exc
            except NoDraft as exc:
                raise JobFailed("no_draft", {"detail": f"{bot!r} has no draft"}) from exc
            except KnowledgeError as exc:  # the knowledge-base draft goes into the test build too
                raise JobFailed("invalid_knowledge", {"detail": "the knowledge base is invalid",
                                                      "errors": exc.problems}) from exc

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
        return test_turn(bot, build, session, body.text, body.block_id)

    @app.post("/v1/builder/bots/{bot}/test/{build}/sessions/{session}/open", dependencies=deps,
              response_model=TestTurnOut)
    def test_open(bot: str = BOT_ID, build: str = VERSION, session: str = SESSION_ID) -> Any:
        """Open a test chat: the welcome block of a new session (410 like messages)."""
        return test_turn(bot, build, session, None, None)

    def test_turn(bot: str, build: str, session: str, text: str | None, block: str | None) -> Any:
        rt = host.test_runtime(bot, build)
        if rt is None:
            raise HTTPException(410, "this test build expired; save the draft again")
        key = f"{bot}{TEST_SESSION_SEP}{session}"
        result = run_turn(rt, key, text, block)
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
                "history": [{"version": d.version, "number": d.number, "note": d.note, "deployed_at": d.deployed_at,
                             "knowledge_version": d.kb_version} for d in history]}

    # ------------------------------------------------------------ knowledge base (지식베이스)

    @app.get("/v1/builder/bots/{bot}/knowledge/draft", dependencies=deps, responses=KB_RESPONSE)
    def get_kb_draft(response: Response, bot: str = BOT_ID) -> Any:
        known(bot)
        draft = host.repository.kb_draft(bot)
        if draft is None:
            raise HTTPException(404, f"{bot!r} has no knowledge base")
        response.headers["ETag"] = etag(draft.revision)
        return draft.document

    @app.put("/v1/builder/bots/{bot}/knowledge/draft", dependencies=deps, response_model=KbSavedOut,
             openapi_extra=KB_BODY)
    def put_kb_draft(document: dict[str, Any], response: Response, bot: str = BOT_ID,
                     if_match: str | None = Header(default=None)) -> Any:
        """Like the bot draft: saved even when invalid; If-Match needed over an existing one (428/412)."""
        known(bot)
        expected = parse_if_match(if_match)
        if expected is None:
            if host.repository.kb_draft(bot) is not None:
                raise HTTPException(428, "send the knowledge base's ETag in If-Match")
            expected = 0
        try:
            check = host.save_kb_draft(bot, document, expected)
        except RevisionConflict as exc:
            return JSONResponse({"detail": "the knowledge base was changed elsewhere; reload it",
                                 "revision": exc.current}, status_code=412)
        response.headers["ETag"] = etag(check.revision)
        return {"bot": bot, "version": check.version, "revision": check.revision, "valid": not check.errors,
                "errors": check.errors}

    @app.post("/v1/builder/bots/{bot}/knowledge/extract", dependencies=deps, response_model=ExtractOut,
              openapi_extra=EXTRACT_BODY)
    async def extract(request: Request, bot: str = BOT_ID) -> Any:
        """The text of an uploaded file (PDF, plain text or Markdown), to put in the draft as a document."""
        known(bot)
        kind = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        body = bytearray()
        async for part in request.stream():
            body += part
            if len(body) > MAX_UPLOAD:
                raise HTTPException(413, f"the file is over {MAX_UPLOAD // (1024 * 1024)} MB")
        if not body:
            raise HTTPException(422, "the file is empty")
        try:
            text = await asyncio.to_thread(extract_text, bytes(body), kind)
        except KnowledgeError as exc:
            raise HTTPException(422, exc.problems) from exc
        return {"text": text}

    @app.post("/v1/builder/bots/{bot}/knowledge/versions", dependencies=deps, status_code=202, response_model=JobOut)
    def build_kb(bot: str = BOT_ID) -> Any:
        """Index the knowledge-base draft as a version; follow it at /v1/builder/jobs/{id}."""
        known(bot)

        def run() -> dict[str, Any]:
            try:
                info = host.build_kb(bot)
            except KnowledgeError as exc:
                raise JobFailed("invalid_draft", {"detail": "the knowledge base is invalid",
                                                  "errors": exc.problems}) from exc
            except NoKnowledgeBase as exc:
                raise JobFailed("no_draft", {"detail": f"{bot!r} has no knowledge base"}) from exc
            return {"version": info.version, "number": info.number, "faq": info.faq, "chunks": info.chunks}

        return jobs.submit("knowledge", bot, run).to_dict()

    @app.get("/v1/builder/bots/{bot}/knowledge/versions", dependencies=deps, response_model=KbVersionsOut)
    def kb_versions(bot: str = BOT_ID) -> Any:
        known(bot)
        history = host.repository.history(bot)
        live = history[0].kb_version if history else None
        return {"bot": bot, "live": live,
                "versions": [{**asdict(v), "live": v.version == live} for v in host.repository.kb_versions(bot)]}

    @app.get("/v1/builder/bots/{bot}/knowledge/versions/{version}/entries", dependencies=deps,
             response_model=KbEntriesOut)
    def kb_entries(bot: str = BOT_ID, version: str = PathParam(pattern=r"^[0-9a-f]{12}$")) -> Any:
        """The FAQ entries and document chunks of a built version, as searched (조각 미리보기)."""
        known(bot)
        try:
            index = host.kb_index(bot, version)
        except VersionNotFound as exc:
            raise HTTPException(404, f"unknown knowledge-base version: {version}") from exc
        except KnowledgeError as exc:  # stored, but cannot be indexed here
            raise unusable_kb(exc) from exc
        return {"bot": bot, "version": version, "entries": [asdict(e) for e in [*index.faq, *index.chunks]]}

    @app.post("/v1/builder/bots/{bot}/knowledge/search", dependencies=deps, response_model=KbSearchOut)
    def kb_search(body: KbSearchIn, bot: str = BOT_ID) -> Any:
        """Search a built version as a conversation would (검색 테스트): candidates, scores, the answer."""
        known(bot)
        try:
            version, result, answer = host.search_kb(bot, body.query, body.version)
        except NoKnowledgeBase as exc:
            raise HTTPException(409, "build the knowledge base first") from exc
        except VersionNotFound as exc:
            raise HTTPException(404, f"unknown knowledge-base version: {body.version}") from exc
        except KnowledgeError as exc:
            raise unusable_kb(exc) from exc
        passed = {h.entry.id for h in result.answers}
        return {"bot": bot, "version": version, "judge": host.judge.name, "threshold": host.judge.threshold,
                "exact": result.exact,
                "answer": None if answer is None else {
                    "text": answer.text, "items": list(answer.items),
                    "replies": [{"label": label, "message": message} for label, message in answer.replies]},
                "generated": result.generated,
                "candidates": [{"id": h.entry.id, "kind": h.entry.kind, "title": h.entry.title, "cosine": h.cosine,
                                "score": h.score, "passed": h.entry.id in passed} for h in result.candidates]}

    @app.get("/v1/builder/llm", dependencies=deps, response_model=LlmOut)
    def llm_status() -> Any:
        """Whether this server has an LLM for generated knowledge-base answers, and whether it answers now
        (one very short generation)."""
        generator = getattr(host, "generator", None)
        if generator is None:
            return {"configured": False}
        start = time.monotonic()
        try:
            generator.client.chat([{"role": "user", "content": "'네'라고만 답하세요."}], max_tokens=8)
        except GenerationError as exc:
            return {"configured": True, "model": generator.model, "ok": False, "error": exc.detail}
        ms = round((time.monotonic() - start) * 1000)
        return {"configured": True, "model": generator.model, "ok": True, "ms": ms}

    @app.get("/v1/builder/bots/{bot}/analytics", dependencies=deps, response_model=AnalyticsOut)
    def analytics_report(bot: str = BOT_ID, from_date: str | None = Query(None, alias="from"),
                         to: str | None = Query(None)) -> Any:
        """Period totals, a daily table, block-call ranking and new-user retention (분석, E12).
        Default period: the 7 days ending today (KST). Only `{bot}:` sessions count."""
        known(bot)
        today = dt.datetime.now(KST).date()
        try:
            to_d = today if to is None else dt.date.fromisoformat(to)
            from_d = (to_d - dt.timedelta(days=6)) if from_date is None else dt.date.fromisoformat(from_date)
        except (ValueError, OverflowError) as exc:
            raise HTTPException(422, f"malformed date: {exc}") from exc
        if not (EARLIEST <= from_d and to_d <= LATEST):  # far dates would overflow the read window
            raise HTTPException(422, f"dates must be between {EARLIEST} and {LATEST}")
        if from_d > to_d:
            raise HTTPException(422, "'from' must not be after 'to'")
        if (to_d - from_d).days + 1 > MAX_DAYS:
            raise HTTPException(422, f"the period must be at most {MAX_DAYS} days")

        runtime = host.runtime(bot)
        if runtime is not None:
            labels = {sid: (s.label or sid) for sid, s in runtime.bot.scenarios.items()}
        else:
            document = host.repository.get_draft(bot)
            scenarios = document.get("scenarios") if isinstance(document, dict) else None
            labels = ({sid: (s.get("label") or sid) if isinstance(s, dict) else sid
                      for sid, s in scenarios.items()} if isinstance(scenarios, dict) else {})

        prefix = f"{bot}:"
        rows = host.store.activity(prefix, ACTIVITY_TYPES, start_of(from_d) - SESSION_GAP,
                                   start_of(to_d + dt.timedelta(days=8)))
        first_seen = host.store.first_seen(prefix)
        report = compute_report(rows, first_seen, from_d, to_d, labels, today)
        return {
            "bot": bot, "from": report.from_, "to": report.to, "timezone": "Asia/Seoul",
            "totals": asdict(report.totals),
            "days": [asdict(d) for d in report.days],
            "blocks": [asdict(b) for b in report.blocks],
            "retention": [asdict(r) for r in report.retention],
        }

    @app.post("/v1/builder/bots/{bot}/deploy", dependencies=deps, response_model=LiveOut)
    def deploy(body: DeployIn, bot: str = BOT_ID) -> Any:
        known(bot)
        knowledge = body.knowledge_version
        if "knowledge_version" not in body.model_fields_set:  # a client that knows nothing of knowledge bases
            live = host.live(bot)
            knowledge = live[1] if live is not None else None
        with deploy_errors(body.version):
            host.deploy(bot, body.version, body.note.strip(), knowledge)
        return {"bot": bot, "live": body.version, "knowledge_version": knowledge}

    @app.post("/v1/builder/bots/{bot}/rollback", dependencies=deps, response_model=LiveOut)
    def rollback(bot: str = BOT_ID) -> Any:
        known(bot)
        with deploy_errors("the previous version"):
            previous = host.rollback(bot)
        return {"bot": bot, "live": previous.version, "knowledge_version": previous.kb_version}


@contextmanager
def deploy_errors(what: str) -> Iterator[None]:
    try:
        yield
    except NoEarlierDeployment as exc:
        raise HTTPException(409, "there is no earlier deployment to go back to") from exc
    except VersionNotFound as exc:  # the bot version or the knowledge-base version ("bot@version")
        raise HTTPException(404, f"unknown version: {exc}") from exc
    except NotTrained as exc:
        raise HTTPException(409, f"{what} has no trained model; train it first") from exc
    except BotError as exc:  # stored, but no longer valid here (e.g. an action was removed)
        raise HTTPException(409, {"detail": f"{what} cannot run here", "errors": exc.problems}) from exc
    except KnowledgeError as exc:  # a stored knowledge base that cannot be indexed here
        raise unusable_kb(exc) from exc


def llm_generator(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Generator | None:
    """The generator the --llm-* options describe, or None without --llm-url."""
    import math
    import os
    import urllib.parse

    if args.llm_url is None:
        return None
    try:
        url = urllib.parse.urlsplit(args.llm_url)
        if url.port is not None and not 0 < url.port:  # url.port: ValueError when not a number
            raise ValueError("port 0")
    except ValueError:
        url = urllib.parse.urlsplit("")
    if url.scheme not in ("http", "https") or not url.hostname:
        parser.error("--llm-url must be an http(s) URL")
    if url.username or url.password:
        parser.error("--llm-url must not hold a user or password (use --llm-api-key-env)")
    if not args.llm_model:
        parser.error("--llm-model is required with --llm-url")
    if not (math.isfinite(args.llm_timeout) and 0 < args.llm_timeout <= 60):
        parser.error("--llm-timeout must be more than 0 and at most 60 seconds")
    key = os.environ.get(args.llm_api_key_env) if args.llm_api_key_env else None
    if args.llm_api_key_env and not key:
        parser.error(f"environment variable {args.llm_api_key_env} is not set")
    log.info("generated knowledge-base answers: %s at %s", args.llm_model, args.llm_url)
    return Generator(LlmClient(LlmConfig(args.llm_url, args.llm_model, args.llm_timeout, key)), depth=GENERATION_DEPTH)


def unusable_kb(exc: KnowledgeError) -> HTTPException:
    return HTTPException(409, {"detail": "the knowledge base cannot be used here", "errors": exc.problems})


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
    parser.add_argument("--skill-allow-private", action="store_true",
                        help="let HTTP skills call internal addresses (127.0.0.1, 10.x, 169.254.x ...): development "
                             "and tests only — on a server this lets anyone with the admin token reach its network")
    parser.add_argument("--kb-judge", choices=("cross-encoder", "cosine"), default=None,
                        help="what decides whether a knowledge-base entry answers: the cross-encoder (default; "
                             "downloads it once) or the cosine (default with --embedder hashing)")
    parser.add_argument("--kb-threshold", type=float, default=None,
                        help="the judge's bar (default: 0.11 cross-encoder, 0.5 cosine; docs/benchmarks/kb.md)")
    parser.add_argument("--llm-url", default=None,
                        help="an OpenAI-compatible server (POST <url>/v1/chat/completions) for knowledge bases whose "
                             "settings ask for generated answers; without it they answer with the passage itself")
    parser.add_argument("--llm-model", default=None, help="the model name to send (required with --llm-url)")
    parser.add_argument("--llm-timeout", type=float, default=LLM_TIMEOUT,
                        help="seconds for one generated answer in all; slower ones show the passage instead "
                             "(the server must answer without a thinking block)")
    parser.add_argument("--llm-api-key-env", default=None,
                        help="the NAME of an environment variable holding the server's API key (never the key itself)")
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
    judge_kind = args.kb_judge or ("cosine" if args.embedder == "hashing" else "cross-encoder")
    cross_encoder = None
    if judge_kind == "cross-encoder":
        from chatcore.knowledge import CROSS_ENCODER_THRESHOLD, CrossEncoderJudge

        cross_encoder = CrossEncoderJudge(args.kb_threshold if args.kb_threshold is not None
                                          else CROSS_ENCODER_THRESHOLD)
    embedder = BatchedEmbedder(inner, keepalive_ms=args.keep_warm_ms, cross_encoder=cross_encoder)
    from chatcore.knowledge import COSINE_THRESHOLD, CosineJudge

    judge = embedder.judge() or CosineJudge(args.kb_threshold if args.kb_threshold is not None else COSINE_THRESHOLD)
    host = BotHost(default_repository(), default_store(), embedder, args.artifacts, bulk_embedder=embedder.bulk(),
                   min_f1=args.min_f1, skill_allow_private=args.skill_allow_private, judge=judge,
                   generator=llm_generator(parser, args))
    if args.skill_allow_private:
        log.warning("--skill-allow-private: HTTP skills may call internal addresses; never use this on a shared server")
    host.start()
    host.start_purge_daemon()  # permanently deletes bots past their 7-day deletion grace period, hourly
    for directory in args.bot:
        host.seed(directory)
    if not host.bots():
        if not host.repository.deletions():
            parser.error("no deployed bots: pass --bot <directory>")
        log.warning("every bot is waiting to be deleted; restore one with POST /v1/builder/bots/{bot}/restore")
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
