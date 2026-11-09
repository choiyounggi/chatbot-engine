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
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from chatcore.dialogue.turn import MAX_INPUT_CHARS
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


def create_app(runtimes: dict[str, BotRuntime]) -> FastAPI:
    app = FastAPI(title="chatcore", version="0.0.1")

    def runtime(bot: str) -> BotRuntime:
        rt = runtimes.get(bot)
        if rt is None:
            raise HTTPException(404, f"unknown bot {bot!r}")
        return rt

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "bots": {k: v.bot.version for k, v in runtimes.items()}}

    @app.post("/v1/bots/{bot}/sessions/{session}/messages", response_model=TurnOut)
    def message(body: MessageIn, bot: str = BOT_ID, session: str = SESSION_ID, debug: bool = False) -> Any:
        rt = runtime(bot)
        started = time.perf_counter()
        try:
            result = rt.processor.handle(f"{bot}:{session}", body.text)
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
            "bot": bot, "session": session, "ms": round((time.perf_counter() - started) * 1000, 1),
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

    return app


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - starts a server
    import uvicorn

    from chatcore.nlu.model import SentenceEmbedder
    from chatcore.runtime import build_runtimes, default_store
    from chatcore.serving import BatchedEmbedder

    parser = argparse.ArgumentParser(description="Serve chatcore bots over HTTP.")
    parser.add_argument("--bot", type=Path, action="append", required=True, help="bot directory (repeatable)")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--keep-warm-ms", type=float, default=None,
                        help="when idle, run the model this often so the first request after a pause stays fast "
                             "(keeps the GPU busy; see docs/benchmarks/api_latency.md)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    embedder = BatchedEmbedder(SentenceEmbedder(), keepalive_ms=args.keep_warm_ms)
    runtimes = build_runtimes(args.bot, args.artifacts, embedder, default_store())
    for rt in runtimes.values():  # load the model weights before the first user waits for them
        rt.model.parse("안녕하세요")
    uvicorn.run(create_app(runtimes), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
