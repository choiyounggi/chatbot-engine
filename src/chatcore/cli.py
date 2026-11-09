"""Chat with a bot in the terminal: `make chat` (uv run python -m chatcore.cli bots/sample).

Type a message and press Enter; `/debug` toggles showing the NLU result, `/quit` exits.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Callable, Sequence
from typing import TextIO

from chatcore.runtime import BotRuntime


def chat(rt: BotRuntime, read: Callable[[], str], out: TextIO) -> int:
    """The REPL loop, separated from model loading so it can be tested."""
    session = f"cli-{uuid.uuid4().hex[:8]}"
    debug = False
    out.write(f"{rt.bot.label or rt.bot.id} — /debug 로 해석 결과 보기, /quit 로 끝내기\n")
    while True:
        try:
            text = read()
        except (EOFError, KeyboardInterrupt):
            out.write("\n")
            return 0
        command = text.strip()
        if command == "/quit":
            return 0
        if command == "/debug":
            debug = not debug
            out.write(f"(debug {'on' if debug else 'off'})\n")
            continue
        result = rt.processor.handle(session, text)
        for message in result.messages:
            out.write(f"봇> {message.text}\n")
            if message.quick_replies:
                out.write(f"    [{' | '.join(message.quick_replies)}]\n")
        if debug:
            nlu = next((e.nlu for e in reversed(result.events) if e.type == "commands_received"), {})
            out.write(f"    ({nlu.get('rule')} {nlu.get('intents', [[None]])[0]} entities={nlu.get('entities')})\n")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - loads the real model
    import argparse
    from pathlib import Path

    from chatcore.nlu.model import SentenceEmbedder
    from chatcore.runtime import build_runtime
    from chatcore.store.memory import InMemoryStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bot", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    rt = build_runtime(args.bot, args.artifacts, SentenceEmbedder(), InMemoryStore())
    return chat(rt, lambda: input("나> "), sys.stdout)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
