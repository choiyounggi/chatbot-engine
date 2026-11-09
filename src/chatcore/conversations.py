"""Run a bot's golden conversations through the real NLU model (`make eval`).

    uv run python -m chatcore.conversations bots/sample [--artifacts artifacts]

Each conversation starts a new session; every turn's response IDs (and,
when given, texts) must match. Actions use synthetic data so the run is
repeatable. Exits 1 if any conversation fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from chatcore.actions.builtin import KST, OpenMeteo, builtin_registry
from chatcore.bot.loader import load_bot
from chatcore.bot.schema import BotDefinition
from chatcore.dialogue.turn import Interpreter, TurnProcessor
from chatcore.dialogue.values import ActionRegistry
from chatcore.store.memory import InMemoryStore

FIXED_NOW = dt.datetime(2030, 1, 1, 15, 0, tzinfo=KST)  # "오후 3시"
FIXED_WEATHER = {"current": {"time": "2030-01-01T15:00", "temperature_2m": 20.0, "weather_code": 0}}  # 맑음, 20°C


def synthetic_actions() -> ActionRegistry:
    """The real built-in actions on a fake network and a fixed clock."""
    return builtin_registry(OpenMeteo(http=lambda url, timeout: FIXED_WEATHER), clock=lambda: FIXED_NOW)


@dataclass(frozen=True)
class TurnFailure:
    conversation: str
    turn: int
    user: str
    expected: list[str]
    got: list[str]
    rule: str


def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    convs = data.get("conversations")
    if not isinstance(convs, list) or not convs:
        raise ValueError(f"{path} has no conversations")
    for c in convs:
        if not c.get("name") or not c.get("turns") or any("user" not in t or "expect" not in t for t in c["turns"]):
            raise ValueError(f"malformed conversation in {path}: {c!r}")
    return convs


def run(bot: BotDefinition, interpreter: Interpreter, conversations: Sequence[dict[str, Any]],
        actions: ActionRegistry | None = None) -> tuple[int, list[TurnFailure]]:
    """(conversations passed, failures). A conversation stops at its first failing turn."""
    store = InMemoryStore()
    proc = TurnProcessor(bot, store, interpreter, actions or synthetic_actions(), rng=random.Random(0))
    passed = 0
    failures: list[TurnFailure] = []
    for i, conv in enumerate(conversations):
        session = f"eval-{i}"
        ok = True
        for n, turn in enumerate(conv["turns"], 1):
            result = proc.handle(session, turn["user"])
            got = [m.response for m in result.messages]
            texts = [m.text for m in result.messages]
            rule = next((e.nlu.get("rule", "") for e in reversed(result.events) if e.type == "commands_received"), "")
            wanted_texts = turn.get("text")
            if got != turn["expect"] or (wanted_texts is not None and texts != wanted_texts):
                shown = got if got != turn["expect"] else texts
                failures.append(TurnFailure(conv["name"], n, turn["user"],
                                            turn["expect"] if got != turn["expect"] else wanted_texts, shown, rule))
                ok = False
                break
        passed += ok
    return passed, failures


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs the trained real model
    from chatcore.nlu.interpreter import NluInterpreter
    from chatcore.nlu.model import NluModel, SentenceEmbedder

    parser = argparse.ArgumentParser(description="Run golden conversations through the real NLU model.")
    parser.add_argument("bot", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    bot = load_bot(args.bot)
    convs = load_conversations(args.bot / "conversations.yaml")
    model = NluModel.load(bot, args.artifacts / bot.id / bot.version, SentenceEmbedder(), today=FIXED_NOW.date)
    passed, failures = run(bot, NluInterpreter(model.parse), convs)
    for f in failures:
        print(f"FAIL {f.conversation} / turn {f.turn} {f.user!r}: expected {f.expected}, got {f.got} (rule {f.rule})")
    print(f"{passed}/{len(convs)} conversations passed")
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
