"""Multi-turn conversation tests (docs/engine-core-plan.md Phase 2 done criteria).

Each YAML file is a dialogue: per user turn, the commands an interpreter would
produce, and the expected response IDs (and optionally texts, quick replies
and the scenario stack afterwards). NLU is not involved yet.
"""

import random
from pathlib import Path

import pytest
import yaml

from chatcore.dialogue.commands import from_dict
from chatcore.dialogue.state import fold
from chatcore.dialogue.turn import ScriptedInterpreter, TurnProcessor
from chatcore.store.memory import InMemoryStore

FILES = sorted((Path(__file__).parent / "conversations").glob("*.yaml"))


def test_conversation_files_exist():
    assert len(FILES) >= 4


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_conversation(path, sample_bot, registry):
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    script = {t["user"]: [from_dict(c) for c in t["commands"]] for t in spec["turns"]}
    store = InMemoryStore()
    proc = TurnProcessor(sample_bot, store, ScriptedInterpreter(script), registry, rng=random.Random(0))
    for i, turn in enumerate(spec["turns"]):
        result = proc.handle("s1", turn["user"])
        where = f"{spec['name']} / turn {i + 1} ({turn['user']!r})"
        assert [m.response for m in result.messages] == turn["expect"], where
        if "text" in turn:
            assert [m.text for m in result.messages] == turn["text"], where
        if "quick_replies" in turn:
            assert [list(m.quick_replies) for m in result.messages] == turn["quick_replies"], where
        if "stack" in turn:
            assert [f.scenario for f in fold(store.events("s1")).stack] == turn["stack"], where
