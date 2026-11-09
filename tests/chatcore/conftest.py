from pathlib import Path

import pytest

from chatcore.bot.loader import load_bot
from chatcore.bot.schema import BotDefinition
from chatcore.dialogue.values import ActionRegistry, StubAction

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "bots/sample"
SAMPLE_ACTIONS = ("get_weather", "get_temperature", "now")


@pytest.fixture(scope="session")
def sample_bot() -> BotDefinition:
    return load_bot(SAMPLE, actions=SAMPLE_ACTIONS)


@pytest.fixture
def stubs() -> dict[str, StubAction]:
    return {
        "get_weather": StubAction({"condition": "맑음"}),
        "get_temperature": StubAction({"celsius": 21}),
        "now": StubAction({"time": "오후 3시 5분"}),
    }


@pytest.fixture
def registry(stubs) -> ActionRegistry:
    reg = ActionRegistry()
    for name, stub in stubs.items():
        reg.register(name, stub)
    return reg
