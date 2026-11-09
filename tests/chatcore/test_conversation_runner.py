"""The `make eval` runner, driven by a scripted interpreter (no model needed)."""

from pathlib import Path

import pytest

from chatcore.conversations import load_conversations, run, synthetic_actions
from chatcore.dialogue.commands import StartScenario
from chatcore.dialogue.turn import ScriptedInterpreter

SCRIPT = {"부산 날씨": [StartScenario("check_weather", {"location": "부산"})], "몇 시야": [StartScenario("current_time")],
          "아산 날씨": [StartScenario("check_weather", {"location": "아산"})]}


def test_passing_and_failing_conversations(sample_bot):
    convs = [
        {"name": "ok", "turns": [{"user": "부산 날씨", "expect": ["weather_report"], "text": ["부산은 지금 맑음이에요."]},
                                 {"user": "몇 시야", "expect": ["time_report"], "text": ["지금은 오후 3시예요."]}]},
        {"name": "wrong id", "turns": [{"user": "부산 날씨", "expect": ["ask_location"]}, {"user": "몇 시야", "expect": ["x"]}]},
        {"name": "wrong text", "turns": [{"user": "부산 날씨", "expect": ["weather_report"], "text": ["틀림"]}]},
        {"name": "unknown city", "turns": [{"user": "아산 날씨", "expect": ["weather_unavailable"]}]},
    ]
    passed, failures = run(sample_bot, ScriptedInterpreter(SCRIPT), convs)
    assert passed == 2
    assert [(f.conversation, f.turn) for f in failures] == [("wrong id", 1), ("wrong text", 1)]  # stops at first failure
    assert failures[0].got == ["weather_report"] and failures[0].rule == "scripted"
    assert failures[1].got == ["부산은 지금 맑음이에요."]


def test_synthetic_actions_are_fixed():
    reg = synthetic_actions()
    assert reg.run("now", {})["time"] == "오후 3시"
    assert reg.run("get_weather", {"location": "서울"})["condition"] == "맑음"


def test_sample_golden_file_loads():
    convs = load_conversations(Path(__file__).resolve().parents[2] / "bots/sample/conversations.yaml")
    assert len(convs) >= 30


@pytest.mark.parametrize("text", ["conversations: []", "nothing: 1", "conversations: [{name: x, turns: [{user: a}]}]",
                                  "conversations: [{turns: [{user: a, expect: []}]}]"])
def test_malformed_files(tmp_path, text):
    f = tmp_path / "c.yaml"
    f.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_conversations(f)
