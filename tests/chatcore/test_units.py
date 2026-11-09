"""Commands, slot values, templates, state folding."""

import random

import pytest

from chatcore.bot.schema import BotDefinition, Response
from chatcore.dialogue.commands import (
    CancelScenario,
    Clarify,
    CommandFormatError,
    Fallback,
    RerunScenario,
    Restart,
    SetSlot,
    StartScenario,
    from_dict,
    to_dict,
)
from chatcore.dialogue.events import (
    ScenarioCompleted,
    ScenarioStarted,
    SessionRestarted,
    SlotSet,
    StepEntered,
    UserUttered,
    from_json,
    to_json,
)
from chatcore.dialogue.state import RECENT_COMPLETED, StateError, fold
from chatcore.dialogue.values import (
    MissingVariable,
    SlotValueError,
    as_text,
    lookup,
    normalize_slot,
    render,
    render_response,
)

# ---------------------------------------------------------------- commands

ALL = [StartScenario("s"), StartScenario("s", {"c": "서울", "n": 3}), SetSlot("c", "서울"), CancelScenario(), Restart(),
       Clarify(("a", "b")), RerunScenario("s", "c", "부산"), Fallback()]


@pytest.mark.parametrize("cmd", ALL, ids=lambda c: type(c).__name__)
def test_command_dict_round_trip(cmd):
    assert from_dict(to_dict(cmd)) == cmd


@pytest.mark.parametrize(
    "bad",
    [{}, [], {"set": {"a": 1, "b": 2}}, {"set": "x"}, {"rerun": "s"}, {"clarify": ["only"]}, {"clarify": "a"},
     {"start": "s", "slots": ["x"]}, {"set": {"c": True}}, {"set": {"c": None}}, {"bogus": 1}, {"set": {"c": float("nan")}}],
)
def test_bad_command_dicts(bad):
    with pytest.raises(CommandFormatError):
        from_dict(bad)


# ---------------------------------------------------------------- slot values

@pytest.fixture(scope="module")
def bot() -> BotDefinition:
    req = {k: {"text": k} for k in ("fallback", "cancelled", "restarted", "error", "clarify", "input_invalid")}
    return BotDefinition.model_validate({
        "id": "t", "version": "v",
        "entities": {"city": {"kind": "dictionary", "values": {"부산": ["부산광역시"]}},
                     "code": {"kind": "regex", "pattern": r"\d{3}"}},
        "scenarios": {"s": {"steps": [{"respond": "fallback"}]}}, "responses": req,
    })


@pytest.mark.parametrize(
    "entity, raw, expected",
    [("city", "부산광역시", "부산"), ("city", " 부산 ", "부산"), ("code", "123", "123"), ("sys.number", "1,200", 1200),
     ("sys.number", "3.5", 3.5), ("sys.number", 7, 7), ("sys.date", "2026-11-09", "2026-11-09"), ("sys.time", "09:30", "09:30")],
)
def test_normalize_slot(bot, entity, raw, expected):
    assert normalize_slot(bot, entity, raw) == expected


@pytest.mark.parametrize(
    "entity, raw",
    [("city", "서울"), ("city", ""), ("code", "12"), ("code", "1234"), ("sys.number", "많이"), ("sys.date", "내일"),
     ("sys.time", "25:00"), ("nope", "x"), ("sys.number", "nan"), ("sys.number", "1e999"), ("sys.number", float("inf"))],
)
def test_normalize_slot_rejects(bot, entity, raw):
    with pytest.raises(SlotValueError):
        normalize_slot(bot, entity, raw)


# ---------------------------------------------------------------- templates

def test_render_and_lookup():
    ctx = {"slots": {"city": "부산", "n": 2.0, "ok": True}, "results": {"w": {"t": 21}}}
    assert render("{{ slots.city }} {{results.w.t}}°C {{ slots.n }} {{ slots.ok }}", ctx) == "부산 21°C 2 true"
    assert render("no placeholders", {}) == "no placeholders"
    assert lookup(ctx, "results.w") == {"t": 21}
    assert as_text(False) == "false" and as_text(2.5) == "2.5"


@pytest.mark.parametrize("template", ["{{ slots.none }}", "{{ results.w.t.deeper }}", "{{ slots.empty }}"])
def test_render_missing_variable(template):
    with pytest.raises(MissingVariable):
        render(template, {"slots": {"empty": None}, "results": {"w": {"t": 1}}})


def test_render_response_picks_variant_and_renders_quick_replies():
    resp = Response(variants=["a {{ slots.x }}", "b {{ slots.x }}"], quick_replies=["{{ slots.x }}!"])
    msg = render_response("r", resp, {"slots": {"x": "1"}}, random.Random(1))
    assert msg.text in {"a 1", "b 1"} and msg.quick_replies == ("1!",) and msg.response == "r"


# ---------------------------------------------------------------- state

def test_fold_rebuilds_stack_slots_and_completed():
    events = [UserUttered(text="hi"), ScenarioStarted(scenario="s"), SlotSet(scenario="s", name="c", value="부산"),
              StepEntered(scenario="s", index=2), ScenarioCompleted(scenario="s")]
    state = fold(events)
    assert state.turns == 1 and state.stack == []
    assert state.completed[-1].scenario == "s" and state.completed[-1].slots == {"c": "부산"}


def test_completed_history_is_capped_and_restart_clears():
    events = []
    for i in range(RECENT_COMPLETED + 3):
        events += [ScenarioStarted(scenario=f"s{i}"), ScenarioCompleted(scenario=f"s{i}")]
    state = fold(events)
    assert len(state.completed) == RECENT_COMPLETED and state.completed[-1].scenario == f"s{RECENT_COMPLETED + 2}"
    assert fold(events + [SessionRestarted()]).completed == []


@pytest.mark.parametrize(
    "events",
    [[StepEntered(scenario="s", index=1)], [ScenarioStarted(scenario="s"), ScenarioCompleted(scenario="t")],
     [SlotSet(scenario="s", name="c", value=1)]],
)
def test_inconsistent_logs_raise(events):
    with pytest.raises(StateError):
        fold(events)


def test_event_json_round_trip():
    for e in [ScenarioStarted(scenario="s"), SlotSet(scenario="s", name="n", value=3)]:
        assert from_json(to_json(e)) == e
    with pytest.raises(ValueError):
        from_json({"type": "nope"})
