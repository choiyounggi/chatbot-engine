"""Scenario engine features beyond the sample conversations: branch, call,
action failures, limits, rejected commands, and the replay invariant."""

import random

import pytest

from chatcore.bot.schema import BotDefinition
from chatcore.dialogue.commands import (
    CancelScenario,
    Clarify,
    Fallback,
    RerunScenario,
    SetSlot,
    StartScenario,
)
from chatcore.dialogue.engine import ScenarioEngine
from chatcore.dialogue.events import ActionFailed, CommandRejected, ScenarioCancelled, SlotRejected
from chatcore.dialogue.state import DialogueState, fold
from chatcore.dialogue.values import ActionError, ActionRegistry, StubAction

REQUIRED = {k: {"text": k} for k in ("fallback", "cancelled", "restarted", "error", "clarify", "input_invalid")}


def make_bot(scenarios: dict, responses: dict | None = None, **settings) -> BotDefinition:
    return BotDefinition.model_validate({
        "id": "t", "version": "v", "settings": settings,
        "intents": {"a": {"label": "에이", "examples": ["a"]}, "b": {"examples": ["b"]}},
        "entities": {"code": {"kind": "regex", "pattern": r"[A-Z]{3}"}},
        "scenarios": scenarios,
        "responses": {**REQUIRED, **(responses or {})},
    })


class Harness:
    """Runs turns against an engine and checks the replay invariant every time."""

    def __init__(self, bot: BotDefinition, actions: ActionRegistry | None = None):
        self.engine = ScenarioEngine(bot, actions or ActionRegistry(), random.Random(0))
        self.state = DialogueState()
        self.events: list = []

    def turn(self, *commands):
        out = self.engine.run(self.state, list(commands))
        self.events += out.events
        replayed = fold(self.events)
        assert replayed.stack == self.state.stack  # every field, incl. results and called
        assert replayed.completed == self.state.completed and replayed.turns == self.state.turns
        return [m.response for m in out.messages], out


def registry(**actions) -> ActionRegistry:
    reg = ActionRegistry()
    for name, fn in actions.items():
        reg.register(name, fn)
    return reg


def test_branch_calls_case_default_or_skips():
    bot = make_bot(
        {
            "main": {"steps": [{"action": "check"}, {"branch": {"on": "results.check.kind", "cases": {"x": "on_x"}, "default": "other"}},
                               {"respond": "after"}]},
            "on_x": {"steps": [{"respond": "rx"}]},
            "other": {"steps": [{"respond": "ro"}]},
            "skip": {"steps": [{"branch": {"on": "slots.missing", "cases": {"1": "on_x"}}}, {"respond": "after"}]},
        },
        {"rx": {"text": "x"}, "ro": {"text": "o"}, "after": {"text": "after"}},
    )
    result = {"kind": "x"}
    h = Harness(bot, registry(check=lambda args: result))
    assert h.turn(StartScenario("main"))[0] == ["rx", "after"]
    result = {"kind": "y"}
    assert h.turn(StartScenario("main"))[0] == ["ro", "after"]
    assert h.turn(StartScenario("skip"))[0] == ["after"]  # missing value, no default: continue
    assert h.state.stack == []


def test_call_returns_to_parent_and_child_sees_parent_slots():
    bot = make_bot(
        {
            "parent": {"slots": {"c": {"entity": "code", "ask": "ask"}},
                       "steps": [{"collect": "c"}, {"call": "child"}, {"respond": "bye"}]},
            "child": {"steps": [{"respond": "show"}]},
        },
        {"ask": {"text": "code?"}, "show": {"text": "code={{ slots.c }}"}, "bye": {"text": "bye"}},
    )
    h = Harness(bot)
    assert h.turn(StartScenario("parent"))[0] == ["ask"]
    responses, out = h.turn(SetSlot("c", "ABC"))
    assert responses == ["show", "bye"] and out.messages[0].text == "code=ABC"


def test_action_failure_cancels_scenario_and_uses_on_error():
    bot = make_bot(
        {"s": {"on_error": "sorry", "steps": [{"action": "boom"}, {"respond": "never"}]},
         "plain": {"steps": [{"action": "boom"}]}},
        {"sorry": {"text": "sorry"}, "never": {"text": "never"}},
    )
    h = Harness(bot, registry(boom=StubAction(error=TimeoutError("slow"))))
    responses, out = h.turn(StartScenario("s"))
    assert responses == ["sorry"] and h.state.stack == []
    failed = [e for e in out.events if isinstance(e, ActionFailed)]
    assert failed and "TimeoutError" in failed[0].error
    assert h.turn(StartScenario("plain"))[0] == ["error"]


def test_action_args_render_templates_and_missing_variable_fails():
    stub = StubAction({"ok": True})
    bot = make_bot({"s": {"steps": [{"action": "fetch", "args": {"q": "{{ slots.nope }}"}}]},
                    "t": {"steps": [{"action": "fetch", "args": {"q": "fixed"}}]}})
    h = Harness(bot, registry(fetch=stub))
    assert h.turn(StartScenario("s"))[0] == ["error"]
    assert stub.calls == []
    h.turn(StartScenario("t"))
    assert stub.calls == [{"q": "fixed"}]


def test_response_with_missing_variable_falls_back():
    bot = make_bot({"s": {"steps": [{"respond": "needs"}]}}, {"needs": {"text": "{{ slots.x }}"}})
    assert Harness(bot).turn(StartScenario("s"))[0] == ["fallback"]


def test_step_limit_cancels_everything():
    chain = {f"s{i}": {"steps": [{"call": f"s{i + 1}"}]} for i in range(3)} | {"s3": {"steps": [{"respond": "end"}]}}
    bot = make_bot(chain, {"end": {"text": "end"}}, max_steps_per_turn=3, max_stack_depth=10)
    h = Harness(bot)
    responses, out = h.turn(StartScenario("s0"))
    assert responses == ["fallback"] and h.state.stack == []
    assert {e.reason for e in out.events if isinstance(e, ScenarioCancelled)} == {"step_limit"}


def test_stack_depth_limit():
    chain = {f"s{i}": {"steps": [{"call": f"s{i + 1}"}]} for i in range(4)} | {"s4": {"steps": [{"respond": "end"}]}}
    bot = make_bot(chain, {"end": {"text": "end"}}, max_stack_depth=2)
    responses, _ = Harness(bot).turn(StartScenario("s0"))
    assert "error" in responses


@pytest.mark.parametrize(
    "command, reason",
    [
        (StartScenario("nope"), "unknown scenario"),
        (SetSlot("nope", "x"), "no running scenario"),
        (CancelScenario(), "nothing to cancel"),
        (Clarify(("a", "zzz")), "at least two known intents"),
        (RerunScenario("s", "c", "ABC"), "has not completed"),
    ],
)
def test_invalid_commands_are_rejected_with_fallback(command, reason):
    bot = make_bot({"s": {"slots": {"c": {"entity": "code", "ask": "ask"}}, "steps": [{"collect": "c"}]}},
                   {"ask": {"text": "?"}})
    responses, out = Harness(bot).turn(command)
    rejected = [e for e in out.events if isinstance(e, CommandRejected)]
    assert responses == ["fallback"] and rejected and reason in rejected[0].reason


def test_unknown_slot_name_in_start_is_rejected_but_scenario_starts():
    bot = make_bot({"s": {"slots": {"c": {"entity": "code", "ask": "ask"}}, "steps": [{"collect": "c"}]}},
                   {"ask": {"text": "?"}})
    responses, out = Harness(bot).turn(StartScenario("s", {"zz": "1"}))
    assert responses == ["ask"] and any(isinstance(e, CommandRejected) for e in out.events)


def test_invalid_value_is_rejected_and_reasked_until_limit():
    bot = make_bot({"s": {"slots": {"c": {"entity": "code", "ask": "ask"}}, "steps": [{"collect": "c"}, {"respond": "ok"}]}},
                   {"ask": {"text": "?"}, "ok": {"text": "ok"}}, max_attempts=2)
    h = Harness(bot)
    h.turn(StartScenario("s"))
    responses, out = h.turn(SetSlot("c", "abc"))  # lowercase: no match
    assert responses == ["ask"] and any(isinstance(e, SlotRejected) for e in out.events)
    assert h.state.top.attempts == 1
    assert h.turn(Fallback())[0] == ["fallback"] and h.state.stack == []


def test_stack_full_rejects_new_scenario():
    bot = make_bot({"s": {"slots": {"c": {"entity": "code", "ask": "ask"}}, "steps": [{"collect": "c"}]},
                    "t": {"steps": [{"respond": "ok"}]}},
                   {"ask": {"text": "?"}, "ok": {"text": "ok"}}, max_stack_depth=1)
    h = Harness(bot)
    h.turn(StartScenario("s"))
    responses, out = h.turn(StartScenario("t"))
    assert any(isinstance(e, CommandRejected) and "full" in e.reason for e in out.events)
    assert responses == ["ask"]  # the pending question is asked again


def test_clarify_uses_intent_labels_or_ids():
    bot = make_bot({"s": {"steps": [{"respond": "ok"}]}}, {"ok": {"text": "ok"}})
    _, out = Harness(bot).turn(Clarify(("a", "b", "a")))
    assert out.messages[0].quick_replies == ("에이", "b")


def test_registry_errors():
    reg = ActionRegistry()
    reg.register("ok", lambda a: {"v": 1})
    with pytest.raises(ValueError):
        reg.register("ok", lambda a: {})
    with pytest.raises(ActionError, match="unknown action"):
        reg.run("missing", {})
    reg.register("bad_type", lambda a: [1, 2])  # type: ignore[arg-type,return-value]
    with pytest.raises(ActionError, match="expected a mapping"):
        reg.run("bad_type", {})
    reg.register("not_json", lambda a: {"x": object()})
    with pytest.raises(ActionError, match="JSON"):
        reg.run("not_json", {})
    assert reg.run("ok", {}) == {"v": 1} and reg.names() == {"ok", "bad_type", "not_json"}


def test_registry_validates_args_with_model():
    from pydantic import BaseModel

    class Args(BaseModel):
        location: str

    reg = ActionRegistry()
    reg.register("w", lambda a: {"ok": True}, Args)
    assert reg.run("w", {"location": "서울"}) == {"ok": True}
    with pytest.raises(ActionError, match="bad arguments"):
        reg.run("w", {})


CHAIN_BOT = dict(
    scenarios={
        "parent": {"steps": [{"respond": "start"}, {"call": "child"}, {"respond": "after"}]},
        "child": {"slots": {"c": {"entity": "code", "ask": "ask"}}, "steps": [{"collect": "c"}, {"action": "fetch"}]},
        "other": {"steps": [{"respond": "other"}]},
    },
    responses={"start": {"text": "start"}, "after": {"text": "after"}, "ask": {"text": "?"}, "other": {"text": "other"}},
)


def test_child_action_failure_cancels_the_calling_parent_too():
    h = Harness(make_bot(**CHAIN_BOT), registry(fetch=StubAction(error=RuntimeError("down"))))
    assert h.turn(StartScenario("parent"))[0] == ["start", "ask"]
    responses, out = h.turn(SetSlot("c", "ABC"))
    assert responses == ["error"] and h.state.stack == []  # no "after", no second error
    assert [e.scenario for e in out.events if isinstance(e, ScenarioCancelled)] == ["child", "parent"]


def test_user_cancel_inside_called_child_ends_the_whole_task():
    h = Harness(make_bot(**CHAIN_BOT), registry(fetch=StubAction({"ok": 1})))
    h.turn(StartScenario("parent"))
    assert h.turn(CancelScenario())[0] == ["cancelled"] and h.state.stack == []


def test_cancel_of_a_digression_resumes_the_scenario_below():
    h = Harness(make_bot(**CHAIN_BOT), registry(fetch=StubAction({"ok": 1})))
    h.turn(StartScenario("child"))
    h.turn(StartScenario("other"))  # completes immediately, child asks again
    assert [f.scenario for f in h.state.stack] == ["child"]
    h.turn(StartScenario("parent"))  # user digression: parent -> called child is already running? no: parent starts
    assert h.turn(CancelScenario())[0][0] == "cancelled"
    assert [f.scenario for f in h.state.stack] == ["child"]


def test_setslot_reaches_a_frame_below_the_top():
    bot = make_bot({"outer": {"slots": {"c": {"entity": "code", "ask": "ask"}}, "steps": [{"collect": "c"}, {"respond": "ok"}]},
                    "inner": {"slots": {"d": {"entity": "code", "ask": "ask2"}}, "steps": [{"collect": "d"}]}},
                   {"ask": {"text": "c?"}, "ask2": {"text": "d?"}, "ok": {"text": "ok"}})
    h = Harness(bot)
    h.turn(StartScenario("outer"))
    h.turn(StartScenario("inner"))
    h.turn(SetSlot("c", "XYZ"))
    assert h.state.stack[0].slots == {"c": "XYZ"} and h.state.top.scenario == "inner"


def test_clarify_while_waiting_asks_only_the_clarification():
    bot = make_bot({"s": {"slots": {"c": {"entity": "code", "ask": "ask"}}, "steps": [{"collect": "c"}]}}, {"ask": {"text": "?"}})
    h = Harness(bot)
    h.turn(StartScenario("s"))
    responses, _ = h.turn(Clarify(("a", "b")))
    assert responses == ["clarify"] and h.state.top.awaiting == "c" and h.state.top.attempts == 0


@pytest.mark.parametrize("cmd, reason", [(RerunScenario("s", "nope", "ABC"), "no slot"), (RerunScenario("s", "c", "bad"), "does not match")])
def test_rerun_validates_before_starting(cmd, reason):
    bot = make_bot({"s": {"slots": {"c": {"entity": "code", "ask": "ask"}}, "steps": [{"collect": "c"}, {"respond": "ok"}]}},
                   {"ask": {"text": "?"}, "ok": {"text": "ok"}})
    h = Harness(bot)
    h.turn(StartScenario("s", {"c": "ABC"}))
    responses, out = h.turn(cmd)
    assert responses == ["fallback"] and h.state.stack == []
    assert any(isinstance(e, CommandRejected) and reason in e.reason for e in out.events)


def test_action_results_are_stored_as_json_would_replay_them():
    bot = make_bot({"s": {"steps": [{"action": "a"}]}})
    h = Harness(bot, registry(a=lambda args: {"pair": (1, 2), 3: "x"}))
    _, out = h.turn(StartScenario("s"))
    executed = [e for e in out.events if type(e).__name__ == "ActionExecuted"][0]
    assert executed.result == {"pair": [1, 2], "3": "x"}


@pytest.mark.parametrize("bad", [{"v": float("nan")}, {"v": float("inf")}])
def test_non_finite_action_results_fail(bad):
    with pytest.raises(ActionError):
        registry(a=lambda args: bad).run("a", {})
