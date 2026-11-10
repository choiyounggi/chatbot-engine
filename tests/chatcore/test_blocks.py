"""Builder stage E-B: Kakao-style blocks (E7) and rich responses (E8).

Welcome / escape / session timeout, block groups, optional slots with defaults
and validation, direct block starts (quick replies and buttons), and image /
card / list / carousel responses. Every new document field is optional: the
sample bot (which uses none of them) must behave exactly as before.
"""

import datetime as dt
import random

import pytest
from pydantic import ValidationError

from chatcore.bot.loader import validate
from chatcore.bot.schema import BotDefinition, Response
from chatcore.dialogue.commands import Escape, SetSlot, StartScenario, from_dict, to_dict
from chatcore.dialogue.engine import ScenarioEngine
from chatcore.dialogue.events import (
    ActionFailed,
    CommandsReceived,
    PromptIssued,
    ScenarioCancelled,
    ScenarioStarted,
    SessionRestarted,
    SessionStarted,
    SlotRejected,
    UserUttered,
)
from chatcore.dialogue.state import DialogueState, fold
from chatcore.dialogue.turn import ScriptedInterpreter, TurnProcessor, UnknownBlock
from chatcore.dialogue.values import ActionError, ActionRegistry, StubAction
from chatcore.store.memory import InMemoryStore

REQUIRED = {k: {"text": k} for k in ("fallback", "cancelled", "restarted", "error", "clarify", "input_invalid")}


def doc(scenarios: dict, responses: dict | None = None, **extra) -> dict:
    return {"id": "t", "intents": {"a": {"label": "에이", "examples": ["a"]}, "b": {"examples": ["b"]}},
            "entities": {"code": {"kind": "regex", "pattern": r"[A-Z]{3}"}},
            "scenarios": scenarios, "responses": {**REQUIRED, **(responses or {})}, **extra}


def bot(scenarios: dict, responses: dict | None = None, **extra) -> BotDefinition:
    return BotDefinition.model_validate({**doc(scenarios, responses, **extra), "version": "v"})


def registry(**actions) -> ActionRegistry:
    reg = ActionRegistry()
    for name, fn in actions.items():
        reg.register(name, fn)
    return reg


class Harness:
    """Engine turns with the replay invariant checked after each one."""

    def __init__(self, b: BotDefinition, actions: ActionRegistry | None = None):
        self.engine = ScenarioEngine(b, actions or ActionRegistry(), random.Random(0))
        self.state = DialogueState()
        self.events: list = []

    def turn(self, *commands):
        out = self.engine.run(self.state, list(commands))
        self.events += out.events
        assert fold(self.events).stack == self.state.stack
        return [m.response for m in out.messages], out


CARD = {"title": "{{ slots.code }} 안내", "description": "설명",
        "thumbnail": {"image_url": "https://img.example/{{ slots.code }}.png"},
        "buttons": [{"label": "열기", "action": "web_link", "web_link_url": "https://example.com"},
                    {"label": "다시", "action": "block", "block_id": "main"},
                    {"label": "전화", "action": "phone", "phone_number": "+82 2-123-4567"},
                    {"label": "말하기", "action": "message", "message_text": "도와줘"}]}


# ---------------------------------------------------------------- E8 schema


@pytest.mark.parametrize("body, kind", [
    ({"simple_image": {"image_url": "https://x/a.png", "alt_text": "그림"}}, "simple_image"),
    ({"basic_card": CARD}, "basic_card"),
    ({"list_card": {"header": {"title": "목록"}, "items": [{"title": "하나", "link": {"web": "https://x"}}]}}, "list_card"),
    ({"carousel": {"items": [CARD, {"title": "둘"}]}}, "carousel"),
    ({"text": "그냥 글"}, "text"),
])
def test_each_rich_kind_parses_as_one_bubble(body, kind):
    r = Response.model_validate(body)
    assert r.kind == kind and (r.rich is None) == (kind == "text")


@pytest.mark.parametrize("body, message", [
    ({"text": "a", "basic_card": {"title": "t"}}, "exactly one"),
    ({}, "exactly one"),
    ({"simple_image": {"image_url": "ftp://x/a.png", "alt_text": "a"}}, "http"),
    ({"simple_image": {"image_url": "https://x/a.png", "alt_text": ""}}, "at least 1"),
    ({"basic_card": {"buttons": []}}, "title, a description or a thumbnail"),
    ({"basic_card": {"title": "t", "buttons": [{"label": "b", "action": "web_link"}]}}, "target field"),
    ({"basic_card": {"title": "t", "buttons": [{"label": "b", "action": "message", "block_id": "x"}]}}, "cannot set"),
    ({"basic_card": {"title": "t", "buttons": [{"label": "b", "action": "phone", "phone_number": "call me"}]}},
     "phone_number"),
    ({"basic_card": {"title": "t", "buttons": [{"label": "b", "action": "web_link", "web_link_url": "https://x",
                                                "message_text": "m"}]}}, "no message_text"),
    ({"list_card": {"header": {"title": "목록"}, "items": []}}, "at least 1"),
    ({"carousel": {"items": []}}, "at least 1"),
    ({"text": "a", "quick_replies": [{"label": "go", "action": "block"}]}, "target field"),
    ({"text": "a", "quick_replies": [{"label": " ", "action": "message"}]}, "label must not be empty"),
    ({"text": "a", "quick_replies": ["x" * 41]}, "1-40"),
])
def test_invalid_rich_bodies_are_rejected(body, message):
    with pytest.raises(ValidationError, match=message):
        Response.model_validate(body)


def test_texts_and_block_targets_cover_every_field():
    r = Response.model_validate({"carousel": {"items": [CARD]},
                                 "quick_replies": ["서울", {"label": "처음", "action": "block", "block_id": "home"},
                                                   {"label": "{{ slots.code }}", "message_text": "코드 {{ slots.code }}"}]})
    texts = r.texts()
    assert "{{ slots.code }} 안내" in texts and "https://img.example/{{ slots.code }}.png" in texts
    assert "코드 {{ slots.code }}" in texts and "web_link" not in texts  # actions are not text
    assert r.block_targets() == {"home", "main"}
    assert [q.label for q in r.replies()] == ["서울", "처음", "{{ slots.code }}"]


# ---------------------------------------------------------------- E7 schema


def test_slot_options_are_checked_by_the_schema():
    with pytest.raises(ValidationError, match="needs 'ask'"):
        bot({"s": {"slots": {"c": {"entity": "code"}}, "steps": [{"collect": "c"}]}})
    with pytest.raises(ValidationError, match="cannot have a default"):
        bot({"s": {"slots": {"c": {"entity": "code", "ask": "q", "default": "ABC"}}, "steps": [{"collect": "c"}]}},
            {"q": {"text": "q"}})
    with pytest.raises(ValidationError, match="not true/false"):
        bot({"s": {"slots": {"c": {"entity": "code", "required": False, "default": True}}, "steps": [{"collect": "c"}]}})
    with pytest.raises(ValidationError, match="at least 1"):
        bot({"s": {"group": "", "steps": [{"respond": "error"}]}})
    with pytest.raises(ValidationError, match="escape phrases must not be empty"):
        bot({"s": {"steps": [{"respond": "error"}]}}, system={"escape": {"phrases": [" "]}})
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        bot({"s": {"steps": [{"respond": "error"}]}}, system={"timeout": {"minutes": 0}})


def test_new_fields_are_optional_and_do_not_change_old_documents(sample_bot):
    assert sample_bot.system.welcome is None and sample_bot.system.escape is None
    assert all(sc.group is None for sc in sample_bot.scenarios.values())
    assert validate(sample_bot).errors == []


# ---------------------------------------------------------------- loader


def test_loader_reports_broken_references():
    b = bot({"s": {"slots": {"c": {"entity": "code", "required": False, "default": "abc",
                                   "validation": {"action": "nope", "fail": "missing"}}},
                   "steps": [{"collect": "c"}, {"respond": "card"}]}},
            {"card": {"basic_card": {"title": "{{ slots.zzz }}", "buttons": [
                {"label": "b", "action": "block", "block_id": "ghost"}]}},
             "bye": {"text": "{{ slots.c }} 안녕"}},
            system={"welcome": "w", "escape": {"phrases": ["그만"], "scenario": "e"},
                    "timeout": {"minutes": 5, "notice": "bye"}})
    errors = validate(b, actions=set()).errors
    expected = ["default 'abc' is not a code", "validated by unknown action 'nope'",
                "fails validation with unknown response 'missing'", "unknown variable {{ slots.zzz }}",
                "block button to unknown scenario 'ghost'", "system.welcome: unknown scenario 'w'",
                "system.escape: unknown scenario 'e'", "remove its variables"]
    for part in expected:
        assert any(part in e for e in errors), (part, errors)
    assert validate(bot({"s": {"steps": [{"respond": "error"}]}}),
                    actions=set()).errors == []  # and a clean bot has none
    unknown_notice = bot({"s": {"steps": [{"respond": "error"}]}}, system={"timeout": {"minutes": 5, "notice": "x"}})
    assert validate(unknown_notice).errors == ["system.timeout: unknown notice response 'x'"]


def test_kakao_limits_are_warnings_only():
    many = [f"q{i}" for i in range(11)]
    b = bot({"s": {"steps": [{"respond": "list"}] * 4}},
            {"list": {"list_card": {"header": {"title": "h"}, "items": [{"title": str(i)} for i in range(6)],
                                    "buttons": [{"label": "열네글자를넘는아주긴버튼이름입니다", "action": "message"}] * 4},
                      "quick_replies": many},
             "cards": {"carousel": {"items": [{"title": str(i)} for i in range(11)]}}})
    report = validate(b)
    assert report.errors == []
    for part in ("4 responses in one block", "11 quick replies", "longer than 14", "more than 3 buttons",
                 "6 list items", "11 carousel cards"):
        assert any(part in w for w in report.warnings), (part, report.warnings)


def test_escape_phrase_that_is_also_an_example_is_a_warning():
    b = bot({"s": {"steps": [{"respond": "error"}]}}, system={"escape": {"phrases": [" a! "]}})
    assert any("always escape: ['a']" in w for w in validate(b).warnings)


# ---------------------------------------------------------------- engine: slots


def slots_bot(**slot):
    return bot({"s": {"slots": {"c": {"entity": "code", **slot}}, "steps": [{"collect": "c"}, {"respond": "done"}]}},
               {"q": {"text": "코드?"}, "done": {"text": "코드 {{ slots.c }}"}, "bad": {"text": "없는 코드예요"},
                "oops": {"text": "확인 실패"}})


def test_optional_slot_is_never_asked_and_takes_its_default():
    h = Harness(slots_bot(required=False, default="ZZZ"))
    names, out = h.turn(StartScenario("s"))
    assert names == ["done"] and out.messages[0].text == "코드 ZZZ"
    names, out = h.turn(StartScenario("s", {"c": "ABC"}))  # a value in the user's words wins
    assert out.messages[0].text == "코드 ABC"


def test_optional_slot_without_default_stays_empty():
    h = Harness(slots_bot(required=False))
    names, _ = h.turn(StartScenario("s"))
    assert names == ["fallback"]  # "{{ slots.c }}" has no value: the response cannot be said
    assert h.state.stack == []


def test_validation_accepts_rejects_and_asks_again():
    check = StubAction(result={"valid": True})
    h = Harness(slots_bot(ask="q", validation={"action": "check", "fail": "bad"}), registry(check=check))
    assert h.turn(StartScenario("s"))[0] == ["q"]
    check.result = {"valid": False}
    names, out = h.turn(from_dict({"set": {"c": "XYZ"}}))
    assert names == ["bad", "q"] and h.state.top.attempts == 1  # a rejected answer is a failed attempt
    assert any(isinstance(e, SlotRejected) and e.reason == "rejected by check" for e in out.events)
    check.result = {"valid": True}
    assert h.turn(from_dict({"set": {"c": "ABC"}}))[0] == ["done"]
    assert check.calls[-1] == {"value": "ABC", "slot": "c"}


@pytest.mark.parametrize("result, error", [({"ok": 1}, None), ({}, ActionError("down"))])
def test_a_failing_validator_says_the_error_response_and_asks_again(result, error):
    check = StubAction(result=result, error=error)
    b = slots_bot(ask="q", validation={"action": "check", "fail": "bad"})
    b = b.model_copy(update={"scenarios": {"s": b.scenarios["s"].model_copy(update={"on_error": "oops"})}})
    h = Harness(b, registry(check=check))
    names, out = h.turn(StartScenario("s", {"c": "ABC"}))
    assert names == ["oops", "q"]  # the value was not the user's fault
    assert any(isinstance(e, ActionFailed) and e.action == "check" for e in out.events)
    assert "c" not in h.state.top.slots


def test_rerun_with_a_rejected_value_does_not_rerun():
    check = StubAction(result={"valid": True})
    h = Harness(slots_bot(ask="q", validation={"action": "check", "fail": "bad"}), registry(check=check))
    assert h.turn(StartScenario("s", {"c": "ABC"}))[0] == ["done"]
    check.result = {"valid": False}
    assert h.turn(from_dict({"rerun": "s", "slot": "c", "value": "XYZ"}))[0] == ["bad"]
    assert h.state.stack == []


# ---------------------------------------------------------------- engine: escape and rich responses


def escape_bot(scenario: str | None):
    return bot({"long": {"slots": {"c": {"entity": "code", "ask": "q"}}, "steps": [{"collect": "c"}]},
                "side": {"slots": {"d": {"entity": "code", "ask": "q"}}, "steps": [{"collect": "d"}]},
                "bye": {"steps": [{"respond": "bye"}]}},
               {"q": {"text": "?"}, "bye": {"text": "처음 화면으로 돌아갈게요"}},
               system={"escape": {"phrases": ["나가기"], "scenario": scenario}})


def test_escape_cancels_every_running_scenario_then_runs_the_escape_block():
    h = Harness(escape_bot("bye"))
    h.turn(StartScenario("long"))
    h.turn(StartScenario("side"))  # a digression on top of the first question
    names, out = h.turn(Escape())
    assert names == ["bye"] and h.state.stack == []
    assert [e.scenario for e in out.events if isinstance(e, ScenarioCancelled)] == ["side", "long"]


@pytest.mark.parametrize("running", [True, False])
def test_escape_without_a_block_says_cancelled(running):
    h = Harness(escape_bot(None))
    if running:
        h.turn(StartScenario("long"))
    assert h.turn(Escape())[0] == ["cancelled"] and h.state.stack == []


def test_escape_command_round_trips():
    assert from_dict(to_dict(Escape())) == Escape() and to_dict(Escape()) == {"escape": True}


def rich_bot():
    return bot({"main": {"slots": {"code": {"entity": "code", "ask": "q"}}, "steps": [{"collect": "code"}, {"respond": "card"}]},
                "pic": {"steps": [{"respond": "pic"}]}},
               {"q": {"text": "?"},
                "card": {"basic_card": {**CARD, "title": "{{ slots.code }} 안내"},
                         "quick_replies": [{"label": "다시", "action": "block", "block_id": "main"}, "그만"]},
                "pic": {"simple_image": {"image_url": "{{ slots.code }}", "alt_text": "그림"}}})


def test_rich_response_is_rendered_with_templates_and_a_text_summary():
    h = Harness(rich_bot())
    names, out = h.turn(StartScenario("main", {"code": "ABC"}))
    msg = out.messages[0]
    assert names == ["card"] and msg.kind == "basic_card"
    assert msg.rich.title == "ABC 안내" and msg.rich.thumbnail.image_url == "https://img.example/ABC.png"
    assert msg.text == "ABC 안내\n설명"
    assert msg.quick_replies == ("다시", "그만") and msg.replies[0].block_id == "main"


def test_a_template_that_breaks_a_url_falls_back():
    b = bot({"pic": {"slots": {"code": {"entity": "code", "required": False}}, "steps": [{"respond": "pic"}]}},
            {"pic": {"simple_image": {"image_url": "{{ slots.code }}", "alt_text": "그림"}}})
    h = Harness(b)
    assert h.turn(StartScenario("pic", {"code": "ABC"}))[0] == ["fallback"]  # "ABC" is not an image address


# ---------------------------------------------------------------- turn processor: open, blocks, timeout


class Clock:
    def __init__(self):
        self.now = dt.datetime(2026, 11, 10, 9, 0, tzinfo=dt.UTC)

    def __call__(self):
        return self.now


def processor(b: BotDefinition, script=None, clock=None, actions=None):
    store = InMemoryStore()
    return TurnProcessor(b, store, ScriptedInterpreter(script or {}), actions or ActionRegistry(),
                         rng=random.Random(0), clock=clock or Clock()), store


def welcome_bot(**system):
    return bot({"hello": {"steps": [{"respond": "hi"}]},
                "ask": {"slots": {"c": {"entity": "code", "ask": "q"}}, "steps": [{"collect": "c"}, {"respond": "done"}]},
                "menu": {"steps": [{"respond": "menu"}]}},
               {"hi": {"text": "어서 오세요", "quick_replies": [{"label": "메뉴", "action": "block", "block_id": "menu"}]},
                "q": {"text": "코드?"}, "done": {"text": "{{ slots.c }}"}, "menu": {"text": "메뉴예요"},
                "late": {"text": "시간이 지나 처음부터 다시 할게요"}},
               system=system)


def test_open_runs_the_welcome_block_once():
    proc, store = processor(welcome_bot(welcome="hello"))
    first = proc.open("s1")
    assert [m.response for m in first.messages] == ["hi"]
    assert isinstance(first.events[0], SessionStarted) and first.events[0].at == "2026-11-10T09:00:00+00:00"
    assert any(isinstance(e, CommandsReceived) and e.nlu == {"rule": "welcome"} for e in first.events)
    assert proc.open("s1").messages == [] and len(store.events("s1")) == len(first.events)  # already open


def test_open_without_a_welcome_block_only_starts_the_session():
    proc, store = processor(welcome_bot())
    assert proc.open("s1").messages == []
    assert [type(e) for e in store.events("s1")] == [SessionStarted]
    assert [m.response for m in proc.handle("s1", "a").messages] == ["fallback"]  # then chats as usual


def test_a_block_button_starts_its_scenario_without_understanding():
    proc, store = processor(welcome_bot(welcome="hello"))
    proc.open("s1")
    result = proc.handle("s1", "메뉴", block="menu")
    assert [m.response for m in result.messages] == ["menu"]
    received = next(e for e in result.events if isinstance(e, CommandsReceived))
    assert received.commands == [{"start": "menu"}] and received.nlu["rule"] == "block"


@pytest.mark.parametrize("block", ["ask", "nope"])
def test_only_blocks_a_response_offers_can_be_started_directly(block):
    proc, store = processor(welcome_bot(welcome="hello"))
    with pytest.raises(UnknownBlock):
        proc.handle("s1", "x", block=block)  # "ask" exists, but no button offers it
    assert store.events("s1") == []


def test_an_idle_session_with_a_question_pending_starts_over_with_a_notice():
    clock = Clock()
    proc, _ = processor(welcome_bot(timeout={"minutes": 10, "notice": "late"}), {"start": [StartScenario("ask")]},
                        clock)
    assert [m.response for m in proc.handle("s1", "start").messages] == ["q"]
    clock.now += dt.timedelta(minutes=10)  # exactly the limit: still the same conversation
    assert [m.response for m in proc.handle("s1", "start").messages] == ["q"]
    clock.now += dt.timedelta(minutes=10, seconds=1)
    result = proc.handle("s1", "ABC")
    assert [m.response for m in result.messages] == ["late", "fallback"]  # "ABC" no longer answers anything
    assert isinstance(result.events[0], SessionRestarted)


def test_an_idle_session_with_nothing_running_forgets_silently():
    clock = Clock()
    proc, _ = processor(welcome_bot(timeout={"minutes": 10}), {"go": [StartScenario("menu")]}, clock)
    proc.handle("s1", "go")
    clock.now += dt.timedelta(hours=1)
    result = proc.handle("s1", "go")
    assert [m.response for m in result.messages] == ["menu"]  # no notice: nothing was interrupted
    assert isinstance(result.events[0], SessionRestarted) and fold(result.events).completed[0].scenario == "menu"


def test_sessions_logged_without_times_never_time_out():
    clock = Clock()
    proc, store = processor(welcome_bot(timeout={"minutes": 1}), {"ABC": [SetSlot("c", "ABC")]}, clock)
    with store.session("s1") as tracker:  # written before events carried times
        tracker.append([SessionStarted(bot_id="t", bot_version="v"), UserUttered(text="x"),
                        ScenarioStarted(scenario="ask"), PromptIssued(scenario="ask", slot="c")])
    clock.now += dt.timedelta(days=1)
    assert [m.response for m in proc.handle("s1", "ABC").messages] == ["done"]  # still the same conversation


def test_no_timeout_configured_keeps_sessions_forever():
    clock = Clock()
    proc, _ = processor(welcome_bot(), {"start": [StartScenario("ask")]}, clock)
    proc.handle("s1", "start")
    clock.now += dt.timedelta(days=30)
    assert not any(isinstance(e, SessionRestarted) for e in proc.handle("s1", "start").events)


# ---------------------------------------------------------------- E9 entities in slots


def test_new_system_entities_fill_slots_and_check_defaults():
    b = bot({"s": {"slots": {"age": {"entity": "sys.number.age", "required": False, "default": "스무 살"},
                             "when": {"entity": "sys.date.period", "ask": "q"}},
                   "steps": [{"collect": "age"}, {"collect": "when"}, {"respond": "done"}]}},
            {"q": {"text": "언제요?"}, "done": {"text": "{{ slots.age }}세, {{ slots.when }}"}})
    assert validate(b).errors == []
    h = Harness(b)
    assert h.turn(StartScenario("s"))[0] == ["q"]
    rejected = h.turn(from_dict({"set": {"when": "2026-11-20/2026-11-10"}}))[1]  # end before start
    assert any(isinstance(e, SlotRejected) for e in rejected.events)
    names, out = h.turn(from_dict({"set": {"when": "2026-11-09/2026-11-15"}}))
    assert names == ["done"] and out.messages[0].text == "20세, 2026-11-09/2026-11-15"
    too_old = bot({"s": {"slots": {"age": {"entity": "sys.number.age", "required": False, "default": 200}},
                         "steps": [{"collect": "age"}]}})
    assert any("default 200 is not a sys.number.age" in e for e in validate(too_old).errors)


def test_punctuation_only_escape_phrase_is_an_error_and_ask_check_ignores_quick_replies():
    b = bot({"s": {"slots": {"c": {"entity": "code", "ask": "q"}}, "steps": [{"collect": "c"}]}},
            {"q": {"text": "코드?", "quick_replies": ["{{ slots.c }}"]}},
            system={"escape": {"phrases": ["...", "나가기"]}})
    errors = validate(b).errors
    assert errors == ["system.escape: phrases with only punctuation never match: ['...']"]  # no "asks ... shows it"


def test_long_quick_reply_labels_from_the_engine_reach_the_api():
    from chatcore.api import messages_out  # noqa: PLC0415
    from chatcore.dialogue.turn import TurnResult  # noqa: PLC0415
    from chatcore.dialogue.values import BotMessage, label_replies  # noqa: PLC0415

    (out,) = messages_out(TurnResult([BotMessage("clarify", "어느 쪽이요?", label_replies(["가" * 60]))], []))
    assert out.quick_replies == ["가" * 60] and out.quick_reply_items[0].label == "가" * 60
    assert out.model_validate(out.model_dump()).quick_reply_items[0].action == "message"
