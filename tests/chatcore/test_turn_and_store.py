"""Turn processor and tracker stores (in-memory always; Postgres when a DB is configured)."""

import os
import random
import threading
import uuid

import pytest

from chatcore.bot.loader import BotError
from chatcore.dialogue.commands import StartScenario
from chatcore.dialogue.events import BotUttered, SessionStarted, TurnFailed, UserUttered
from chatcore.dialogue.turn import MAX_INPUT_CHARS, ScriptedInterpreter, TurnProcessor
from chatcore.dialogue.values import ActionRegistry
from chatcore.store.base import SessionBusy
from chatcore.store.memory import InMemoryStore


def processor(bot, registry, store=None, interpreter=None):
    script = {"날씨": [StartScenario("check_weather")], "안녕": [StartScenario("greet")]}
    return TurnProcessor(bot, store or InMemoryStore(), interpreter or ScriptedInterpreter(script), registry,
                         rng=random.Random(0))


def test_rejects_bot_whose_actions_are_not_registered(sample_bot):
    with pytest.raises(BotError, match="unknown action"):
        processor(sample_bot, ActionRegistry())


@pytest.mark.parametrize("text", ["", "   ", "x" * (MAX_INPUT_CHARS + 1), None])
def test_invalid_input_is_answered_without_storing(sample_bot, registry, text):
    store = InMemoryStore()
    result = processor(sample_bot, registry, store).handle("s", text)
    assert [m.response for m in result.messages] == ["input_invalid"] and result.events == []
    assert store.events("s") == []


def test_input_at_max_length_is_accepted(sample_bot, registry):
    result = processor(sample_bot, registry).handle("s", "가" * MAX_INPUT_CHARS)
    assert [m.response for m in result.messages] == ["fallback"]


def test_session_started_once_and_whitespace_normalized(sample_bot, registry):
    store = InMemoryStore()
    proc = processor(sample_bot, registry, store)
    proc.handle("s", "  안녕  ")
    proc.handle("s", "안녕")
    events = store.events("s")
    assert sum(isinstance(e, SessionStarted) for e in events) == 1
    assert [e.text for e in events if isinstance(e, UserUttered)] == ["안녕", "안녕"]
    assert events[0].bot_version == sample_bot.version


class Exploding:
    def interpret(self, text, state, bot):
        raise RuntimeError("bug")


def test_interpreter_crash_answers_error_and_keeps_session_usable(sample_bot, registry):
    store = InMemoryStore()
    result = processor(sample_bot, registry, store, Exploding()).handle("s", "아무말")
    assert [m.response for m in result.messages] == ["error"]
    assert any(isinstance(e, TurnFailed) and e.reason == "RuntimeError" for e in store.events("s"))
    assert [m.response for m in processor(sample_bot, registry, store).handle("s", "안녕").messages] == ["greeting"]


def test_engine_crash_discards_partial_engine_events(sample_bot, registry, monkeypatch):
    import chatcore.dialogue.engine as engine_mod

    real = engine_mod.render_response
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # the engine has already emitted ScenarioStarted/PromptIssued by now
            raise RuntimeError("render bug")
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "render_response", flaky)
    store = InMemoryStore()
    processor(sample_bot, registry, store).handle("s", "날씨")
    kinds = [type(e).__name__ for e in store.events("s")]
    assert kinds == ["SessionStarted", "UserUttered", "TurnFailed", "BotUttered"]


def test_bot_edit_restarts_sessions_whose_scenarios_vanished(sample_bot, registry):
    store = InMemoryStore()
    processor(sample_bot, registry, store).handle("s", "날씨")  # check_weather waits for a location
    edited = sample_bot.model_copy(update={
        "version": "edited",
        "scenarios": {k: v for k, v in sample_bot.scenarios.items() if k != "check_weather"},
        "responses": {k: v for k, v in sample_bot.responses.items() if k != "weather_report"},
    })
    result = processor(edited, registry, store).handle("s", "안녕")
    assert [m.response for m in result.messages] == ["restarted", "greeting"]  # the user is told
    kinds = [type(e).__name__ for e in result.events]
    assert kinds[:3] == ["SessionRestarted", "BotUttered", "SessionStarted"]


def test_new_version_restarts_a_scenario_in_progress_even_if_it_still_exists(sample_bot, registry):
    store = InMemoryStore()
    processor(sample_bot, registry, store).handle("s", "날씨")  # check_weather waits for a location
    edited = sample_bot.model_copy(update={"version": "edited"})  # its steps may differ; the engine cannot tell
    result = processor(edited, registry, store).handle("s", "부산")
    assert [m.response for m in result.messages][0] == "restarted"


def test_new_version_keeps_an_idle_session_without_a_notice(sample_bot, registry):
    store = InMemoryStore()
    processor(sample_bot, registry, store).handle("s", "안녕")  # greet finishes; nothing in progress
    edited = sample_bot.model_copy(update={"version": "edited"})
    result = processor(edited, registry, store).handle("s", "안녕")
    assert [m.response for m in result.messages] == ["greeting"]
    assert [type(e).__name__ for e in result.events][:2] == ["SessionStarted", "UserUttered"]


def test_events_are_persisted_only_when_the_turn_completes():
    store = InMemoryStore()
    with pytest.raises(RuntimeError):
        with store.session("s") as tracker:
            tracker.append([UserUttered(text="x")])
            raise RuntimeError("boom")
    assert store.events("s") == []


def test_bot_messages_match_bot_uttered_events(sample_bot, registry):
    store = InMemoryStore()
    result = processor(sample_bot, registry, store).handle("s", "날씨")
    uttered = [e for e in store.events("s") if isinstance(e, BotUttered)]
    assert [(m.response, m.text) for m in result.messages] == [(e.response, e.text) for e in uttered]


@pytest.mark.parametrize("sid", ["", "  ", "x" * 129, "a\x00b", "line\nbreak"])
def test_session_id_validation(sid):
    with pytest.raises(ValueError):
        with InMemoryStore().session(sid):
            pass


# ---------------------------------------------------------------- stores (shared contract)

DB_URL = os.environ.get("CHATCORE_TEST_DATABASE_URL")


def postgres_store():
    if not DB_URL:
        if os.environ.get("CI"):
            pytest.fail("CHATCORE_TEST_DATABASE_URL must be set in CI")
        pytest.skip("set CHATCORE_TEST_DATABASE_URL (make test-db) to run Postgres tests")
    from chatcore.store.postgres import PostgresStore

    store = PostgresStore(DB_URL)
    store.ensure_schema()
    return store


STORES = {"memory": InMemoryStore, "postgres": postgres_store}


@pytest.fixture(params=sorted(STORES))
def store(request):
    return STORES[request.param]()


def test_store_round_trip_and_ordering(store):
    sid = f"t-{uuid.uuid4()}"
    with store.session(sid) as tracker:
        assert tracker.events == []
        tracker.append([SessionStarted(bot_id="b", bot_version="v"), UserUttered(text="하나")])
    with store.session(sid) as tracker:
        assert [type(e).__name__ for e in tracker.events] == ["SessionStarted", "UserUttered"]
        tracker.append([UserUttered(text="둘")])
    assert [e.text for e in store.events(sid) if isinstance(e, UserUttered)] == ["하나", "둘"]


def test_store_rolls_back_on_error(store):
    sid = f"t-{uuid.uuid4()}"
    with pytest.raises(RuntimeError):
        with store.session(sid) as tracker:
            tracker.append([UserUttered(text="x")])
            raise RuntimeError("boom")
    assert store.events(sid) == []
    with store.session(sid) as tracker:  # the lock was released by the failed turn
        assert tracker.events == []


def test_store_rejects_concurrent_turn_of_same_session(store):
    sid = f"t-{uuid.uuid4()}"
    entered, release = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def first():
        try:
            with store.session(sid):
                entered.set()
                release.wait(5)
        except BaseException as exc:  # surfaced by the assertion below
            errors.append(exc)

    t = threading.Thread(target=first)
    t.start()
    assert entered.wait(5)
    try:
        with pytest.raises(SessionBusy):
            with store.session(sid):
                pass
        with store.session(f"{sid}-other"):  # other sessions are not blocked
            pass
    finally:
        release.set()
        t.join(5)
    assert errors == []
    with store.session(sid):  # lock released after the first turn
        pass


def test_processor_with_store(sample_bot, registry, store):
    sid = f"t-{uuid.uuid4()}"
    proc = processor(sample_bot, registry, store)
    assert [m.response for m in proc.handle(sid, "날씨").messages] == ["ask_location"]
    assert any(isinstance(e, SessionStarted) for e in store.events(sid))


def test_recent_filters_by_prefix_and_type_newest_first(store):
    run = uuid.uuid4().hex[:8]
    a, look_alike = f"a_{run}:s1", f"ax{run}:s1"  # "_" must not act as a LIKE wildcard
    for sid, text in ((a, "first"), (look_alike, "other bot"), (a, "second")):
        with store.session(sid) as tracker:
            tracker.append([UserUttered(text=text), BotUttered(response="r", text="r")])
    found = store.recent(f"a_{run}:", "user_uttered", 10)
    assert [(sid, e.text) for sid, e in found] == [(a, "second"), (a, "first")]
    assert [e.text for _, e in store.recent(f"a_{run}:", "user_uttered", 1)] == ["second"]
    assert store.recent(f"a_{run}:", "user_uttered", 0) == [] and store.recent(f"none_{run}:", "user_uttered", 5) == []
