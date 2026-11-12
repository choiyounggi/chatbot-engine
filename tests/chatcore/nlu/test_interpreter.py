"""Context-first interpretation rules 1-6 (docs/engine-core-plan.md §3.2)."""

import datetime as dt
import random

import pytest

from chatcore.bot.schema import Escape, Scenario, System
from chatcore.dialogue.commands import (
    CancelScenario,
    Clarify,
    Fallback,
    RerunScenario,
    Restart,
    SetSlot,
    StartScenario,
)
from chatcore.dialogue.commands import Escape as EscapeCommand
from chatcore.dialogue.events import PromptIssued, ScenarioCompleted, ScenarioStarted, UserUttered
from chatcore.dialogue.state import fold
from chatcore.dialogue.turn import TurnProcessor
from chatcore.nlu.interpreter import NluInterpreter
from chatcore.nlu.model import OOS, CachedEmbedder, IntentScore, NluModel, NluResult
from chatcore.nlu.train import train
from chatcore.store.memory import InMemoryStore

TODAY = dt.date(2026, 11, 10)


def result(extractor, text, intents, similarity=0.9, matched_by="classifier"):
    scores = tuple(IntentScore(i, s) for i, s in intents)
    return NluResult(text, scores, similarity, tuple(extractor.extract(text, TODAY)), matched_by)


def interp(extractor, intents, **kw):
    return NluInterpreter(lambda text: result(extractor, text, intents, **kw))


def waiting_for_location():
    return fold([UserUttered(text="날씨"), ScenarioStarted(scenario="check_weather"),
                 PromptIssued(scenario="check_weather", slot="location")])


def finished_weather(turns_since):
    events = [UserUttered(text="서울 날씨"), ScenarioStarted(scenario="check_weather"), ScenarioCompleted(scenario="check_weather")]
    return fold(events + [UserUttered(text="...")] * turns_since)


@pytest.mark.parametrize("text, cmd", [("취소", CancelScenario()), ("그만할래!", CancelScenario()), ("처음으로", Restart()),
                                       (" 처음부터 다시. ", Restart())])
def test_rule1_system_phrases(sample_bot, extractor, text, cmd):
    out = interp(extractor, [("weather", 0.99)]).interpret(text, fold([]), sample_bot)
    assert out.commands == [cmd] and out.info["rule"] == "system"


def test_rule2_pending_slot_is_filled(sample_bot, extractor):
    out = interp(extractor, [(OOS, 0.6), ("weather", 0.3)]).interpret("부산", waiting_for_location(), sample_bot)
    assert out.commands == [SetSlot("location", "부산")] and out.info["rule"] == "slot"


def test_rule2_same_intent_with_entity_only_fills_the_slot(sample_bot, extractor):
    out = interp(extractor, [("weather", 0.95)]).interpret("부산 날씨 알려줘", waiting_for_location(), sample_bot)
    assert out.commands == [SetSlot("location", "부산")]


def test_rule2_confident_other_intent_switches(sample_bot, extractor):
    out = interp(extractor, [("temperature", 0.9), ("weather", 0.05)]).interpret("부산 기온", waiting_for_location(), sample_bot)
    assert out.commands == [StartScenario("check_temperature", {"location": "부산"})] and out.info["rule"] == "intent"


def test_rule2_other_intent_below_switch_threshold_still_fills(sample_bot, extractor):
    out = interp(extractor, [("temperature", 0.7), ("weather", 0.2)]).interpret("부산 기온", waiting_for_location(), sample_bot)
    assert out.commands == [SetSlot("location", "부산")]


def test_rule2_wrong_entity_type_does_not_fill(sample_bot, extractor):
    out = interp(extractor, [(OOS, 0.9)]).interpret("3시", waiting_for_location(), sample_bot)
    assert out.commands == [Fallback()]


def test_rule3_intent_with_slots_from_text(sample_bot, extractor):
    out = interp(extractor, [("weather", 0.9), ("temperature", 0.05)]).interpret("제주도 날씨", fold([]), sample_bot)
    assert out.commands == [StartScenario("check_weather", {"location": "제주"})]


def test_rule3_ignores_entities_the_scenario_does_not_use(sample_bot, extractor):
    out = interp(extractor, [("greeting", 0.9)]).interpret("세종대왕 안녕", fold([]), sample_bot)
    assert out.commands == [StartScenario("greet")]


@pytest.mark.parametrize("intents, similarity", [([("weather", 0.2)], 0.9), ([("weather", 0.9)], 0.1), ([(OOS, 0.9)], 0.9)])
def test_rule3_not_confident(sample_bot, extractor, intents, similarity):
    out = interp(extractor, intents, similarity=similarity).interpret("아무 말", fold([]), sample_bot)
    assert out.commands == [Fallback()]


def test_rule4_rerun_recent_scenario(sample_bot, extractor):
    out = interp(extractor, [(OOS, 0.3)]).interpret("부산은?", finished_weather(turns_since=1), sample_bot)
    assert out.commands == [RerunScenario("check_weather", "location", "부산")] and out.info["rule"] == "rerun"


def test_rule4_window_expired(sample_bot, extractor):
    out = interp(extractor, [(OOS, 0.3)]).interpret("부산은?", finished_weather(turns_since=sample_bot.settings.rerun_window + 1), sample_bot)
    assert out.commands == [Fallback()]


def test_rule5_clarify_close_intents(sample_bot, extractor):
    out = interp(extractor, [("weather", 0.45), ("temperature", 0.40)]).interpret("밖에 어때", fold([]), sample_bot)
    assert out.commands == [Clarify(("weather", "temperature"))] and out.info["rule"] == "clarify"


def test_rule5_never_offers_out_of_scope_as_a_choice(sample_bot, extractor):
    # An out_of_scope runner-up is not a choice to offer; the top intent decides by the
    # same thresholds Phase 0 measured (probability + similarity), so it starts.
    out = interp(extractor, [("weather", 0.45), (OOS, 0.40)]).interpret("밖에 어때", fold([]), sample_bot)
    assert out.commands == [StartScenario("check_weather")]
    low = interp(extractor, [("weather", 0.25), (OOS, 0.24)]).interpret("밖에 어때", fold([]), sample_bot)
    assert low.commands == [Fallback()]


def test_exact_match_is_never_ambiguous(sample_bot, extractor):
    out = interp(extractor, [("weather", 1.0), ("temperature", 0.95)], matched_by="exact").interpret("날씨", fold([]), sample_bot)
    assert out.commands == [StartScenario("check_weather")]


def test_info_is_logged(sample_bot, extractor):
    info = interp(extractor, [("weather", 0.9)]).interpret("부산 날씨", fold([]), sample_bot).info
    assert info["intents"][0] == ["weather", 0.9] and info["entities"][0]["value"] == "부산"


# ---------------------------------------------------------------- end to end with a trained model

@pytest.fixture(scope="module")
def nlu_model(sample_bot, fake_embedder, extractor, tmp_path_factory):
    directory, _ = train(sample_bot, CachedEmbedder(fake_embedder), tmp_path_factory.mktemp("art"))
    return NluModel.load(sample_bot, directory, fake_embedder, extractor, today=lambda: TODAY)


def test_conversation_through_nlu(sample_bot, registry, nlu_model):
    """Exact-match sentences make the intent deterministic regardless of the embedder."""
    store = InMemoryStore()
    proc = TurnProcessor(sample_bot, store, NluInterpreter(nlu_model.parse), registry, rng=random.Random(0))
    said = lambda text: [m.response for m in proc.handle("s", text).messages]  # noqa: E731
    assert said("안녕하세요") == ["greeting"]
    assert said("지금 몇 시야?") == ["time_report"]
    assert said("오늘 날씨 알려줘") == ["ask_location"]  # exact example without a location
    assert said("부산") == ["weather_report"]
    assert said("제주는?") == ["weather_report"]  # rule 4: re-run with a new location
    assert said("취소") == ["fallback"]  # nothing to cancel
    logged = [e for e in store.events("s") if type(e).__name__ == "CommandsReceived"]
    assert [e.nlu["rule"] for e in logged] == ["intent", "intent", "intent", "slot", "rerun", "system"]


def test_rule4_skipped_when_clearly_out_of_scope(sample_bot, extractor):
    out = interp(extractor, [(OOS, 0.95)]).interpret("부산 맛집 추천해줘", finished_weather(turns_since=1), sample_bot)
    assert out.commands == [Fallback()]


def test_rule5_needs_the_same_gate_as_rule3(sample_bot, extractor):
    low_sim = interp(extractor, [("weather", 0.45), ("temperature", 0.40)], similarity=0.2)
    assert low_sim.interpret("밖에 어때", fold([]), sample_bot).commands == [Fallback()]
    low_prob = interp(extractor, [("weather", 0.25), ("temperature", 0.24)])
    assert low_prob.interpret("밖에 어때", fold([]), sample_bot).commands == [Fallback()]


@pytest.mark.parametrize("text", ["부산은?", "제주도는 어때?", "그럼 광주는요"])
def test_entity_only_followup_beats_a_confident_spurious_intent(sample_bot, extractor, text):
    # the real model scores "부산은?" as greeting 0.42 (measured); the follow-up must still win
    out = interp(extractor, [("greeting", 0.42)]).interpret(text, finished_weather(turns_since=1), sample_bot)
    assert out.commands[0].__class__.__name__ == "RerunScenario" and out.info["rule"] == "rerun"


def test_entity_only_input_cannot_start_a_scenario_without_that_slot(sample_bot, extractor):
    out = interp(extractor, [("greeting", 0.42)]).interpret("부산은?", fold([]), sample_bot)
    assert out.commands == [Fallback()]
    weather = interp(extractor, [("weather", 0.6)]).interpret("부산은?", fold([]), sample_bot)
    assert weather.commands == [StartScenario("check_weather", {"location": "부산"})]


@pytest.mark.parametrize("text, expected", [("부산은?", ["부산"]), ("인천도", ["인천"]), ("부산 날씨", []),
                                            ("세종대왕은 누구야", []), ("안녕", [])])
def test_entity_only_detection(extractor, text, expected):
    from chatcore.nlu.interpreter import entity_only

    assert [m.value for m in entity_only(text, tuple(extractor.extract(text, TODAY)))] == expected


def test_entity_only_followup_stays_on_the_finished_scenario(sample_bot, extractor):
    # Weather just finished; "부산은?" means "and Busan's weather?" even if the classifier
    # leans to temperature (whose scenario also takes a location).
    out = interp(extractor, [("temperature", 0.6)]).interpret("부산은?", finished_weather(turns_since=1), sample_bot)
    assert out.commands == [RerunScenario("check_weather", "location", "부산")]


# ---------------------------------------------------------------- E-B: escape phrases and free-text slots


def with_escape_and_memo(bot, phrases):
    """The sample bot plus escape phrases and a scenario that asks for any text."""
    memo = Scenario.model_validate({"slots": {"note": {"entity": "sys.text", "ask": "ask_location"}},
                                    "steps": [{"collect": "note"}]})
    return bot.model_copy(update={"system": System(escape=Escape(phrases=phrases)),
                                  "scenarios": {**bot.scenarios, "memo": memo}})


def waiting_for_note():
    return fold([UserUttered(text="메모"), ScenarioStarted(scenario="memo"), PromptIssued(scenario="memo", slot="note")])


@pytest.mark.parametrize("text", ["나가기", " 나가기! ", "취소"])
def test_rule1_escape_phrases_come_first(sample_bot, extractor, text):
    bot = with_escape_and_memo(sample_bot, ["나가기", "취소"])  # the bot's escape wins over the built-in cancel
    out = interp(extractor, [("weather", 0.99)]).interpret(text, waiting_for_note(), bot)
    assert out.commands == [EscapeCommand()] and out.info["rule"] == "system"


def test_without_escape_phrases_cancel_is_unchanged(sample_bot, extractor):
    out = interp(extractor, [(OOS, 0.9)]).interpret("나가기", fold([]), sample_bot)
    assert out.commands == [Fallback()]


def test_a_pending_free_text_slot_takes_the_whole_answer(sample_bot, extractor):
    bot = with_escape_and_memo(sample_bot, ["나가기"])
    out = interp(extractor, [(OOS, 0.6)]).interpret("내일  우유 사기 ", waiting_for_note(), bot)
    assert out.commands == [SetSlot("note", "내일 우유 사기")] and out.info["rule"] == "slot"


def test_a_free_text_slot_yields_to_a_confident_other_intent_and_to_cancel(sample_bot, extractor):
    bot = with_escape_and_memo(sample_bot, ["나가기"])
    switched = interp(extractor, [("weather", 0.95)]).interpret("부산 날씨 알려줘", waiting_for_note(), bot)
    assert switched.commands == [StartScenario("check_weather", {"location": "부산"})]
    assert interp(extractor, [(OOS, 0.6)]).interpret("취소", waiting_for_note(), bot).commands == [CancelScenario()]


@pytest.mark.parametrize("text", ["주말은 어때?", "이번 주는 어때?", "25도는?", "50%"])
def test_new_entity_types_do_not_change_an_old_bots_decisions(sample_bot, extractor, text):
    # the sample bot's slots use none of the E9 entities: they must not make the input "entity-only"
    out = interp(extractor, [("weather", 0.95)]).interpret(text, fold([]), sample_bot)
    assert out.commands == [StartScenario("check_weather", {})] and out.info["rule"] == "intent"


def test_new_entity_types_count_for_bots_whose_slots_use_them(sample_bot, extractor):
    plan = Scenario.model_validate({"trigger": {"intent": "help"}, "slots": {
        "when": {"entity": "sys.date.period", "ask": "ask_location"}}, "steps": [{"collect": "when"}]})
    bot = sample_bot.model_copy(update={"scenarios": {**sample_bot.scenarios, "show_help": plan}})
    out = interp(extractor, [("help", 0.95)]).interpret("다음 주 일정 도와줘", fold([]), bot)
    assert out.commands == [StartScenario("show_help", {"when": "2026-11-16/2026-11-22"})]


def test_punctuation_only_escape_phrases_never_match(sample_bot, extractor):
    bot = with_escape_and_memo(sample_bot, ["...", "나가기"])
    out = interp(extractor, [(OOS, 0.9)]).interpret("???", waiting_for_note(), bot)
    assert out.commands == [SetSlot("note", "???")]  # an answer, not an escape


def test_with_a_speller_a_free_text_slot_keeps_what_was_typed_and_the_knowledge_base_gets_the_fix(sample_bot, extractor):
    from chatcore.nlu.spelling import Speller

    class Recorder:
        asked: list[str] = []

        def consult(self, text, rule, similarity, vector):
            self.asked.append(text)
            return None, None

    bot = with_escape_and_memo(sample_bot, ["나가기"])
    speller = Speller(["우유 사기", "오늘 날씨"])  # no language model: every one-jamo match is taken
    memo = NluInterpreter(lambda t: result(extractor, t, [(OOS, 0.6)]), None, speller)
    out = memo.interpret("우유 사가", waiting_for_note(), bot)
    assert out.commands == [SetSlot("note", "우유 사가")] and out.info["corrected"]["text"] == "우유 사기"
    kb = Recorder()
    NluInterpreter(lambda t: result(extractor, t, [(OOS, 0.6)]), kb, speller).interpret("오늘 날시", fold([]), sample_bot)
    assert kb.asked == ["오늘 날씨"]
