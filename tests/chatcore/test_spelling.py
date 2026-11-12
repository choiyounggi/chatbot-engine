"""E-I: typo correction against the bot's own words."""

import pytest
from kiwipiepy import Kiwi

from chatcore.dialogue.state import DialogueState
from chatcore.knowledge import KnowledgeIndex
from chatcore.nlu.hashing import HashingEmbedder
from chatcore.nlu.interpreter import NluInterpreter
from chatcore.nlu.model import IntentScore, NluResult
from chatcore.nlu.spelling import Speller, _one_edit, jamo, kiwi_score, vocabulary

from .test_knowledge import KB

WORDS = ["오늘 날씨 알려줘", "날씨가 어때요", "부산 날씨", "고마워요", "고마워", "미세먼지 나쁨 기준", "먼저 확인하세요", "모임 일정"]


@pytest.fixture(scope="module")
def kiwi_speller():
    return Speller(WORDS, kiwi_score(Kiwi()))


def test_jamo_lines_up_initial_medial_final() -> None:
    assert jamo("날씨") == "ㄴㅏㄹㅆㅣ·" and jamo("A가") == "Aㄱㅏ·" and jamo("") == ""
    assert _one_edit(jamo("날시"), jamo("날씨")) and not _one_edit(jamo("날씨"), jamo("날씨"))
    assert _one_edit("abc", "ab") and _one_edit("ab", "abc") and not _one_edit("abc", "a") and not _one_edit("ab", "ba")


@pytest.mark.parametrize("typed, understood, changes", [
    ("오늘 날시 알려줘", "오늘 날씨 알려줘", (("날시", "날씨"),)),
    ("날시가 어때요", "날씨가 어때요", (("날시가", "날씨가"),)),  # the word with its particle, as the bot wrote it
    ("고마어", "고마워", (("고마어", "고마워"),)),
    ("부삼 날씨", "부산 날씨", (("부삼", "부산"),)),
    ("미새먼지 기준", "미세먼지 기준", (("미새먼지", "미세먼지"),)),
])
def test_one_jamo_slips_become_the_bots_word(kiwi_speller, typed, understood, changes) -> None:
    fixed = kiwi_speller.correct(typed)
    assert (fixed.text, fixed.changes) == (understood, changes)


@pytest.mark.parametrize("typed", ["뭐예요", "날씨에요", "먼지", "주식 추천해줘", "날씨 어때", "가", "Hello 날씨", "?!"])
def test_known_words_real_words_one_syllable_and_non_hangul_are_left_alone(kiwi_speller, typed) -> None:
    assert kiwi_speller.correct(typed).text == typed and kiwi_speller.correct(typed).changes == ()


def test_without_a_language_model_every_one_jamo_match_is_taken_and_ties_are_not() -> None:
    plain = Speller(["날씨", "날씨"])
    assert plain.correct("날시").text == "날씨" and len(plain) == 1
    tie = Speller(["가방", "가반"])  # "가밤" is one jamo from both, each seen once
    assert tie.correct("가밤").text == "가밤"
    assert Speller([]).correct("날시").text == "날시"


def test_a_margin_keeps_words_the_model_finds_about_as_likely() -> None:
    scores = {"날씨": -16.0, "날시": -26.0, "먼저": -15.0, "먼지": -18.0}
    sp = Speller(["날씨", "먼저"], scores.__getitem__, margin=5.0)
    assert sp.correct("날시").text == "날씨" and sp.correct("먼지").text == "먼지"
    assert Speller(["날씨"], scores.__getitem__, margin=20.0).correct("날시").text == "날시"


def test_vocabulary_has_examples_entities_and_the_knowledge_base(sample_bot) -> None:
    words = vocabulary(sample_bot)
    assert "안녕하세요" in words and "부산" in words  # an example and an entity value
    with_kb = vocabulary(sample_bot, KnowledgeIndex.build(KB, HashingEmbedder()))
    assert "구독 환불은 어떻게 하나요" in with_kb and any("30일 뒤에 지웁니다" in w for w in with_kb)


def test_the_interpreter_understands_the_corrected_sentence_and_logs_it(sample_bot) -> None:
    seen = []

    def parse(text):
        seen.append(text)
        return NluResult(text, (IntentScore("weather", 0.9),), 0.9, (), "classifier")

    interp = NluInterpreter(parse, None, Speller(["오늘 날씨 알려줘"]))
    info = interp.interpret("오늘 날시 알려줘", DialogueState(), sample_bot).info
    assert seen == ["오늘 날씨 알려줘"] and info["text"] == "오늘 날시 알려줘"
    assert info["corrected"] == {"text": "오늘 날씨 알려줘", "changes": [["날시", "날씨"]]}
    plain = NluInterpreter(parse, None, Speller(["오늘 날씨 알려줘"])).interpret("오늘 날씨 알려줘", DialogueState(), sample_bot).info
    assert "corrected" not in plain


def test_a_host_with_spelling_on_corrects_a_conversation_turn(fake_embedder, tmp_path, trained_sample) -> None:
    import shutil

    from chatcore.bot.repository import InMemoryRepository
    from chatcore.conversations import synthetic_actions
    from chatcore.host import BotHost
    from chatcore.store.memory import InMemoryStore

    from .conftest import SAMPLE

    shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)

    def host(spelling):
        h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions,
                    keep=1, min_f1=0.0, spelling=spelling)
        h.seed(SAMPLE)
        return h

    on = host(True)
    on.runtime("sample").processor.handle("s", "안녕하세여")
    nlu = [e for e in on.store.events("s") if e.type == "commands_received"][-1].nlu
    assert nlu["corrected"]["text"] == "안녕하세요" and nlu["text"] == "안녕하세여"
    off = host(False)
    off.runtime("sample").processor.handle("s", "안녕하세여")
    assert "corrected" not in [e for e in off.store.events("s") if e.type == "commands_received"][-1].nlu
