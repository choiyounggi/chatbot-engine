import datetime as dt

import pytest

from chatcore.bot.schema import BotDefinition
from chatcore.nlu.entities import EntityExtractor, EntityMatch, bare, normalize, pick

TODAY = dt.date(2026, 11, 10)


def values(matches, entity):
    return [m.value for m in matches if m.entity == entity]


@pytest.mark.parametrize(
    "text, expected",
    [("부산광역시 날씨 어때?", ["부산"]), ("부산은?", ["부산"]), ("서울역 가는 길", []), ("제주도 기온", ["제주"]), ("", [])],
)
def test_dictionary_values_on_token_boundaries(extractor, text, expected):
    assert values(extractor.extract(text, TODAY), "location") == expected


def test_all_candidates_are_returned_and_pick_resolves_overlaps(extractor):
    found = extractor.extract("내일 오후 3시 부산", TODAY)
    assert {"sys.date", "sys.time", "sys.number", "location"} <= {m.entity for m in found}
    chosen = pick(found, {"sys.time", "sys.number"})
    assert [m.value for m in chosen] == ["15:00"]  # the time span contains the number; longer wins
    assert pick(found, set()) == []


def test_regex_entities():
    req = {k: {"text": k} for k in ("fallback", "cancelled", "restarted", "error", "clarify", "input_invalid")}
    bot = BotDefinition.model_validate({
        "id": "t", "version": "v", "entities": {"order": {"kind": "regex", "pattern": r"[A-Z]{2}\d{4}"}},
        "scenarios": {"s": {"steps": [{"respond": "fallback"}]}}, "responses": req,
    })
    assert values(EntityExtractor(bot).extract("주문 AB1234 확인", TODAY), "order") == ["AB1234"]


def test_normalize_and_bare():
    assert normalize(" 날씨　 알려줘 ") == "날씨 알려줘"
    assert bare("  취소!!  ") == "취소" and bare("...") == ""


def test_pick_prefers_longer_then_earlier():
    a = EntityMatch("x", "a", "ab", 0, 2)
    b = EntityMatch("x", "b", "abc", 1, 4)
    c = EntityMatch("x", "c", "d", 5, 6)
    assert pick([a, b, c], {"x"}) == [b, c]


def test_shorter_form_is_tried_when_longer_one_fails_the_token_boundary(sample_bot):
    req = {k: {"text": k} for k in ("fallback", "cancelled", "restarted", "error", "clarify", "input_invalid")}
    bot = BotDefinition.model_validate({
        "id": "t", "version": "v", "entities": {"city": {"kind": "dictionary", "values": {"서울": ["서울시"]}}},
        "scenarios": {"s": {"steps": [{"respond": "fallback"}]}}, "responses": req,
    })
    ex = EntityExtractor(bot)

    class Tok:
        def __init__(self, start, length):
            self.start, self.len = start, length

    class SplitKiwi:  # "서울" + "시립대": "서울시" crosses a boundary, "서울" does not
        def tokenize(self, text):
            return [Tok(0, 2), Tok(2, 3)]

    ex.kiwi = SplitKiwi()
    assert [(m.value, m.text) for m in ex.extract("서울시립대", TODAY) if m.entity == "city"] == [("서울", "서울")]


def test_the_pooled_test_kiwi_extracts_exactly_what_a_real_kiwi_does(sample_bot):
    from kiwipiepy import Kiwi

    from tests.chatcore.conftest import PooledKiwi

    real, pooled = EntityExtractor(sample_bot, kiwi=Kiwi()), EntityExtractor(sample_bot)
    assert isinstance(pooled.kiwi, PooledKiwi)  # the suite-wide stand-in is active
    # 광명날씨 / 군포날씨 are found only with the bot's user words: they prove the words reach the real Kiwi
    for text in ("광명날씨", "군포날씨 알려줘", "서울역 날씨", "서울시립대 앞", "부산은?", "광주 내일 3시 날씨", ""):
        assert pooled.extract(text, TODAY) == real.extract(text, TODAY), text
    assert [m.value for m in pooled.extract("광명날씨", TODAY) if m.entity == "location"] == ["광명"]
    with pytest.raises(AssertionError, match="before the first tokenize"):
        pooled.kiwi.add_user_word("늦은단어", "NNP")
