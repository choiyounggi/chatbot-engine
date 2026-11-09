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
