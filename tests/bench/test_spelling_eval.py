import pytest

from bench.augment_eval import Sentence
from bench.kb_eval import BenchError, Gold, Item, Question
from bench.spelling_eval import choose, criteria, kb_rows, load_natural, scores


def test_scores_count_in_scope_and_each_out_of_scope_kind() -> None:
    out = scores(["weather", "out_of_scope:hard", "out_of_scope:natural", "time"],
                 [("intent", "weather"), ("fallback", None), ("intent", "time"), ("intent", "weather")])
    assert out == {"in_scope": [True, False], "oos_hard": [True], "oos_easy": [], "oos_natural": [False]}


def test_kb_rows_tell_right_wrong_and_unanswerable() -> None:
    gold = Gold(frozenset({"faq_1"}), ())
    qs = [Question("a", "answerable", gold), Question("b", "answerable", gold), Question("c", "answerable", gold),
          Question("d", "unanswerable"), Question("e", "unanswerable")]
    rows = kb_rows(qs, [Item("faq_1", "faq", "t", "t"), Item("faq_2", "faq", "t", "t"), None, False, None])
    assert rows == {"correct": [True, False, False], "wrong": [False, True, False], "unans": [True, False]}


def test_the_natural_set_maps_to_the_original_gold(tmp_path) -> None:
    gold = Gold(frozenset({"faq_1"}), ())
    original = [Question("환불 되나요", "answerable", gold, half=1), Question("주식 추천", "unanswerable")]
    path = tmp_path / "typo.yaml"
    path.write_text("blocks: {weather: ['날시 어때']}\nout_of_scope: ['주싴 추천']\n"
                    "kb: [{q: '환불 돼나요', source: '환불 되나요'}]\nkb_unanswerable: [{q: '주식 추쳔', source: '주식 추천'}]\n",
                    encoding="utf-8")
    blocks, kb = load_natural(path, original)
    assert blocks == [Sentence("날시 어때", "weather"), Sentence("주싴 추천", "out_of_scope:natural")]
    assert kb == [Question("환불 돼나요", "answerable", gold, half=1), Question("주식 추쳔", "unanswerable")]
    path.write_text("blocks: {}\nout_of_scope: []\nkb: [{q: x, source: '없는 질문'}]\nkb_unanswerable: []\n", encoding="utf-8")
    with pytest.raises(BenchError):
        load_natural(path, original)


def test_margin_is_chosen_on_rule_typos_and_criteria_cover_the_spec() -> None:
    t, f = True, False

    def side(ins, hard, easy, nat):
        return {"in_scope": ins, "oos_hard": hard, "oos_easy": easy, "oos_natural": nat}

    blocks = {m: {"원문": side([t, t], [t], [t], []), "규칙 오타": side([t, m is not None and m > 3], [t], [t], []),
                  "자연 오타": side([f] * 10 if m is None else [t] * 10, [], [], [t])} for m in (None, 3.0, 5.0, 8.0)}
    kb = {("i", m): {s: {"correct": [t] * 5 + [f] * 5, "wrong": [f] * 10, "unans": [f]} for s in ("원문", "규칙 오타", "자연 오타")}
          for m in (None, 3.0, 5.0, 8.0)}
    for m in (3.0, 5.0, 8.0):
        kb[("i", m)]["자연 오타"]["correct"] = [t] * 10
    assert choose(blocks, kb, "i") == 8.0  # 5 and 8 tie on rule typos: the larger (fewer changes)
    fewer = {**blocks, 8.0: {**blocks[8.0], "규칙 오타": side([t, f], [t], [t], [])}}
    assert choose(fewer, kb, "i") == 5.0  # more correct answers beat the tie rule
    checks = dict(criteria(blocks, kb, "i", 5.0))
    assert len(checks) == 9 and all(checks.values())
    worse = {**kb[("i", 5.0)], "자연 오타": {**kb[("i", 5.0)]["자연 오타"], "unans": [t]}}
    assert not dict(criteria(blocks, {**kb, ("i", 5.0): worse}, "i", 5.0))["자연 오타 답 없는 질문에 답함 +0.02 이내"]
