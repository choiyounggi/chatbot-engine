import pytest

from bench.augment_eval import (
    KINDS,
    Sentence,
    _edits,
    augmented_faq,
    augmented_nlu,
    block_scores,
    kb_correct,
    load_block_test,
    noise,
    paired,
    proposals,
)
from bench.kb_eval import BenchError, Gold, Item, Question


@pytest.mark.parametrize("text, kind, expected", [
    ("새해", "vowel", {"세해", "새헤"}),
    ("있어", "tense", {"잇어"}),
    ("까치", "tense", {"가치"}),
    ("밥", "final", {"바"}),
    ("오늘 날씨", "space", {"오 늘 날씨", "오늘 날 씨", "오늘날씨"}),
    ("알려줘요", "yo", {"알려줘여"}),
])
def test_each_kind_of_typo(text, kind, expected) -> None:
    assert set(_edits(text).get(kind, [])) == expected


def test_noise_makes_one_seeded_typo_and_leaves_text_without_hangul_alone() -> None:
    noisy, kind = noise("오늘 날씨 알려줘요")
    assert kind in KINDS and noisy != "오늘 날씨 알려줘요"
    assert noise("오늘 날씨 알려줘요") == (noisy, kind)  # the same every run
    assert noise("ok 123") == ("ok 123", "none") and noise("") == ("", "none")


def test_augmenting_documents_appends_without_changing_the_original() -> None:
    nlu = {"intents": {"weather": {"examples": ["날씨"]}, "time": {"examples": ["몇 시"]}}}
    out = augmented_nlu(nlu, {"weather": ["날시"]})
    assert out["intents"]["weather"]["examples"] == ["날씨", "날시"] and nlu["intents"]["weather"]["examples"] == ["날씨"]
    faq = [{"id": "a", "question": "q", "answer": "x"}, {"id": "b", "question": "r", "answer": "y", "similar": ["r?"]}]
    assert augmented_faq(faq, {"b": ["r!"]}) == [{**faq[0], "similar": []}, {**faq[1], "similar": ["r?", "r!"]}]


def test_scoring() -> None:
    gold = Gold(frozenset({"faq_1"}), (("a.md", "증거"),))
    q = Question("q", "answerable", gold=gold)
    assert kb_correct(q, Item("faq_1", "faq", "t", "t")) and not kb_correct(q, Item("faq_2", "faq", "t", "t"))
    assert kb_correct(q, Item("a#0", "doc", "이 조각에 증거가 있다", "", "a.md")) and not kb_correct(q, None)
    assert not kb_correct(Question("q", "unanswerable"), Item("faq_1", "faq", "t", "t"))
    scores = block_scores(["weather", "out_of_scope:hard", "out_of_scope:easy", "time"],
                          [("intent", "weather"), ("fallback", None), ("intent", "time"), ("clarify", "time")])
    assert scores == {"in_scope": [True, False], "oos_hard": [True], "oos_easy": [False]}
    assert paired([False, False, True], [True, False, True])[0] == pytest.approx(1 / 3)
    with pytest.raises(BenchError):
        paired([], [])


def test_the_block_test_set_and_the_proposal_cache() -> None:
    sentences = load_block_test()
    assert Sentence("안녕하세요", "greeting") in sentences and any(s.gold == "out_of_scope:hard" for s in sentences)
    cache: dict = {}
    calls = []
    messages = [{"role": "user", "content": "x"}]
    ask = lambda m: calls.append(1) or [("날시", "typo")]  # noqa: E731
    assert proposals("m", [messages], cache, ask) == [("날시", "typo")] == proposals("m", [messages], cache, ask)
    assert calls == [1] and proposals("other-model", [messages], cache, ask) and calls == [1, 1]
    from chatcore.generation import GenerationError

    def broken(m):
        raise GenerationError("error", "no sentences")

    failed: list[str] = []
    assert proposals("third", [messages], cache, broken, failed) == [] and failed == ["error: no sentences"]
    assert "third" not in str(cache.keys()) and len(cache) == 2  # not cached: asked again next run


def test_criteria_gate_every_set_the_spec_names_and_kinds_split_the_typo_scores() -> None:
    from bench.augment_eval import by_kind, criteria

    t, f = True, False
    side = lambda ins, hard, easy: {"in_scope": ins, "oos_hard": hard, "oos_easy": easy}  # noqa: E731
    base = {"clean": side([t, f], [t, t], [t, t]), "noisy": side([f, f], [t, t], [t, t])}
    kbb = {"clean": {"correct": [t, f], "unans": [f, f]}, "noisy": {"correct": [f, f], "unans": [f, f]}}
    better = {"clean": side([t, f], [t, t], [t, t]), "noisy": side([t, t], [t, t], [t, t])}
    kbg = {"clean": {"correct": [t, f], "unans": [f, f]}, "noisy": {"correct": [t, t], "unans": [f, f]}}
    names = [n for n, _ in criteria(base, kbb, better, kbg)]
    assert len(names) == 10 and "오타판 범위 밖(easy) 거절률 −0.02 이내" in names and "오타판 답 없는 질문에 답함 +0.02 이내" in names
    assert all(ok for _, ok in criteria(base, kbb, better, kbg))
    worse = {**better, "noisy": side([t, t], [t, t], [f, f])}  # easy out-of-scope rejection fell on the typo variant
    assert dict(criteria(base, kbb, worse, kbg))["오타판 범위 밖(easy) 거절률 −0.02 이내"] is False
    assert by_kind([t, f, t], ["yo", "space", "yo"]) == {"yo": [t, t], "space": [f]}
