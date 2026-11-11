import numpy as np
import pytest

from bench.kb_eval import (
    BM25,
    CHUNK_MAX,
    CRITERIA,
    OVERRIDE_MARGIN,
    BenchError,
    Gold,
    Item,
    Nlu,
    PoolResult,
    Question,
    bootstrap,
    choose_thresholds,
    chunk_document,
    cosine_candidates,
    final,
    judged_answer,
    kb_answer,
    load_kb,
    load_questions,
    outcomes,
    rates,
    rates_ci,
    recall_mrr,
    rrf,
    split_halves,
)

FAQ = [Item("faq_001", "faq", "알림이 안 와요\n설정에서 켜세요", "알림이 안 와요"),
       Item("faq_002", "faq", "환불 되나요\n7일 안에 됩니다", "환불 되나요")]
DOC = [Item("refund.md#0", "doc", "환불 - 기간\n결제 후 7일 이내에는 전액 환불합니다.", "", "refund.md"),
       Item("manual.md#0", "doc", "설명서 - 위젯\n위젯은 홈 화면에서 추가합니다.", "", "manual.md")]
POOLS = {"faq": FAQ, "doc": DOC}


def decide(nlu: Nlu, faq: PoolResult, doc: PoolResult, t_faq: float, t_doc: float) -> tuple[str, tuple[str, int] | None]:
    return final(nlu, kb_answer(faq, doc, t_faq, t_doc), faq)


def judge_outcomes(qs, nlu, faq, doc, t_faq, t_doc):  # type: ignore[no-untyped-def]
    return outcomes(qs, nlu, faq, [kb_answer(f, d, t_faq, t_doc) for f, d in zip(faq, doc, strict=True)], POOLS)


def pool(cosines: list[float]) -> PoolResult:
    c = np.array(cosines)
    return PoolResult([int(i) for i in np.argsort(-c, kind="stable")], c)


# ---------------------------------------------------------------- chunking


def test_chunks_keep_sentences_whole_and_stay_in_their_section() -> None:
    sentence = "맑음노트는 매시 정각에 예보를 갱신합니다."  # 22 chars
    body = " ".join([sentence] * 40)  # ~920 chars: needs at least two chunks
    text = f"# 설명서\n\n## 갱신\n\n{body}\n\n## 위젯\n\n위젯은 홈 화면에서 추가합니다."
    chunks = chunk_document("manual.md", text)
    assert [c.id for c in chunks][-1] == f"manual.md#{len(chunks) - 1}"
    widget = [c for c in chunks if "위젯은" in c.text]
    assert len(widget) == 1 and widget[0].text.startswith("설명서 - 위젯\n") and "갱신합니다" not in widget[0].text
    for c in chunks:
        body_part = c.text.split("\n", 1)[1]
        assert len(body_part) <= CHUNK_MAX
        if c not in widget:
            assert body_part.replace(sentence, "").strip() == ""  # only whole sentences


def test_consecutive_chunks_repeat_a_short_last_sentence() -> None:
    sents = [f"문장 {i}번은 겹침을 확인하려고 쓴 꽤 긴 예시 문장으로 약 마흔 글자 정도입니다." for i in range(30)]
    chunks = chunk_document("a.md", "# 제목\n\n## 절\n\n" + " ".join(sents))
    assert len(chunks) >= 2
    first, second = chunks[0].text.split("\n", 1)[1], chunks[1].text.split("\n", 1)[1]
    last_of_first = first.split(" 문장 ")[-1]
    assert second.startswith("문장 " + last_of_first) or second.startswith(last_of_first)


def test_a_sentence_longer_than_a_chunk_is_never_cut() -> None:
    long = "가" * (CHUNK_MAX + 100) + "."
    chunks = chunk_document("a.md", f"# 제목\n\n## 절\n\n짧은 문장입니다. {long} 끝 문장입니다.")
    assert any(long in c.text for c in chunks)


def test_empty_document_has_no_chunks() -> None:
    assert chunk_document("a.md", "") == []
    assert chunk_document("a.md", "# 제목만\n\n## 절만") == []


def test_load_kb_refuses_missing_faq_and_duplicates(tmp_path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "faq.yaml").write_text("faq: []\n", encoding="utf-8")
    with pytest.raises(BenchError, match="no faq"):
        load_kb(tmp_path)
    (tmp_path / "faq.yaml").write_text("faq:\n- {id: f1, question: q, answer: a}\n- {id: f1, question: r, answer: b}\n",
                                       encoding="utf-8")
    (tmp_path / "docs/a.md").write_text("# t\n\n## s\n\n내용입니다.", encoding="utf-8")
    with pytest.raises(BenchError, match="duplicate"):
        load_kb(tmp_path)


# ---------------------------------------------------------------- questions


def test_load_questions_checks_gold_and_splits_each_kind_in_halves(tmp_path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(
        "answerable:\n"
        "- {q: 알림 설정, gold: [faq_001]}\n"
        "- {q: 환불 기간, gold: [{doc: refund.md, evidence: 7일 이내}]}\n"
        "unanswerable: [주식 추천, 맛집 추천]\n"
        "blocks:\n- {q: 날씨 알려줘, intent: weather}\n- {q: 몇 시야, intent: time}\n",
        encoding="utf-8")
    qs = load_questions(path, FAQ + DOC)
    assert [q.kind for q in qs] == ["answerable"] * 2 + ["unanswerable"] * 2 + ["block"] * 2
    assert sorted(q.half for q in qs if q.kind == "unanswerable") == [0, 1]
    refund = next(q for q in qs if q.text == "환불 기간")
    assert refund.gold is not None and refund.gold.hit(DOC[0]) and not refund.gold.hit(DOC[1])


def test_load_questions_refuses_unknown_or_unmatched_gold(tmp_path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text("answerable:\n- {q: x, gold: [faq_999]}\n", encoding="utf-8")
    with pytest.raises(BenchError, match="unknown FAQ id"):
        load_questions(path, FAQ + DOC)
    path.write_text("answerable:\n- {q: x, gold: [{doc: refund.md, evidence: 없는 문구}]}\n", encoding="utf-8")
    with pytest.raises(BenchError, match="no knowledge-base item"):
        load_questions(path, FAQ + DOC)
    path.write_text("answerable: []\n", encoding="utf-8")
    with pytest.raises(BenchError, match="no questions"):
        load_questions(path, FAQ + DOC)


def test_split_halves_is_deterministic() -> None:
    qs = [Question(f"q{i}", "unanswerable") for i in range(9)]
    a, b = split_halves(qs), split_halves(qs)
    assert [(q.text, q.half) for q in a] == [(q.text, q.half) for q in b]
    assert sum(q.half for q in a) == 4


# ---------------------------------------------------------------- ranking


def test_bm25_prefers_rare_matching_terms() -> None:
    bm = BM25([["환불", "기간"], ["알림", "설정"], ["알림", "환불", "위젯"]])
    s = bm.scores(["기간"])
    assert s[0] > 0 and s[1] == 0 and s[2] == 0
    s = bm.scores(["알림", "설정"])
    assert int(np.argmax(s)) == 1


def test_bm25_empty_inputs_score_zero() -> None:
    assert BM25([]).scores(["x"]).shape == (0,)
    assert not BM25([["a"]]).scores([]).any()
    assert not BM25([[]]).scores(["a"]).any()


def test_rrf_rewards_agreement_and_breaks_ties_by_the_first_ranking() -> None:
    assert rrf([0, 1, 2], [2, 1, 0]) == [0, 2, 1]  # 1/61+1/63 > 2/62; 0 and 2 tie, first ranking decides
    assert rrf([0, 1], [1, 0]) == [0, 1]  # equal scores: first ranking's order
    assert rrf([3, 1], [1, 4])[0] == 1
    assert rrf([], []) == []


# ---------------------------------------------------------------- decisions


def test_kb_answer_prefers_a_passing_faq_then_a_passing_chunk() -> None:
    assert kb_answer(pool([0.9, 0.1]), pool([0.8, 0.1]), 0.8, 0.5) == ("faq", 0)
    assert kb_answer(pool([0.7, 0.1]), pool([0.1, 0.6]), 0.8, 0.5) == ("doc", 1)
    assert kb_answer(pool([0.7, 0.1]), pool([0.4, 0.1]), 0.8, 0.5) is None
    assert kb_answer(pool([0.8]), pool([]), 0.8, 0.5) == ("faq", 0)  # exactly at the threshold answers
    assert kb_answer(PoolResult([], np.zeros(0)), PoolResult([], np.zeros(0)), 0.0, 0.0) is None


def test_final_follows_the_dialogue_rules() -> None:
    sure = Nlu("intent", "check_weather", 0.80)
    assert decide(sure, pool([0.80 + OVERRIDE_MARGIN, 0]), pool([0.9]), 0.7, 0.5) == ("kb", ("faq", 0))
    assert decide(sure, pool([0.84, 0]), pool([0.9]), 0.7, 0.5) == ("block", None)  # a chunk never overrides a block
    assert decide(sure, pool([0.95, 0]), pool([0.9]), 0.96, 0.5) == ("block", None)  # FAQ must pass its own threshold
    for rule in ("clarify", "fallback"):
        assert decide(Nlu(rule, None, 0.3), pool([0.1]), pool([0.6]), 0.7, 0.5) == ("kb", ("doc", 0))
        assert decide(Nlu(rule, None, 0.3), pool([0.1]), pool([0.4]), 0.7, 0.5) == (rule, None)
    assert decide(Nlu("rerun", None, 0.3), pool([0.99]), pool([0.99]), 0.5, 0.5) == ("block", None)


def test_outcomes_and_rates_per_kind_and_half() -> None:
    qs = [Question("알림", "answerable", Gold(frozenset({"faq_001"}), ()), half=0),
          Question("환불", "answerable", Gold(frozenset({"faq_002"}), ()), half=1),
          Question("주식", "unanswerable", half=0), Question("코인", "unanswerable", half=1),
          Question("날씨", "block", intent="weather", half=0), Question("몇 시", "block", intent="time", half=1)]
    fb = Nlu("fallback", None, 0.2)
    nlu = [fb, fb, fb, fb, Nlu("intent", "check_weather", 0.5), Nlu("intent", "tell_time", 0.9)]
    faq = [pool([0.9, 0.1]), pool([0.9, 0.1]), pool([0.2, 0.1]), pool([0.9, 0.1]), pool([0.9, 0.1]), pool([0.9, 0.1])]
    doc = [pool([0.1, 0.1])] * 6
    o = judge_outcomes(qs, nlu, faq, doc, 0.8, 0.5)
    assert o.correct.tolist() == [True, False, False, False, False, False]
    assert o.wrong.tolist() == [False, True, False, False, False, False]  # "환불" answered with faq_001
    assert o.fallback.tolist() == [False, False, True, False, False, False]
    assert o.kb_takes_block.tolist() == [False, False, False, False, True, False]  # 0.9 >= 0.5 + margin
    dev = rates(o, qs, 0)
    assert dev == {"correct": 1.0, "wrong": 0.0, "block_takes_kb": 0.0, "fallback": 1.0, "kb_takes_block": 1.0}
    assert rates(o, qs, None)["fallback"] == 0.5
    test = rates_ci(o, qs, 1)
    assert test["wrong"][0] == 1.0 and test["wrong"][1] == (1.0, 1.0)


def test_rates_refuse_a_half_without_questions() -> None:
    qs = [Question("주식", "unanswerable", half=0)]
    o = judge_outcomes(qs, [Nlu("fallback", None, 0.1)], [pool([0.1])], [pool([0.1])], 0.5, 0.5)
    with pytest.raises(BenchError, match="no answerable"):
        rates(o, qs, 0)


def test_choose_thresholds_meets_the_constraints_then_maximizes_correct_answers() -> None:
    def evaluate(t: tuple[float, ...]) -> dict[str, float]:
        t_faq, t_doc = t
        return {"correct": 1 - t_faq / 2 - t_doc / 4, "wrong": 0.0,
                "fallback": 0.85 if t_faq < 0.6 else 0.95, "kb_takes_block": 0.10 if t_doc < 0.5 else 0.0,
                "block_takes_kb": 0.0}

    chosen = choose_thresholds(evaluate, cosine_candidates((0.4, 0.5, 0.6, 0.7)))
    assert chosen is not None
    (t_faq, t_doc), r = chosen
    assert (t_faq, t_doc) == (0.6, 0.5)
    assert r["fallback"] >= CRITERIA["fallback"] and r["kb_takes_block"] <= CRITERIA["kb_takes_block"]
    assert choose_thresholds(lambda t: {"correct": 1.0, "wrong": 0.0, "fallback": 0.0, "kb_takes_block": 1.0},
                             cosine_candidates((0.5,))) is None


def test_recall_mrr_uses_the_best_rank_in_either_pool() -> None:
    gold = Gold(frozenset({"faq_002"}), (("refund.md", "7일 이내"),))
    assert recall_mrr(gold, FAQ, DOC, pool([0.9, 0.8]), pool([0.9, 0.1])) == (True, 1.0)  # doc rank 1
    assert recall_mrr(gold, FAQ, DOC, pool([0.9, 0.8]), pool([0.1, 0.9])) == (True, 0.5)
    missing = Gold(frozenset({"faq_404"}), ())
    assert recall_mrr(missing, FAQ, DOC, pool([0.9, 0.8]), pool([0.1, 0.9])) == (False, 0.0)


def test_bootstrap_brackets_the_mean_and_refuses_empty() -> None:
    lo, hi = bootstrap(np.array([0.0, 1.0] * 50))
    assert lo < 0.5 < hi
    assert bootstrap(np.ones(5)) == (1.0, 1.0)
    with pytest.raises(BenchError):
        bootstrap(np.array([]))


def test_a_chunk_exactly_at_the_document_threshold_answers() -> None:
    assert kb_answer(pool([0.1]), pool([0.5]), 0.8, 0.5) == ("doc", 0)


def test_choose_thresholds_breaks_ties_with_the_highest_thresholds() -> None:
    flat = {"correct": 0.5, "wrong": 0.0, "fallback": 1.0, "kb_takes_block": 0.0, "block_takes_kb": 0.0}
    chosen = choose_thresholds(lambda t: flat, cosine_candidates((0.5, 0.6, 0.7)))
    assert chosen is not None and chosen[0] == (0.7, 0.7)
    single = choose_thresholds(lambda t: flat, [(0.1,), (0.3,), (0.2,)])
    assert single is not None and single[0] == (0.3,)


def test_an_unanswerable_question_that_ends_in_clarify_counts_as_not_answered() -> None:
    qs = [Question("둘 중 뭐지", "unanswerable")]
    o = judge_outcomes(qs, [Nlu("clarify", None, 0.6)], [pool([0.1])], [pool([0.1])], 0.8, 0.5)
    assert o.fallback.tolist() == [True]


def test_a_short_first_sentence_is_not_left_as_a_tiny_chunk() -> None:
    long1, long2 = "나" * 495 + ".", "다" * 400 + "."  # short + long1 is over CHUNK_MAX
    chunks = chunk_document("a.md", f"# 제목\n\n## 절\n\n짧은 문장입니다. {long1} {long2}")
    assert "짧은 문장입니다." in chunks[0].text and long1 in chunks[0].text


def test_bm25_weighs_a_term_every_document_has_below_a_rare_one() -> None:
    bm = BM25([["흔함", "드묾"], ["흔함", "다른"], ["흔함", "또"]])
    assert bm.scores(["드묾"])[0] > bm.scores(["흔함"])[0] > 0


def test_judged_answer_takes_the_best_scored_candidate_at_or_above_the_threshold() -> None:
    cands = [("faq", 1), ("faq", 0), ("doc", 0)]
    assert judged_answer(cands, np.array([0.2, 0.7, 0.4]), 0.5) == ("faq", 0)
    assert judged_answer(cands, np.array([0.2, 0.5, 0.4]), 0.5) == ("faq", 0)  # exactly at the threshold
    assert judged_answer(cands, np.array([0.2, 0.49, 0.4]), 0.5) is None
    assert judged_answer([], np.zeros(0), 0.0) is None


def test_a_judged_chunk_never_overrides_a_sure_block_but_a_judged_faq_can() -> None:
    sure = Nlu("intent", "check_weather", 0.70)
    faq = pool([0.80, 0.1])
    assert final(sure, ("doc", 0), faq) == ("block", None)
    assert final(sure, ("faq", 0), faq) == ("kb", ("faq", 0))  # 0.80 >= 0.70 + margin
    assert final(sure, ("faq", 1), faq) == ("block", None)  # its own cosine decides the override
    assert final(Nlu("fallback", None, 0.2), None, faq) == ("fallback", None)
