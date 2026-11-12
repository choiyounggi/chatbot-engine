import pytest
from kiwipiepy import Kiwi

from bench.kb_eval import BenchError, Question
from bench.rag_eval import (
    NEGATION,
    Case,
    cache_key,
    choose,
    evaluate,
    fact_score,
    fact_tokens,
    final_text,
    judged,
    migrate,
    paired,
    passes,
)
from chatcore.knowledge import Entry

KIWI = Kiwi()
TOK = KIWI.tokenize


def test_fact_tokens_are_nouns_and_numbers() -> None:
    toks = fact_tokens("연 구독은 결제일로부터 14일 이내에 39,000원을 환불해요.", TOK)
    assert {"구독", "결제", "환불", "14", "39000"} <= toks
    assert "해요" not in toks and fact_tokens("", TOK) == set()


def test_fact_score_takes_the_best_gold_and_is_zero_without_facts() -> None:
    gold = ("결제일로부터 14일 이내 전액 환불", "위젯은 홈 화면에서 추가")
    assert fact_score("결제일로부터 14일 이내라 전액 환불돼요", gold, TOK) == pytest.approx(1.0)
    assert fact_score("14일 안에 결제했으면 전액 환불돼요", gold, TOK) == pytest.approx(4 / 6)  # a paraphrase loses 이내·일
    assert fact_score("홈 화면을 길게 누르세요", gold, TOK) == pytest.approx(0.5)  # 홈·화면 of 홈·화면·위젯·추가
    assert fact_score("아무 관련 없는 말", gold, TOK) == 0.0
    assert fact_score("무엇이든", (), TOK) == 0.0 and fact_score("무엇이든", ("...",), TOK) == 0.0


def case(extract):
    return Case(Question("q", "answerable"), "fallback", (), extract, ("g",), 10.0)


def test_final_text_per_configuration() -> None:
    gen_ok, gen_bad = {"text": "생성 답", "outcome": "generated"}, {"text": None, "outcome": "no_answer"}
    failed_call = {"text": None, "outcome": "timeout"}
    assert final_text(case(None), gen_ok, "extract") is None  # the knowledge base did not answer
    assert final_text(case("추출 답"), None, "-") == "추출 답"  # extractive
    assert final_text(case("추출 답"), gen_ok, "fallback") == "생성 답"
    assert final_text(case("추출 답"), gen_bad, "extract") == "추출 답"
    assert final_text(case("추출 답"), gen_bad, "fallback") is None
    assert final_text(case("추출 답"), failed_call, "fallback") == "추출 답"  # the LLM did not answer: as the engine


PASSAGE = Entry("refund", "faq", "환불 되나요?", "결제 후 7일 안에 환불돼요.")


def test_a_cached_generation_is_judged_again_with_the_engine_timeout() -> None:
    assert judged({"raw": "7일 안에 돼요.", "ms": 900}, [PASSAGE], 3500) == {"text": "7일 안에 돼요.", "outcome": "generated", "ms": 900}
    assert judged({"raw": "14일 안에 돼요.", "ms": 900}, [PASSAGE], 3500)["outcome"] == "unsupported_number"
    assert judged({"raw": "7일 안에 돼요.", "ms": 3501}, [PASSAGE], 3500) == {"text": None, "outcome": "timeout", "ms": 3500}


def test_the_cache_key_is_what_was_sent_and_old_entries_move_to_it() -> None:
    c = Case(Question("환불 돼요?", "answerable"), "fallback", (PASSAGE,), "추출", (), 1.0)
    other = Case(Question("환불 돼요?", "answerable"), "fallback", (Entry("x", "faq", "다른", "자료"),), "추출", (), 1.0)
    assert cache_key("m", 1, c) == cache_key("m", 1, c) != cache_key("m", 1, other) != cache_key("m2", 1, other)
    old = {"m|1|환불 돼요?": {"text": "t", "outcome": "generated", "ms": 5, "raw": "r"},
           "m|3|환불 돼요?": {"text": None, "outcome": "timeout", "ms": 60000, "raw": ""},  # a failure: tried again
           "m|1|모르는 질문": {"text": "t", "outcome": "generated", "ms": 5, "raw": "r"}}
    assert migrate(old, [c]) == {cache_key("m", 1, c): {"raw": "r", "ms": 5}}
    assert migrate({}, [c]) == {}


def test_choose_takes_the_first_passing_else_the_closest() -> None:
    base = {"correct": (0.6, (0, 0)), "wrong": (0.06, (0, 0)), "unans_answered": (0.06, (0, 0)), "p95_ms": 100.0, "median_chars": 50}
    near = {**base, "wrong": (0.07, (0, 0))}  # misses one criterion
    far = {**base, "wrong": (0.07, (0, 0)), "median_chars": 300}
    good = {**base, "correct": (0.59, (0, 0))}
    cfgs = [(None, 1, "-"), ("a", 1, "extract"), ("b", 1, "extract"), ("c", 1, "extract")]
    assert choose(cfgs, {cfgs[0]: base, cfgs[1]: far, cfgs[2]: good, cfgs[3]: good}) == (cfgs[2], True)
    assert choose(cfgs, {cfgs[0]: base, cfgs[1]: far, cfgs[2]: near, cfgs[3]: far}) == (cfgs[2], False)


def test_evaluate_and_paired_count_one_half_and_skip_exact_matches() -> None:
    def q(text, kind, half):
        return Question(text, kind, half=half)

    src = ("결제 후 7일 안에 환불돼요.",)
    cases = [Case(q("환불 돼요?", "answerable", 0), "fallback", (PASSAGE,), "결제 후 7일 안에 환불돼요.", (), 10.0, src),
             Case(q("환불 되나요?", "answerable", 0), "fallback", (PASSAGE,), "결제 후 7일 안에 환불돼요.", (), 10.0, src, exact=True),
             Case(q("주식 추천", "unanswerable", 0), "fallback", (PASSAGE,), "결제 후 7일 안에 환불돼요.", (), 10.0),
             Case(q("다른 절반", "answerable", 1), "fallback", (), None, (), 10.0, src)]
    cache = {cache_key("m", 1, cases[0]): {"raw": "7일 안에 환불돼요.", "ms": 500},
             cache_key("m", 1, cases[2]): {"raw": "NO_ANSWER", "ms": 300}}  # none for the exact match: never asked
    r = evaluate(cases, cache, TOK, "m", 1, "fallback", half=0, timeout_ms=3500)
    assert r["correct"][0] == 1.0 and r["unans_answered"][0] == 0.0 and r["p95_ms"] == pytest.approx(495, abs=10)
    base = evaluate(cases, cache, TOK, None, 1, "-", half=0, timeout_ms=3500)
    assert base["unans_answered"][0] == 1.0
    p = paired(cases, cache, TOK, "m", 1, "fallback", half=0, timeout_ms=3500)
    assert p["correct"][0] == 0.0 and p["unans"] == (0.0, 1.0)
    with pytest.raises(BenchError):  # a generation that should be cached and is not
        evaluate(cases, {}, TOK, "m", 1, "fallback", half=0, timeout_ms=3500)


def test_passes_needs_every_criterion() -> None:
    base = {"correct": (0.6, (0, 0)), "wrong": (0.06, (0, 0)), "unans_answered": (0.06, (0, 0))}
    good = {"correct": (0.59, (0, 0)), "wrong": (0.05, (0, 0)), "unans_answered": (0.06, (0, 0)),
            "p95_ms": 5000.0, "median_chars": 200}
    assert passes(good, base)
    for key, bad in (("correct", (0.57, (0, 0))), ("wrong", (0.07, (0, 0))), ("unans_answered", (0.07, (0, 0))),
                     ("p95_ms", 5001.0), ("median_chars", 201)):
        assert not passes({**good, key: bad}, base), key


def test_backed_needs_facts_from_one_right_source_and_all_its_numbers() -> None:
    from bench.rag_eval import backed

    src = ("PM10 24시간 평균 기준으로 좋음 0~30, 보통 31~80, 나쁨 81~150, 매우나쁨 151 이상으로 표시합니다.",)
    assert backed("나쁨입니다.", src, TOK)  # short and right
    assert backed("81이면 나쁨이에요.", src, TOK)
    assert not backed("70이면 나쁨이에요.", src, TOK)  # 70 is not in the source
    assert not backed("위젯은 홈 화면에서 추가해요.", src, TOK)  # about something else
    assert not backed("네, 그렇습니다.", src, TOK)  # states nothing checkable
    assert not backed("나쁨입니다.", (), TOK)  # no right source


def test_a_negation_is_a_fact_and_one_without_a_negating_source_is_flagged() -> None:
    from bench.rag_eval import backed, unbacked_negation

    assert NEGATION in fact_tokens("환불되지 않아요", TOK) and NEGATION not in fact_tokens("환불돼요", TOK)
    assert not backed("환불되지 않아요.", ("환불돼요.",), TOK)  # a short flip fails the share
    flipped, source = "연간 구독은 환불되지 않아요.", ("연간 구독은 결제 후 7일 안에 환불돼요.",)
    assert backed(flipped, source, TOK) and unbacked_negation(flipped, source, TOK)  # a longer one: flagged
    assert not unbacked_negation(flipped, ("연간 구독은 7일이 지나면 환불되지 않습니다.",), TOK)
    assert not unbacked_negation("연간 구독은 환불돼요.", source, TOK)


def test_disagreements_list_what_a_person_should_read() -> None:
    from bench.rag_eval import disagreements

    src = ("결제 후 7일 안에 환불돼요.",)
    cases = [Case(Question("환불 돼요?", "answerable"), "fallback", (PASSAGE,), "결제 후 7일 안에 환불돼요.", (), 1.0, src),
             Case(Question("환불 되나요?", "answerable"), "fallback", (PASSAGE,), "결제 후 7일 안에 환불돼요.", (), 1.0, src),
             Case(Question("주식", "unanswerable"), "fallback", (PASSAGE,), "결제 후 7일 안에 환불돼요.", (), 1.0)]
    cache = {cache_key("m", 1, cases[0]): {"raw": "결제 후 7일 안에 환불돼요.", "ms": 1},  # same score: not listed
             cache_key("m", 1, cases[1]): {"raw": "결제 후 7일 안에는 환불이 안 돼요.", "ms": 1},  # negates
             cache_key("m", 1, cases[2]): {"raw": "NO_ANSWER", "ms": 1}}  # better than extractive
    rows = disagreements(cases, cache, TOK, "m", 1, "fallback", 3500)
    assert [(r["question"], r["mine"], r["theirs"], r["negates"]) for r in rows] == [
        ("환불 되나요?", True, True, True), ("주식", True, False, False)]
    assert rows[1]["text"] is None and rows[1]["outcome"] == "no_answer"
