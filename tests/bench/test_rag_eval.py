import pytest
from kiwipiepy import Kiwi

from bench.kb_eval import Question
from bench.rag_eval import Case, fact_score, fact_tokens, final_text, passes

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
    gen_ok, gen_bad = {"text": "생성 답"}, {"text": None}
    assert final_text(case(None), gen_ok, "extract") is None  # the knowledge base did not answer
    assert final_text(case("추출 답"), None, "-") == "추출 답"  # extractive
    assert final_text(case("추출 답"), gen_ok, "fallback") == "생성 답"
    assert final_text(case("추출 답"), gen_bad, "extract") == "추출 답"
    assert final_text(case("추출 답"), gen_bad, "fallback") is None


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
