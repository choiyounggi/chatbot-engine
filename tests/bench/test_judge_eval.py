import pytest

from bench.judge_eval import NOW, choose, criteria, memo, report, rows_for
from bench.kb_eval import BenchError, Gold, Item, Question

GOLD = Gold(frozenset({"f1"}), ())
RIGHT, OTHER = Item("f1", "faq", "환불", "환불"), Item("f2", "faq", "해지", "해지")
T, F = True, False


def rows(correct, wrong=None, unans=(F,) * 10, block=(F,) * 20):
    return {"correct": list(correct), "wrong": list(wrong or [F] * len(correct)), "unans": list(unans),
            "kb_takes_block": list(block)}


def test_rows_count_right_wrong_unanswerable_and_taken_block_questions() -> None:
    qs = [Question("a", "answerable", GOLD), Question("b", "answerable", GOLD), Question("c", "answerable", GOLD),
          Question("d", "unanswerable"), Question("e", "block", intent="weather"), Question("f", "block", intent="time")]
    out = rows_for(qs, [("knowledge", RIGHT), ("knowledge", OTHER), ("fallback", None), ("knowledge", False),
                        ("knowledge", RIGHT), ("intent", None)])
    assert out == {"correct": [T, F, F], "wrong": [F, T, F], "unans": [T], "kb_takes_block": [T, F]}
    assert rows_for([], []) == {"correct": [], "wrong": [], "unans": [], "kb_takes_block": []}


def test_choose_keeps_the_constraints_then_takes_the_most_correct() -> None:
    dev = {NOW: rows([T] * 5 + [F] * 5),
           ("mean", 0.3): rows([T] * 9 + [F], wrong=[F] * 9 + [T]),  # +0.10 wrong: not allowed
           ("mean", 0.5): rows([T] * 7 + [F] * 3),
           ("geo", 0.2): rows([T] * 8 + [F] * 2, unans=[T, T] + [F] * 8),  # 0.80 unanswered: not allowed
           ("geo", 0.4): rows([T] * 7 + [F] * 3, block=[T, T] + [F] * 18)}  # 0.10 taken: not allowed
    assert choose(dev) == ("mean", 0.5)
    tie = {**dev, ("geo", 0.6): rows([T] * 7 + [F] * 3)}
    assert choose(tie) == ("geo", 0.6)  # same correct and wrong: the higher threshold
    assert choose(dev, [("geo", 0.2), NOW]) == NOW
    with pytest.raises(BenchError, match="no views"):
        choose(dev, [("mean", 0.3), ("geo", 0.2)])
    with pytest.raises(BenchError, match="today"):
        choose({("mean", 0.5): dev[("mean", 0.5)]})


def test_criteria_follow_the_plan() -> None:
    now = rows([T] * 20 + [F] * 20, unans=[F] * 10)
    better = rows([T] * 32 + [F] * 8, unans=[T] + [F] * 9)
    checks = dict(criteria(now, better, [T] * 10, [T] * 10, 150.0))
    assert len(checks) == 6 and all(checks.values())
    worse = dict(criteria(now, rows([T] * 20 + [F] * 20, wrong=[T] * 2 + [F] * 38, unans=[T, T] + [F] * 8,
                                    block=[T] * 2 + [F] * 18), [T] * 10, [T] * 8 + [F] * 2, 250.0))
    assert not any(worse.values())


def test_memo_runs_each_key_once_and_the_report_names_the_verdict() -> None:
    calls = []
    twice = memo(lambda k: calls.append(k) or k * 2)
    assert (twice(2), twice(2), calls) == (4, 4, [2])
    r = rows([T, F])
    chosen = ("mean", 0.3)
    text = report(chosen, {NOW: r, chosen: r}, {"원문": {NOW: r, chosen: r}}, {c: {"in_scope": [T], "oos": [T]} for c in (NOW, chosen)},
                  {NOW: [10.0], chosen: [20.0]}, [("① x", True), ("② y", False)], {"답할 수 있는": 2})
    assert "고른 것: **mean 0.3**" in text and "✅ ① x · ❌ ② y" in text and "| 지식베이스 턴 p95 (해석 + 판단, ms) | 10 | 20 |" in text
