"""E-J measurement: does the judge answer more when it also reads the parts of a candidate?
(docs/engine-core-plan.md "빌더 단계 E-J")

The product path: the sample bot's deploy-trained model, `NluInterpreter` with `Knowledge.consult`, the
cross-encoder judge reading each candidate whole ("whole", today) or, for the best of them, whole plus
parts ("geo", "mean"; chatcore.knowledge.score_candidates). The views and threshold are chosen on the dev half of
bench/data/kb_eval.yaml; the verdict is the test half against today's judge (0.11, whole).

Usage: uv run python bench/judge_eval.py
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.augment_eval import SCRATCH, load_block_test, noise, paired, runtime_for, write_bot  # noqa: E402
from bench.kb_eval import JUDGE_GRID, QUESTIONS, BenchError, Question, load_kb, load_questions  # noqa: E402
from bench.spelling_eval import NATURAL, kb_rows, load_natural  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs/benchmarks/judge.md"
VIEWS = ("whole", "geo", "mean")
NOW = ("whole", 0.11)  # chatcore.knowledge.CROSS_ENCODER_THRESHOLD
UNANSWERED_MIN = 0.90  # answerable-by-nothing questions left unanswered (bench/kb_eval.py CRITERIA "fallback")
KB_TAKES_BLOCK_MAX = 0.05
WRONG_MORE = 0.03
UNANS_MORE = 1  # questions
BLOCK_DROP = 0.02
P95_MS = 200.0

Config = tuple[str, float]
Rows = dict[str, list[bool]]


def rows_for(questions: Sequence[Question], outcomes: Sequence[tuple[str, Any]]) -> Rows:
    """`outcomes[i]`: (rule, the knowledge-base item answered | False for a non-benchmark entry | None)."""
    kb = [(q, o) for q, o in zip(questions, outcomes, strict=True) if q.kind != "block"]
    rows = kb_rows([q for q, _ in kb], [item for _, (_, item) in kb])
    rows["kb_takes_block"] = [rule == "knowledge" for q, (rule, _) in zip(questions, outcomes, strict=True)
                              if q.kind == "block"]
    return rows


def mean(v: Sequence[bool]) -> float:
    return float(np.mean(v)) if len(v) else 0.0


def choose(dev: Mapping[Config, Rows], among: Sequence[Config] | None = None) -> Config:
    """On the dev half (from `among`, default all): unanswerable questions left unanswered at least
    UNANSWERED_MIN, block questions taken at most KB_TAKES_BLOCK_MAX, wrong answers at most WRONG_MORE
    over today's; then the most correct, the fewest wrong, the highest threshold."""
    if NOW not in dev:
        raise BenchError("today's judge was not measured")
    wrong_now = mean(dev[NOW]["wrong"])

    def allowed(r: Rows) -> bool:
        return (1 - mean(r["unans"]) >= UNANSWERED_MIN and mean(r["kb_takes_block"]) <= KB_TAKES_BLOCK_MAX
                and mean(r["wrong"]) <= wrong_now + WRONG_MORE + 1e-9)

    ok = [c for c in (dev if among is None else among) if allowed(dev[c])]
    if not ok:
        raise BenchError("no views and threshold meet the dev-half constraints")
    return max(ok, key=lambda c: (sum(dev[c]["correct"]), -sum(dev[c]["wrong"]), c[1]))


def criteria(now: Rows, chosen: Rows, block_now: Sequence[bool], block_chosen: Sequence[bool],
             p95_ms: float) -> list[tuple[str, bool]]:
    """The pre-registered verdict on the test half (docs/engine-core-plan.md "빌더 단계 E-J")."""
    _, (low, _) = paired(now["correct"], chosen["correct"])
    return [
        ("① 맞게 답함이 오름(구간 하한 > 0)", low > 0),
        (f"② 틀리게 답함 +{WRONG_MORE} 이내", mean(chosen["wrong"]) - mean(now["wrong"]) <= WRONG_MORE + 1e-9),
        (f"③ 답 없는 질문에 답함 +{UNANS_MORE}문항 이내", sum(chosen["unans"]) - sum(now["unans"]) <= UNANS_MORE),
        (f"④ 블록 질문을 지식베이스가 가져감 {KB_TAKES_BLOCK_MAX} 이하", mean(chosen["kb_takes_block"]) <= KB_TAKES_BLOCK_MAX),
        (f"⑤ 블록 독립 테스트셋 범위 안 정확도 −{BLOCK_DROP} 이내", mean(block_chosen) - mean(block_now) >= -BLOCK_DROP - 1e-9),
        (f"⑥ 지식베이스 턴 p95 {P95_MS:g}ms 이하", p95_ms <= P95_MS),
    ]


def memo(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    seen: dict[Any, Any] = {}

    def run(key: Any) -> Any:
        if key not in seen:
            seen[key] = fn(key)
        return seen[key]
    return run


def cell(now: Sequence[bool], other: Sequence[bool]) -> str:
    d, (lo, hi) = paired(now, other)
    return f"{mean(other):.3f} ({d:+.3f} [{lo:+.3f}, {hi:+.3f}])"


def report(chosen: Config, dev: Mapping[Config, Rows], test: Mapping[str, Mapping[Config, Rows]],
           blocks: Mapping[Config, Mapping[str, list[bool]]], latency: Mapping[Config, list[float]],
           checks: list[tuple[str, bool]], counts: Mapping[str, int]) -> str:
    name = {NOW: f"지금 ({NOW[0]} {NOW[1]:g})", chosen: f"고른 것 ({chosen[0]} {chosen[1]:g})"}
    lines = ["# 판단 모델이 부분도 읽기 측정 (E-J)", "",
             "`uv run python bench/judge_eval.py`가 만든 파일. 손으로 고치지 말 것.", "",
             "- 경로: 배포 학습한 샘플 봇 + `NluInterpreter` + `Knowledge.consult`(교차 인코더 판단). 질문은 `bench/data/kb_eval.yaml`.",
             "- whole: 후보 전체만(지금). geo / mean: 판단 1위 후보만 부분(문서는 한 문장·이웃 두 문장, FAQ는 질문·답)도 읽어, 전체 점수와 가장 높은 부분 점수의 기하·산술 평균을 그 후보의 점수로.",
             "- 고르기: 개발용 절반·원문. 판정: 시험용 절반. 칸: 값 (지금과의 차이 [같은 질문끼리 쌍 부트스트랩 95% 구간]).",
             f"- 시험용 절반 문항 수: {', '.join(f'{k} {v}' for k, v in counts.items())}.", "",
             "## 개발용 절반: 합치기마다 가장 좋은 기준점", "",
             "| 합치기 | 기준점 | 맞게 답함 | 틀리게 답함 | 답 없는 질문에 답함 | 블록 질문 가져감 |", "|---|---|---|---|---|---|"]
    for views in VIEWS:
        try:
            best = choose(dev, [c for c in dev if c[0] == views])
        except BenchError:
            lines.append(f"| {views} | (조건을 맞추는 기준점 없음) | | | | |")
            continue
        r = dev[best]
        lines.append(f"| {views} | {best[1]:g} | {mean(r['correct']):.3f} | {mean(r['wrong']):.3f} | "
                     f"{mean(r['unans']):.3f} | {mean(r['kb_takes_block']):.3f} |")
    lines += ["", f"고른 것: **{chosen[0]} {chosen[1]:g}**", "", "## 시험용 절반", "",
              f"| | {name[NOW]} | {name[chosen]} |", "|---|---|---|"]
    for set_name, by in test.items():
        a, b = by[NOW], by[chosen]
        lines += [f"| {set_name} 맞게 답함 (n={len(a['correct'])}) | {mean(a['correct']):.3f} | {cell(a['correct'], b['correct'])} |",
                  f"| {set_name} 틀리게 답함 | {mean(a['wrong']):.3f} | {cell(a['wrong'], b['wrong'])} |",
                  f"| {set_name} 답 없는 질문에 답함 (n={len(a['unans'])}) | {mean(a['unans']):.3f} | {cell(a['unans'], b['unans'])} |"]
        if a["kb_takes_block"]:
            lines.append(f"| {set_name} 블록 질문 가져감 (n={len(a['kb_takes_block'])}) | {mean(a['kb_takes_block']):.3f} | "
                         f"{cell(a['kb_takes_block'], b['kb_takes_block'])} |")
    for key, label in (("in_scope", "범위 안 정확도"), ("oos", "범위 밖 거절률")):
        a, b = blocks[NOW][key], blocks[chosen][key]
        lines.append(f"| 블록 독립 테스트셋 {label} (n={len(a)}) | {mean(a):.3f} | {cell(a, b)} |")
    p95 = {c: float(np.percentile(v, 95)) for c, v in latency.items()}
    lines.append(f"| 지식베이스 턴 p95 (해석 + 판단, ms) | {p95[NOW]:.0f} | {p95[chosen]:.0f} |")
    lines += ["", "## 기준 (미리 정함, 시험용 절반·원문)", "", " · ".join(f"{'✅' if ok else '❌'} {t}" for t, ok in checks), ""]
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - real models
    from chatcore.dialogue.state import DialogueState
    from chatcore.knowledge import CrossEncoderJudge, Knowledge, KnowledgeIndex, score_candidates
    from chatcore.nlu.interpreter import NluInterpreter
    from chatcore.nlu.model import SentenceEmbedder

    embedder = SentenceEmbedder()
    nlu = yaml.safe_load((ROOT / "bots/sample/nlu.yaml").read_text(encoding="utf-8"))
    rt = runtime_for(write_bot(SCRATCH / "base", nlu), embedder)
    kb_root = ROOT / "bench/data/kb"
    faq = yaml.safe_load((kb_root / "faq.yaml").read_text(encoding="utf-8"))["faq"]
    docs = [{"id": p.stem, "name": p.name, "text": p.read_text(encoding="utf-8")} for p in sorted((kb_root / "docs").glob("*.md"))]
    index = KnowledgeIndex.build({"faq": faq, "documents": docs}, embedder)
    entries = {e.id: e for e in [*index.faq, *index.chunks]}
    items = load_kb()
    by_text, by_id = {i.text: i for i in items}, {i.id: i for i in items}
    questions = load_questions(QUESTIONS, items)
    cross = CrossEncoderJudge()
    pair_score = memo(lambda pair: float(cross.score_pairs([pair])[0]))

    class Judge:
        name = "cross-encoder"

        def __init__(self, views: str, threshold: float, fn: Callable[[Any], np.ndarray]) -> None:
            self.views, self.threshold, self.fn = views, threshold, fn

        def score(self, query: str, candidates: Sequence[Any]) -> np.ndarray:
            return score_candidates(self.fn, query, candidates, self.views)  # type: ignore[arg-type]

    def cached(texts: list[tuple[str, str]]) -> np.ndarray:
        return np.array([pair_score(p) for p in texts])

    parse = memo(rt.model.parse)

    def outcomes(cfg: Config, qs: Sequence[Question], fn: Callable[[Any], np.ndarray] = cached,
                 the_parse: Callable[[str], Any] = parse) -> list[tuple[str, Any]]:
        interp = NluInterpreter(the_parse, Knowledge(index, Judge(cfg[0], cfg[1], fn), embedder))
        out: list[tuple[str, Any]] = []
        for q in qs:
            info = interp.interpret(q.text, DialogueState(), rt.bot).info
            answered = (info.get("knowledge") or {}).get("answered") if info["rule"] == "knowledge" else None
            if not answered:
                out.append((info["rule"], None))
                continue
            first = entries[answered[0]]
            out.append((info["rule"], (by_id.get(first.id) if first.kind == "faq" else by_text.get(first.judge_text)) or False))
        return out

    dev_q = [q for q in questions if q.half == 0]
    grid = [(v, float(t)) for v in VIEWS for t in JUDGE_GRID] + [NOW]
    dev = {cfg: rows_for(dev_q, outcomes(cfg, dev_q)) for cfg in grid}
    chosen = choose(dev)
    print("chosen", chosen, flush=True)

    test_q = [q for q in questions if q.half == 1]
    _, natural_kb = load_natural(NATURAL, [q for q in questions if q.kind != "block"])
    sets = {"원문": test_q,
            "규칙 오타": [Question(noise(q.text)[0], q.kind, q.gold, q.intent, q.half) for q in test_q],
            "자연 오타": [q for q in natural_kb if q.half == 1]}
    test = {name: {cfg: rows_for(qs, outcomes(cfg, qs)) for cfg in (NOW, chosen)} for name, qs in sets.items()}
    block = load_block_test()
    blocks: dict[Config, dict[str, list[bool]]] = {}
    for cfg in (NOW, chosen):
        interp = NluInterpreter(parse, Knowledge(index, Judge(cfg[0], cfg[1], cached), embedder))
        in_scope, oos = [], []
        for s in block:
            info = interp.interpret(s.text, DialogueState(), rt.bot).info
            if s.gold.startswith("out_of_scope:"):
                oos.append(info["rule"] == "fallback")
            else:
                in_scope.append(info["rule"] == "intent" and bool(info.get("intents")) and info["intents"][0][0] == s.gold)
        blocks[cfg] = {"in_scope": in_scope, "oos": oos}

    kb_test = [q for q in test_q if q.kind != "block"]
    latency: dict[Config, list[float]] = {}
    for cfg in (NOW, chosen):  # uncached: the real cost of a turn (warm the models first)
        outcomes(cfg, kb_test[:5], lambda texts: cross.score_pairs(texts), rt.model.parse)
        times = []
        for q in kb_test:
            t0 = time.perf_counter()
            outcomes(cfg, [q], lambda texts: cross.score_pairs(texts), rt.model.parse)
            times.append((time.perf_counter() - t0) * 1000)
        latency[cfg] = times

    p95 = float(np.percentile(latency[chosen], 95))
    checks = criteria(test["원문"][NOW], test["원문"][chosen], blocks[NOW]["in_scope"], blocks[chosen]["in_scope"], p95)
    counts = {"답할 수 있는": sum(q.kind == "answerable" for q in test_q), "답 없는": sum(q.kind == "unanswerable" for q in test_q),
              "블록": sum(q.kind == "block" for q in test_q)}
    REPORT.write_text(report(chosen, dev, test, blocks, latency, checks, counts), encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
