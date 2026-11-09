"""Evaluate fixed NLU configurations on the independent (lockbox) test set.

bench/data/test_independent.yaml was written by a separate agent that never saw
the training data or earlier failures. Use it to compare configurations that
were fully decided beforehand — never to tune them.

Each configuration: bot NLU data + embedding model + C + thresholds chosen by
bench/phase0.py cross-validation, and whether the greeting/bye position rule
is on. Writes docs/benchmarks/phase0b.md.

Usage: uv run python bench/eval_independent.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from kiwipiepy import Kiwi
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.phase0 import (  # noqa: E402
    EMBEDDINGS,
    OOS,
    BenchError,
    Delexicalizer,
    Embedder,
    IntentModel,
    Setup,
    decide,
    load_bot,
    normalize,
    prepare,
    training_rows,
)

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "bench/data/test_independent.yaml"
EDGE = ROOT / "bench/data/test_dialogues_edge.yaml"  # mid-chat re-greetings, first-turn goodbyes
FAR = 0.80  # test sentences with max arctic cosine to every training row below this are "far"
REPORT = ROOT / "docs/benchmarks/phase0b.md"

PAIR = ("greeting", "bye")
PAIR_MARGIN = 0.5  # fixed before looking at the test set


@dataclass(frozen=True)
class Config:
    name: str
    nlu: str  # path in the working tree, or "<commit>:<path>"
    setup: Setup
    threshold: float
    sim_threshold: float
    pair_rule: bool


def resolve_pair(label: str, probs: np.ndarray, classes: np.ndarray, turn_index: int, margin: float = PAIR_MARGIN) -> str:
    """Greeting vs bye by dialogue position: on the first user turn prefer
    greeting unless bye wins by >= margin; later, prefer bye unless greeting
    wins by >= margin. Other labels pass through."""
    if label not in PAIR:
        return label
    if turn_index < 0:
        raise BenchError("turn_index must be >= 0")
    index = {c: i for i, c in enumerate(classes)}
    if not all(p in index for p in PAIR):
        raise BenchError("classes need greeting and bye")
    pg, pb = probs[index["greeting"]], probs[index["bye"]]
    if turn_index == 0:
        return "bye" if pb - pg >= margin else "greeting"
    return "greeting" if pg - pb >= margin else "bye"


def load_test(path: Path) -> dict:
    t = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("in_scope", "out_of_scope", "dialogues"):
        if key not in t:
            raise BenchError(f"{path} lacks {key!r}")
    return t


def read_nlu(spec: str, scratch: Path) -> Path:
    """A working-tree path, or '<commit>:<path>' materialized under scratch."""
    if ":" not in spec:
        return ROOT / spec
    out = scratch / spec.replace(":", "_").replace("/", "_")
    r = subprocess.run(["git", "-C", str(ROOT), "show", spec], capture_output=True, text=True)
    if r.returncode != 0:
        raise BenchError(f"git show {spec} failed: {r.stderr.strip()}")
    out.write_text(r.stdout, encoding="utf-8")
    return out


def evaluate_config(cfg: Config, test: dict, kiwi: Kiwi, embedder: Embedder, scratch: Path, edge: dict | None = None) -> dict:
    intents, forms = load_bot(read_nlu(cfg.nlu, scratch))
    delex = Delexicalizer(forms, kiwi)
    rows, row_labels = training_rows(cfg.setup, intents, forms, delex)
    model = IntentModel(cfg.setup, kiwi, embedder, fresh=True).fit(rows, row_labels)
    train_texts = {normalize(r) for r in rows} | {e for exs in intents.values() for e in exs}
    train_emb = embedder.encode(rows)

    def predict(texts: list[str]) -> tuple[list[str], np.ndarray]:
        probs, sims = model.predict([prepare(cfg.setup, t, delex) for t in texts])
        return decide(probs, model.classes_, cfg.threshold, sims, cfg.sim_threshold), probs

    sents, labels, kinds = [], [], []
    for intent, exs in test["in_scope"].items():
        sents += exs
        labels += [intent] * len(exs)
        kinds += ["in"] * len(exs)
    for kind in ("hard", "easy"):
        exs = test["out_of_scope"][kind]
        sents += exs
        labels += [OOS] * len(exs)
        kinds += [kind] * len(exs)
    preds, _ = predict(sents)
    novel = [normalize(s) not in train_texts for s in sents]
    nearest = (embedder.encode([normalize(s) for s in sents]) @ train_emb.T).max(axis=1)
    far = [bool(m < FAR) for m in nearest]

    def scores(mask: list[bool]) -> dict:
        y = [lab for lab, m in zip(labels, mask, strict=True) if m]
        p = [q for q, m in zip(preds, mask, strict=True) if m]
        k = [q for q, m in zip(kinds, mask, strict=True) if m]
        in_rows = [(a, b) for a, b, c in zip(y, p, k, strict=True) if c == "in"]
        rej = {kind: [b == OOS for b, c in zip(p, k, strict=True) if c == kind] for kind in ("hard", "easy")}
        return {
            "n": len(y),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "in_acc": float(np.mean([a == b for a, b in in_rows])),
            "hard_reject": float(np.mean(rej["hard"])), "easy_reject": float(np.mean(rej["easy"])),
        }

    all_scores = scores([True] * len(sents))
    novel_scores = scores(novel)
    far_scores = scores(far)
    errors = [(s, lab, p) for s, lab, p in zip(sents, labels, preds, strict=True) if lab != p]

    def run_dialogues(dialogues: list[dict]) -> list[tuple[str, str, str]]:
        """(case, gold, final) per turn; turn_index = position of the user turn in the session."""
        out = []
        for d in dialogues:
            if len(d["turns"]) != len(d["labels"]):
                raise BenchError(f"dialogue turns/labels differ: {d['turns']}")
            t_preds, t_probs = predict(d["turns"])
            for i, (gold, pred) in enumerate(zip(d["labels"], t_preds, strict=True)):
                final = resolve_pair(pred, t_probs[i], model.classes_, i) if cfg.pair_rule else pred
                out.append((d.get("case", "main"), gold, final))
        return out

    def acc(rows: list[tuple[str, str, str]], only_pair: bool = False) -> tuple[float, int]:
        rows = [r for r in rows if not only_pair or r[1] in PAIR]
        return (float(np.mean([g == p for _, g, p in rows])) if rows else float("nan")), len(rows)

    turn_rows = run_dialogues(test["dialogues"])
    edge_rows = run_dialogues(edge["dialogues"]) if edge else []
    edge_by_case = {c: acc([r for r in edge_rows if r[0] == c], only_pair=True) for c in sorted({r[0] for r in edge_rows})}
    return {
        "all": all_scores, "novel": novel_scores, "far": far_scores, "errors": errors,
        "turn": acc(turn_rows), "pair": acc(turn_rows, True),
        "edge_turn": acc(edge_rows) if edge_rows else (float("nan"), 0), "edge_pair": acc(edge_rows, True) if edge_rows else (float("nan"), 0),
        "edge_by_case": edge_by_case,
    }


def render(results: list[tuple[Config, dict]]) -> str:
    lines = [
        "# Phase 0b — 독립 테스트셋 평가",
        "",
        "`uv run python bench/eval_independent.py`가 생성한 파일이다. 손으로 고치지 말 것.",
        "",
        "- 테스트셋 `bench/data/test_independent.yaml`은 학습 데이터와 이전 실패 사례를 보지 않은 별도 에이전트가 썼다(범위 안 135, 범위 밖 hard 60 / easy 40, 대화 20개).",
        "- 구성과 규칙(인사/작별 위치 규칙 margin "
        f"{PAIR_MARGIN})은 이 테스트를 보기 전에 정했다. 임계값은 `bench/phase0.py` 교차검증에서 고른 값이다.",
        "- '학습에 없는 문장만'은 실제 학습 행(지역을 채운 예문 포함)과 글자까지 같은 문장(\"안녕\" 등)을 뺀 값이다.",
        f"- '학습과 먼 문장만'은 `arctic` 임베딩으로 모든 학습 행과의 최대 코사인 < {FAR}인 문장만 센 값이다. hard negative 학습 문장과 테스트 문장은 같은 방식(의도 핵심어 + 다른 요구)으로 같은 날 쓰였다. 그래서 전체 수치는 **분포 안 성능**이고, 처음 보는 유형으로의 일반화는 이 열이 더 가깝다. 다만 구성마다 학습 데이터가 달라 n도 다르다.",
        "",
        "| 구성 | macro-F1 (전체) | 범위 안 정확도 | hard 범위 밖 거절률 | easy 범위 밖 거절률 | macro-F1 (학습에 없는 문장만) | 학습과 먼 문장만: macro-F1 / hard 거절률 |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg, r in results:
        a, n, f = r["all"], r["novel"], r["far"]
        lines.append(f"| {cfg.name} | {a['macro_f1']:.3f} | {a['in_acc']:.3f} | {a['hard_reject']:.2f} | {a['easy_reject']:.2f} "
                     f"| {n['macro_f1']:.3f} (n={n['n']}) | {f['macro_f1']:.3f} / {f['hard_reject']:.2f} (n={f['n']}) |")
    lines += ["", "## 대화 평가 (인사/작별 위치 규칙)", "",
              "- 기본 대화: 테스트셋의 대화 20개. 대부분 '첫 턴 인사, 마지막 턴 작별' 구조라 규칙에 유리하다.",
              "- 경계 대화: `bench/data/test_dialogues_edge.yaml`(대화 중간 재인사, 첫 턴 작별, 같은 말의 두 뜻, 섞인 경우). 규칙을 정한 뒤 별도 에이전트가 썼다.",
              "",
              "| 구성 | 기본: 턴 정확도 | 기본: 인사·작별 턴 | 경계: 턴 정확도 | 경계: 인사·작별 턴 | 경계 사례별 인사·작별 턴 정확도 |",
              "|---|---|---|---|---|---|"]
    for cfg, r in results:
        cases = ", ".join(f"{c} {v:.2f} (n={k})" for c, (v, k) in r["edge_by_case"].items()) or "-"
        lines.append(f"| {cfg.name} | {r['turn'][0]:.3f} (n={r['turn'][1]}) | {r['pair'][0]:.3f} (n={r['pair'][1]}) "
                     f"| {r['edge_turn'][0]:.3f} (n={r['edge_turn'][1]}) | {r['edge_pair'][0]:.3f} (n={r['edge_pair'][1]}) | {cases} |")
    lines += ["", "## 구성", ""]
    for cfg, _ in results:
        lines.append(f"- **{cfg.name}**: 데이터 `{cfg.nlu}`, `{cfg.setup.name}`, 확률 임계값 {cfg.threshold}, 유사도 임계값 {cfg.sim_threshold}, 위치 규칙 {'켬' if cfg.pair_rule else '끔'}")
    for cfg, r in results:
        lines += ["", f"## 틀린 문장 — {cfg.name} ({len(r['errors'])}개)", "", "| 문장 | 정답 | 예측 |", "|---|---|---|"]
        lines += [f"| {s} | {lab} | {p} |" for s, lab, p in r["errors"]] or ["| - | - | - |"]
    return "\n".join(lines) + "\n"


ARCTIC = Setup("expand", "emb:arctic", "logreg", 10.0)
# Thresholds: bench/phase0.py cross-validation on each data version.
CONFIGS = [
    Config("A. 기준 (커밋 c626440 데이터)", "c626440:bots/sample/nlu.yaml", ARCTIC, 0.35, 0.38, False),
    Config("B. 기준 + 위치 규칙", "c626440:bots/sample/nlu.yaml", ARCTIC, 0.35, 0.38, True),
    Config("C. hard negative 추가", "bots/sample/nlu.yaml", ARCTIC, 0.30, 0.39, False),
    Config("D. hard negative 추가 + 위치 규칙", "bots/sample/nlu.yaml", ARCTIC, 0.30, 0.39, True),
]


def main() -> int:
    test = load_test(TEST)
    edge = yaml.safe_load(EDGE.read_text(encoding="utf-8")) if EDGE.exists() else None
    kiwi = Kiwi()
    embedder = Embedder(EMBEDDINGS["arctic"])
    scratch = ROOT / ".claude/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        results = [(cfg, evaluate_config(cfg, test, kiwi, embedder, scratch, edge)) for cfg in CONFIGS]
    finally:
        for f in scratch.glob("*_bots_sample_nlu.yaml"):
            f.unlink()
    for cfg, r in results:
        a = r["all"]
        print(f"{cfg.name}: F1={a['macro_f1']:.3f} in={a['in_acc']:.3f} hard={a['hard_reject']:.2f} easy={a['easy_reject']:.2f} "
              f"novelF1={r['novel']['macro_f1']:.3f} far F1/hard={r['far']['macro_f1']:.3f}/{r['far']['hard_reject']:.2f} (n={r['far']['n']}) "
              f"pair={r['pair'][0]:.3f} edge pair={r['edge_pair'][0]:.3f} {r['edge_by_case']}")
    REPORT.write_text(render(results), encoding="utf-8")
    print(f"report → {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
