"""E-I measurement: does correcting typos against the bot's own words help? (docs/engine-core-plan.md
"빌더 단계 E-I")

The product path with and without `chatcore.nlu.spelling.Speller` in the interpreter: the sample bot's
deploy-trained model, a fresh session per sentence, the knowledge base through the same interpreter
(rule first, then `Knowledge.consult`, cross-encoder judge). Sentences: as written, with one rule-made
typo (picks MARGIN only — the rule is the same kind of edit the speller undoes), and the natural-typo
set a separate agent wrote without seeing the method (bench/data/typo_eval.yaml — the verdict).

Usage: uv run python bench/spelling_eval.py   (the E-H proposal cache in .claude/tmp/augment_cache.json
gives the 14B similar questions; without it that configuration is skipped)
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.augment_eval import (  # noqa: E402
    CACHE,
    SCRATCH,
    Sentence,
    augmented_faq,
    kb_correct,
    load_block_test,
    noise,
    paired,
    proposals,
    runtime_for,
    write_bot,
)
from bench.kb_eval import QUESTIONS, BenchError, Question, load_kb, load_questions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs/benchmarks/spelling.md"
NATURAL = ROOT / "bench/data/typo_eval.yaml"
MARGINS = (3.0, 5.0, 8.0)
DROP = 0.02
SIMILAR_MODEL = "mlx-community/Qwen3-14B-4bit"


def load_natural(path: Path, questions: Sequence[Question]) -> tuple[list[Sentence], list[Question]]:
    """The natural-typo set: block sentences (in scope by intent, out of scope) and knowledge-base
    questions mapped to the gold of the kb_eval.yaml question each was rewritten from."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    blocks = [Sentence(t, intent) for intent, texts in raw["blocks"].items() for t in texts]
    blocks += [Sentence(t, "out_of_scope:natural") for t in raw["out_of_scope"]]
    by_text = {q.text: q for q in questions}
    kb: list[Question] = []
    for row in [*raw["kb"], *raw["kb_unanswerable"]]:
        source = by_text.get(row["source"])
        if source is None:
            raise BenchError(f"{row['source']!r} is not a kb_eval.yaml question")
        kb.append(Question(row["q"], source.kind, source.gold, source.intent, source.half))
    return blocks, kb


def scores(golds: Sequence[str], decisions: Sequence[tuple[str, str | None]]) -> dict[str, list[bool]]:
    """Per sentence: in scope -> the interpreter started the gold intent; out of scope (hard / easy /
    natural) -> it fell back."""
    out: dict[str, list[bool]] = {"in_scope": [], "oos_hard": [], "oos_easy": [], "oos_natural": []}
    for gold, (rule, top) in zip(golds, decisions, strict=True):
        if gold.startswith("out_of_scope:"):
            out["oos_" + gold.split(":")[1]].append(rule == "fallback")
        else:
            out["in_scope"].append(rule == "intent" and top == gold)
    return out


def kb_rows(questions: Sequence[Question], answers: Sequence[Any]) -> dict[str, list[bool]]:
    """`answers`: the knowledge-base item each question was answered with, False for an answer that is
    not a knowledge-base item, None for no knowledge-base answer."""
    rows: dict[str, list[bool]] = {"correct": [], "wrong": [], "unans": []}
    for q, item in zip(questions, answers, strict=True):
        if q.kind == "answerable":
            ok = item not in (None, False) and kb_correct(q, item)
            rows["correct"].append(ok)
            rows["wrong"].append(item is not None and not ok)
        else:
            rows["unans"].append(item is not None)
    return rows


def main() -> int:  # pragma: no cover - real models
    from kiwipiepy import Kiwi

    from chatcore.dialogue.state import DialogueState
    from chatcore.knowledge import CrossEncoderJudge, Knowledge, KnowledgeIndex
    from chatcore.nlu.interpreter import NluInterpreter
    from chatcore.nlu.model import SentenceEmbedder
    from chatcore.nlu.spelling import Speller, kiwi_score, vocabulary

    embedder = SentenceEmbedder()
    nlu = yaml.safe_load((ROOT / "bots/sample/nlu.yaml").read_text(encoding="utf-8"))
    rt = runtime_for(write_bot(SCRATCH / "base", nlu), embedder)
    kb_root = ROOT / "bench/data/kb"
    faq = yaml.safe_load((kb_root / "faq.yaml").read_text(encoding="utf-8"))["faq"]
    docs = [{"id": p.stem, "name": p.name, "text": p.read_text(encoding="utf-8")} for p in sorted((kb_root / "docs").glob("*.md"))]
    indexes = {"원래 색인": KnowledgeIndex.build({"faq": faq, "documents": docs}, embedder)}
    if CACHE.exists():
        from chatcore import augment

        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        similar = {}
        for f in faq:
            items = proposals(SIMILAR_MODEL, augment.faq_messages(f["question"], f["answer"], [], 10), cache, lambda m: [])
            similar[f["id"]] = [p.text for p in augment.screen(items, [f["question"]], set(), None) if p.status == "ok"][:50]
        indexes["E-H 비슷한 질문 색인"] = KnowledgeIndex.build({"faq": augmented_faq(faq, similar), "documents": docs}, embedder)

    items = load_kb()
    by_text, by_id = {i.text: i for i in items}, {i.id: i for i in items}
    questions = [q for q in load_questions(QUESTIONS, items) if q.kind != "block"]
    natural_blocks, natural_kb = load_natural(NATURAL, questions)
    block = load_block_test()
    sets_blocks = {"원문": block, "규칙 오타": [Sentence(noise(s.text)[0], s.gold) for s in block], "자연 오타": natural_blocks}
    sets_kb = {"원문": questions, "규칙 오타": [Question(noise(q.text)[0], q.kind, q.gold, q.intent, q.half) for q in questions],
               "자연 오타": natural_kb}
    kiwi = Kiwi()
    score = kiwi_score(kiwi)
    judge = CrossEncoderJudge()

    def speller(margin: float | None, index: Any) -> Speller | None:
        return None if margin is None else Speller(vocabulary(rt.bot, index), score, margin)

    configs: list[float | None] = [None, *MARGINS]
    blocks: dict[Any, dict[str, dict[str, list[bool]]]] = {}
    for margin in configs:
        interp = NluInterpreter(rt.model.parse, None, speller(margin, None))
        blocks[margin] = {}
        for name, sentences in sets_blocks.items():
            dec = []
            for s in sentences:
                info = interp.interpret(s.text, DialogueState(), rt.bot).info
                dec.append((info["rule"], info["intents"][0][0] if info.get("intents") else None))
            blocks[margin][name] = scores([s.gold for s in sentences], dec)
        print("blocks", margin, flush=True)
    kb: dict[Any, dict[str, dict[str, list[bool]]]] = {}
    for iname, index in indexes.items():
        knowledge = Knowledge(index, judge, embedder)
        entries = {e.id: e for e in [*index.faq, *index.chunks]}
        for margin in configs:
            interp = NluInterpreter(rt.model.parse, knowledge, speller(margin, index))
            kb[(iname, margin)] = {}
            for name, qs in sets_kb.items():
                answers: list[Any] = []
                for q in qs:
                    info = interp.interpret(q.text, DialogueState(), rt.bot).info
                    answered = (info.get("knowledge") or {}).get("answered") if info["rule"] == "knowledge" else None
                    if not answered:
                        answers.append(None)
                        continue
                    first = entries[answered[0]]
                    answers.append((by_id.get(first.id) if first.kind == "faq" else by_text.get(first.judge_text)) or False)
                kb[(iname, margin)][name] = kb_rows(qs, answers)
            print("kb", iname, margin, flush=True)
    examples = []
    for text in ("오늘 날시 알려줘", "고마어", "미새먼지 나쁨은 몇부터예요?", "부삼 날씨 어때", "미제먼지 나쁨 기줌이 뭐예요"):
        fixed = Speller(vocabulary(rt.bot, indexes["원래 색인"]), score, 5.0).correct(text)
        examples.append((text, fixed.text))
    REPORT.write_text(report(blocks, kb, list(indexes), examples, sets_blocks, sets_kb), encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 0


def choose(blocks: dict[Any, Any], kb: dict[Any, Any], index: str) -> float:
    """MARGIN by the rule-typo set only: the most correct answers (blocks in scope + knowledge base)
    with out-of-scope rejection and unanswerable handling within DROP of no correction."""
    def ok(m: float) -> bool:
        b, k, b0, k0 = blocks[m]["규칙 오타"], kb[(index, m)]["규칙 오타"], blocks[None]["규칙 오타"], kb[(index, None)]["규칙 오타"]
        return (np.mean(b["oos_hard"]) >= np.mean(b0["oos_hard"]) - DROP
                and np.mean(k["unans"]) <= np.mean(k0["unans"]) + DROP)
    # a tie goes to the larger MARGIN (fewer changes to what was typed); added after 3 and 5 tied (209 = 209)
    ranked = sorted(MARGINS, key=lambda m: (-(sum(blocks[m]["규칙 오타"]["in_scope"]) + sum(kb[(index, m)]["규칙 오타"]["correct"])), -m))
    return next((m for m in ranked if ok(m)), max(MARGINS))


def criteria(blocks: dict[Any, Any], kb: dict[Any, Any], index: str, m: float) -> list[tuple[str, bool]]:
    mean = lambda v: float(np.mean(v)) if v else 0.0  # noqa: E731
    b, b0 = blocks[m], blocks[None]
    k, k0 = kb[(index, m)], kb[(index, None)]
    return [
        ("자연 오타 블록 범위 안 정확도가 오름", paired(b0["자연 오타"]["in_scope"], b["자연 오타"]["in_scope"])[1][0] > 0),
        ("자연 오타 지식베이스 맞게 답함이 오름", paired(k0["자연 오타"]["correct"], k["자연 오타"]["correct"])[1][0] > 0),
        ("원문 블록 정확도 −0.02 이내", mean(b["원문"]["in_scope"]) >= mean(b0["원문"]["in_scope"]) - DROP),
        ("원문 지식베이스 맞게 답함 −0.02 이내", mean(k["원문"]["correct"]) >= mean(k0["원문"]["correct"]) - DROP),
        ("원문 범위 밖(hard) 거절률 −0.02 이내", mean(b["원문"]["oos_hard"]) >= mean(b0["원문"]["oos_hard"]) - DROP),
        ("원문 범위 밖(easy) 거절률 −0.02 이내", mean(b["원문"]["oos_easy"]) >= mean(b0["원문"]["oos_easy"]) - DROP),
        ("자연 오타 범위 밖 거절률 −0.02 이내", mean(b["자연 오타"]["oos_natural"]) >= mean(b0["자연 오타"]["oos_natural"]) - DROP),
        ("원문 답 없는 질문에 답함 +0.02 이내", mean(k["원문"]["unans"]) <= mean(k0["원문"]["unans"]) + DROP),
        ("자연 오타 답 없는 질문에 답함 +0.02 이내", mean(k["자연 오타"]["unans"]) <= mean(k0["자연 오타"]["unans"]) + DROP),
    ]


def report(blocks: dict[Any, Any], kb: dict[Any, Any], indexes: list[str], examples: list[tuple[str, str]],
           sets_blocks: dict[str, Any], sets_kb: dict[str, Any]) -> str:  # pragma: no cover - formatting
    def cell(a: list[bool], b: list[bool]) -> str:
        if not a:
            return "—"
        if a is b:
            return f"{np.mean(a):.3f}"
        m, (lo, hi) = paired(a, b)
        return f"{np.mean(b):.3f} ({m:+.3f} [{lo:+.3f}, {hi:+.3f}])"

    names = ["끔", *[f"MARGIN {m:g}" for m in MARGINS]]
    configs: list[float | None] = [None, *MARGINS]
    lines = ["# 봇의 낱말로 오타 고치기 측정 (E-I)", "", "`uv run python bench/spelling_eval.py`가 만든 파일. 손으로 고치지 말 것.", "",
             "- 경로: 배포 학습한 샘플 봇 모델 + 해석기에 고치기를 넣고 뺌(새 세션마다 한 문장). 지식베이스도 같은 해석기(블록 판단 → `Knowledge.consult`).",
             "- 평가 문장: 원문, 규칙 오타(자모 하나 — 이 방법과 같은 꼴이라 MARGIN 고르기에만), **자연 오타**(`bench/data/typo_eval.yaml`, 방법을 모르는 별도 에이전트가 씀 — 판정).",
             "- 블록 문장 수: " + ", ".join(f"{k} {len(v)}" for k, v in sets_blocks.items()) + "; 지식베이스: " + ", ".join(f"{k} {len(v)}" for k, v in sets_kb.items()) + ".",
             "- 칸: 값 (끔과의 차이 [같은 문장끼리 쌍 부트스트랩 95% 구간]).", "",
             "## 블록", "", "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for set_name in ("원문", "규칙 오타", "자연 오타"):
        for key, title in (("in_scope", "범위 안 정확도"), ("oos_hard", "범위 밖(hard) 거절률"), ("oos_easy", "범위 밖(easy) 거절률"),
                           ("oos_natural", "범위 밖(자연 오타) 거절률")):
            base = blocks[None][set_name].get(key)
            if not base:
                continue
            lines.append(f"| {set_name} {title} (n={len(base)}) | " + " | ".join(cell(base, blocks[c][set_name][key]) for c in configs) + " |")
    for index in indexes:
        lines += ["", f"## 지식베이스 — {index}", "", "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for set_name in ("원문", "규칙 오타", "자연 오타"):
            for key, title in (("correct", "맞게 답함"), ("wrong", "틀리게 답함"), ("unans", "답 없는 질문에 답함")):
                base = kb[(index, None)][set_name][key]
                lines.append(f"| {set_name} {title} (n={len(base)}) | " + " | ".join(cell(base, kb[(index, c)][set_name][key]) for c in configs) + " |")
    lines += ["", "## 고친 예 (MARGIN 5)", "", "| 친 문장 | 이해한 문장 |", "|---|---|"] + [f"| {a} | {b} |" for a, b in examples]
    lines += ["", "## MARGIN과 기준 (미리 정함)", ""]
    for index in indexes:
        m = choose(blocks, kb, index)
        checks = criteria(blocks, kb, index, m)
        lines.append(f"- **{index}:** 규칙 오타로 고른 MARGIN {m:g} — " + " · ".join(f"{'✅' if ok else '❌'} {t}" for t, ok in checks))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
