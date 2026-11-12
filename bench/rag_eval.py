"""Generative RAG measurement (engine stage E-F; docs/engine-core-plan.md "빌더 단계 E-F").

Same knowledge base and 300 questions as bench/kb_eval.py, and the engine's own path: the sample
bot's NLU decides the rule, `Knowledge.consult` (embedding ranking + cross-encoder judge) decides
whether the knowledge base answers, and `chatcore.generation.Generator` writes the answer through an
OpenAI-compatible server (`mlx_lm.server`, started here for each model).

Configurations: extractive (today) / generative per model x passages given (k) x what happens when a
generated answer is rejected ("extract": show the extractive answer, "fallback": answer nothing).
Metrics on the test half (dev half chooses), bootstrap 95% intervals:
  correct  = answerable question, answered, and what the answer says is backed by a right source: at
             least one fact (nouns, numbers, content verb/adjective stems; Kiwi), >= 60% of its facts found in a right source (the
             gold FAQ entry, or the chunk holding the gold evidence) and every number in that source
  wrong    = answered but not correct (answerable), or answered at all (unanswerable)
  recall   = the first metric tried: share of the gold text's facts found in the answer (>= 0.6). It
             marks a short, right answer ("나쁨입니다.") wrong because it drops the rest of the gold
             sentence, and an extractive answer (the gold text itself) right — reported, not used to choose
  unsupported = generated answers with a number not in the passages, BEFORE the check (model quality)
  length   = median characters of answers given; latency = knowledge-base turn p50/p95 incl. generation
No LLM grades another LLM: every metric is counted on strings.

Usage: uv run python bench/rag_eval.py   (needs `uv tool install mlx-lm` and the models downloaded)
Writes docs/benchmarks/rag.md; generations are cached in .claude/tmp/rag_cache.json.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.kb_eval import KB, QUESTIONS, BenchError, Question, bootstrap, load_kb, load_questions, nlu_decisions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs/benchmarks/rag.md"
CACHE = ROOT / ".claude/tmp/rag_cache.json"
MODELS = ("mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-14B-4bit")
DEPTHS = (1, 3)
PORT = 8800
FACT_BAR = 0.6
CRITERIA = {"fact_drop": 0.02, "p95_ms": 5000.0, "median_chars": 200}


# ---------------------------------------------------------------- facts (pure)


GENERIC_STEMS = frozenset({"하", "되", "있", "없", "그렇", "이렇", "저렇", "같", "않", "이", "아니"})


def fact_tokens(text: str, tokenize: Any) -> set[str]:
    """Numbers, nouns and content verb/adjective stems ("나쁘" of 나쁨) of a text; generic stems
    (하다, 되다, 있다, 그렇다 ...) say nothing checkable and are left out."""
    from chatcore.generation import numbers

    words = {t.form.lower() for t in tokenize(text)
             if t.tag.startswith(("NNG", "NNP", "SL", "VV", "VA", "XR")) and t.form not in GENERIC_STEMS}
    return words | numbers(text)


def fact_score(answer: str, gold_texts: Sequence[str], tokenize: Any) -> float:
    """Best share of one gold text's facts found in the answer (0 when nothing to compare)."""
    got = fact_tokens(answer, tokenize)
    best = 0.0
    for g in gold_texts:
        want = fact_tokens(g, tokenize)
        if want:
            best = max(best, len(want & got) / len(want))
    return best


@dataclass(frozen=True)
class Case:
    q: Question
    rule: str
    passages: tuple[Any, ...]  # engine Entries that passed the judge, best first (empty: no KB answer)
    extract: str | None  # today's answer text
    gold: tuple[str, ...]
    kb_ms: float  # retrieval + judge
    sources: tuple[str, ...] = ()  # right sources in full: gold FAQ entries (question + answer), chunks with the evidence


def backed(answer: str, sources: Sequence[str], tokenize: Any) -> bool:
    """The answer states something (a fact) and what it states is in one right source: >= FACT_BAR of
    its facts, and all of its numbers."""
    from chatcore.generation import numbers

    facts = fact_tokens(answer, tokenize)
    if not facts:
        return False
    for src in sources:
        have = fact_tokens(src, tokenize)
        if len(facts & have) / len(facts) >= FACT_BAR and numbers(answer) <= numbers(src):
            return True
    return False


def final_text(case: Case, generated: dict[str, Any] | None, on_reject: str) -> str | None:
    """What the bot says from the knowledge base under a configuration (None: it does not answer)."""
    if case.extract is None:
        return None
    if generated is None:
        return case.extract
    if generated["text"] is not None:
        return str(generated["text"])
    return case.extract if on_reject == "extract" else None


# ---------------------------------------------------------------- running (real models)


def build_cases() -> tuple[list[Case], Any]:  # pragma: no cover - real models
    from kiwipiepy import Kiwi

    from chatcore.knowledge import CrossEncoderJudge, Knowledge, KnowledgeIndex
    from chatcore.nlu.model import SentenceEmbedder

    items = load_kb()
    questions = load_questions(QUESTIONS, items)
    nlu, _ = nlu_decisions(questions)
    faq = yaml.safe_load((KB / "faq.yaml").read_text(encoding="utf-8"))["faq"]
    answers = {f["id"]: f["answer"] for f in faq}
    docs = [{"id": p.stem, "name": p.name, "text": p.read_text(encoding="utf-8")} for p in sorted((KB / "docs").glob("*.md"))]
    embedder = SentenceEmbedder()
    knowledge = Knowledge(KnowledgeIndex.build({"faq": faq, "documents": docs}, embedder), CrossEncoderJudge(), embedder)
    cases = []
    for q, n in zip(questions, nlu, strict=True):
        start = time.perf_counter()
        result, answer = knowledge.consult(q.text, n.rule, n.similarity, None)
        kb_ms = (time.perf_counter() - start) * 1000
        gold: tuple[str, ...] = ()
        sources: tuple[str, ...] = ()
        if q.gold is not None:
            gold = tuple(answers[i] for i in sorted(q.gold.faq)) + tuple(p for _, p in q.gold.phrases)
            sources = tuple(i.text for i in items if q.gold.hit(i))
        passages = tuple(h.entry for h in result.answers) if (result is not None and answer is not None) else ()
        cases.append(Case(q, n.rule, passages, answer.text if answer is not None else None, gold, kb_ms, sources))
    return cases, Kiwi()


def serve(model: str) -> subprocess.Popen[bytes]:  # pragma: no cover - starts a server
    proc = subprocess.Popen(["mlx_lm.server", "--model", model, "--port", str(PORT), "--host", "127.0.0.1",
                             "--chat-template-args", json.dumps({"enable_thinking": False})],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise BenchError(f"{model} did not start")


def generate_all(cases: list[Case]) -> dict[str, Any]:  # pragma: no cover - real models
    from chatcore.generation import Generator, LlmClient, LlmConfig

    cache: dict[str, Any] = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [c for c in cases if c.passages]
    for model in MODELS:
        if all(f"{model}|{k}|{c.q.text}" in cache for k in DEPTHS for c in todo):
            continue
        proc = serve(model)
        try:
            gen = {k: Generator(LlmClient(LlmConfig(f"http://127.0.0.1:{PORT}", model, timeout=60)), depth=k)
                   for k in DEPTHS}
            gen[DEPTHS[0]].generate("안녕하세요", todo[0].passages)  # warm up (first call loads the weights)
            for k in DEPTHS:
                for c in todo:
                    key = f"{model}|{k}|{c.q.text}"
                    if key in cache:
                        continue
                    g = gen[k].generate(c.q.text, c.passages)
                    cache[key] = {"text": g.text, "outcome": g.outcome, "ms": g.ms, "raw": g.raw}
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(model, "k", k, "done", flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    return cache


def evaluate(cases: list[Case], cache: dict[str, Any], tokenize: Any, model: str | None, k: int,
             on_reject: str, half: int) -> dict[str, Any]:
    """Metrics of one configuration on one half (model None: extractive)."""
    correct, wrong_ans, wrong_unans, recall, lengths, lat, unsupported, attempts = [], [], [], [], [], [], 0, 0
    for c in cases:
        if c.q.half != half or c.q.kind == "block":
            continue
        g = cache.get(f"{model}|{k}|{c.q.text}") if model is not None and c.passages else None
        if model is not None and c.passages and g is None:
            raise BenchError(f"no generation cached for {model} k={k}: {c.q.text}")
        if g is not None:
            attempts += 1
            from chatcore.generation import check

            unsupported += check(g["raw"], c.passages[:k]) == "unsupported_number" if g["raw"] else 0
        text = final_text(c, g, on_reject)
        if c.extract is not None:
            lat.append(c.kb_ms + (g["ms"] if g is not None else 0))
        if text is not None:
            lengths.append(len(text))
        if c.q.kind == "answerable":
            ok = text is not None and backed(text, c.sources, tokenize)
            correct.append(ok)
            wrong_ans.append(text is not None and not ok)
            recall.append(text is not None and fact_score(text, c.gold, tokenize) >= FACT_BAR)
        else:
            wrong_unans.append(text is not None)
    arr = lambda v: np.array(v, dtype=float)  # noqa: E731
    return {"correct": (float(np.mean(correct)), bootstrap(arr(correct))),
            "wrong": (float(np.mean(wrong_ans)), bootstrap(arr(wrong_ans))),
            "unans_answered": (float(np.mean(wrong_unans)), bootstrap(arr(wrong_unans))),
            "recall": float(np.mean(recall)) if recall else 0.0,
            "unsupported": unsupported / attempts if attempts else 0.0,
            "median_chars": statistics.median(lengths) if lengths else 0,
            "p50_ms": float(np.percentile(lat, 50)) if lat else 0.0, "p95_ms": float(np.percentile(lat, 95)) if lat else 0.0}


def paired(cases: list[Case], cache: dict[str, Any], tokenize: Any, model: str | None, k: int, on_reject: str,
           seed: int = 7, n: int = 4000) -> dict[str, Any]:
    """This configuration minus extractive, question by question over BOTH halves (answerable questions
    for correct/wrong, unanswerable for answered), with paired bootstrap 95% intervals."""
    def per_question(m: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        corr, wrong, unans, doc_chars = [], [], [], []
        for c in cases:
            if c.q.kind == "block":
                continue
            g = cache.get(f"{m}|{k}|{c.q.text}") if m is not None and c.passages else None
            text = final_text(c, g, on_reject if m is not None else "-")
            if c.q.kind == "answerable":
                ok = text is not None and backed(text, c.sources, tokenize)
                corr.append(ok)
                wrong.append(text is not None and not ok)
                if text is not None and c.passages and c.passages[0].kind == "doc":
                    doc_chars.append(len(text))
            else:
                unans.append(text is not None)
        return np.array(corr, float), np.array(wrong, float), np.array(unans, float), doc_chars

    base, cur = per_question(None), per_question(model)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for i, name in ((0, "correct"), (1, "wrong")):
        d = cur[i] - base[i]
        boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
        out[name] = (float(d.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))))
    out["unans"] = (float(cur[2].mean()), float(base[2].mean()))
    out["doc_chars"] = (statistics.median(cur[3]) if cur[3] else 0, statistics.median(base[3]) if base[3] else 0)
    return out


# The configuration the engine ships (깜몬's decision from the paired comparison below; see report())
RECOMMENDED = ("mlx-community/Qwen3-8B-4bit", 1, "fallback")


def passes(r: dict[str, Any], base: dict[str, Any]) -> bool:
    return (r["correct"][0] >= base["correct"][0] - CRITERIA["fact_drop"] and r["wrong"][0] <= base["wrong"][0]
            and r["unans_answered"][0] <= base["unans_answered"][0] and r["p95_ms"] <= CRITERIA["p95_ms"]
            and r["median_chars"] <= CRITERIA["median_chars"])


def main() -> int:  # pragma: no cover - real models
    cases, kiwi = build_cases()
    print("answered by the knowledge base:", sum(bool(c.passages) for c in cases), flush=True)
    cache = generate_all(cases)
    tokenize = kiwi.tokenize
    configs: list[tuple[str | None, int, str]] = [(None, 1, "-")] + [(m, k, r) for m in MODELS for k in DEPTHS
                                                                  for r in ("extract", "fallback")]
    dev = {cfg: evaluate(cases, cache, tokenize, *cfg, half=0) for cfg in configs}
    test = {cfg: evaluate(cases, cache, tokenize, *cfg, half=1) for cfg in configs}
    base_dev = dev[configs[0]]
    chosen = next((cfg for cfg in configs[1:] if passes(dev[cfg], base_dev)), None)  # smallest model, k, first
    pairs = {cfg: paired(cases, cache, tokenize, *cfg) for cfg in configs[1:]}
    REPORT.write_text(report(configs, dev, test, chosen, cases, pairs), encoding="utf-8")
    print("chosen:", chosen, flush=True)
    return 0


def report(configs: Sequence[tuple[str | None, int, str]], dev: dict[Any, Any], test: dict[Any, Any],
           chosen: tuple[str | None, int, str] | None, cases: Sequence[Case],
           pairs: dict[Any, Any]) -> str:  # pragma: no cover
    def ci(v: tuple[float, tuple[float, float]]) -> str:
        return f"{v[0]:.3f} [{v[1][0]:.3f}–{v[1][1]:.3f}]"

    lines = ["# 생성형 RAG 측정 (E-F)", "", "`uv run python bench/rag_eval.py`가 만든 파일. 손으로 고치지 말 것.", "",
             f"- 질문·지식베이스·검색과 판단은 `docs/benchmarks/kb.md`와 같음. 지식베이스가 답한 질문 {sum(bool(c.passages) for c in cases)}개에만 생성.",
             "- 표는 확인용 절반(설정은 개발용 절반에서 고름), [괄호]는 부트스트랩 95% 구간.",
             "- **맞게 답함** = 답이 사실(명사·숫자·내용 있는 용언 어간)을 하나 이상 말하고, 그 사실의 60% 이상과 숫자 전부가 맞는 출처(정답 FAQ 항목, 정답 증거가 든 조각)에 있음.",
             "- **재현율(참고)** = 정답 문장의 사실 60% 이상이 답에 있음 — 처음 쓴 지표. 짧고 맞는 답(\"나쁨입니다.\")을 틀렸다고 보고 정답 문장을 그대로 내는 추출형만 유리해서 고르는 데 쓰지 않음.",
             "- 생성을 버렸을 때: extract = 추출형 답을 대신 보여 줌, fallback = 답하지 않음(폴백).", "",
             "| 설정 | 맞게 답함 | 틀리게 답함 | 답 없는 질문에 답함 | 재현율(참고) | 근거 없는 숫자(검사 전) | 답 길이 중앙값 | p50 ms | p95 ms | 기준 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    base_dev = dev[configs[0]]
    for cfg in configs:
        model, k, rej = cfg
        name = "추출형(지금)" if model is None else f"{model.split('/')[-1]} · k={k} · {rej}"
        t = test[cfg]
        mark = "—" if model is None else ("✅" if passes(dev[cfg], base_dev) else "❌") + (" ← 채택" if cfg == chosen else "")
        lines.append(f"| {name} | {ci(t['correct'])} | {ci(t['wrong'])} | {ci(t['unans_answered'])} | {t['recall']:.3f} | {t['unsupported']:.3f} "
                     f"| {t['median_chars']:.0f} | {t['p50_ms']:.0f} | {t['p95_ms']:.0f} | {mark} |")
    lines += ["", f"기준(개발용 절반): 맞게 답함이 추출형보다 {CRITERIA['fact_drop']} 넘게 낮지 않음, 틀리게 답함·답 없는 질문에 답함이 추출형 이하, "
              f"p95 ≤ {CRITERIA['p95_ms']:.0f}ms, 답 길이 중앙값 ≤ {CRITERIA['median_chars']}자. 통과한 것 중 가장 작은 모델·적은 자료·단순한 처리를 고른다.",
              "", "사후 검사 뒤에는 근거 없는 숫자가 나가지 않는다(검사에서 걸러져 위 '버렸을 때' 처리로 감).", "",
              "## 추출형과 같은 질문끼리 비교 (두 절반 합침, 쌍 부트스트랩 95% 구간)", "",
              "| 설정 | 맞게 답함 차이 | 틀리게 답함 차이 | 답 없는 질문에 답함 (생성 / 추출형) | 문서 답 길이 중앙값 (생성 / 추출형) |",
              "|---|---|---|---|---|"]
    for cfg, p in pairs.items():
        model, k, rej = cfg
        lines.append(f"| {str(model).split('/')[-1]} · k={k} · {rej}{' ← 채택' if cfg == RECOMMENDED else ''} "
                     f"| {p['correct'][0]:+.3f} [{p['correct'][1][0]:+.3f}, {p['correct'][1][1]:+.3f}] "
                     f"| {p['wrong'][0]:+.3f} [{p['wrong'][1][0]:+.3f}, {p['wrong'][1][1]:+.3f}] "
                     f"| {p['unans'][0]:.3f} / {p['unans'][1]:.3f} | {p['doc_chars'][0]:.0f} / {p['doc_chars'][1]:.0f} |")
    rec = pairs[RECOMMENDED]
    lines += ["", "## 결정 (깜몬, 2026-11-12)",
              f"- 미리 정한 기준(점추정)은 개발용 절반에서 모든 설정이 넘지 못했다. 가장 가까운 {RECOMMENDED[0].split('/')[-1]} · k={RECOMMENDED[1]}은 "
              "틀리게 답함이 추출형보다 1문항(88문항 중) 많아 떨어졌고, 확인용 절반에서는 넘었다.",
              f"- 같은 질문끼리 비교하면 이 설정은 맞게 답함 {rec['correct'][0]:+.3f}, 틀리게 답함 {rec['wrong'][0]:+.3f}로 구간이 모두 0을 포함해 "
              f"추출형과 구분되지 않고, 답 없는 질문에 답하는 비율은 {rec['unans'][1]:.3f} → {rec['unans'][0]:.3f}로 줄며(모델이 NO_ANSWER로 거절), "
              f"문서에서 찾은 답은 {rec['doc_chars'][1]:.0f}자 조각 대신 {rec['doc_chars'][0]:.0f}자 답이 된다. 4B는 맞게 답함이 구분될 만큼 낮고, 14B는 낫지 않으면서 느리다.",
              f"- 그래서 엔진 기본값은 {RECOMMENDED[0]} · 자료 {RECOMMENDED[1]}개 · 생성을 버리면 폴백(`chatcore.generation.GENERATION_DEPTH`, "
              "`chatcore.knowledge.ON_REJECT`)이고, 생성은 봇마다 지식베이스 설정으로 켠다(기본 꺼짐 — 켜지 않은 봇은 그대로)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
