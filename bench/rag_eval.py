"""Generative RAG measurement (engine stage E-F; docs/engine-core-plan.md "빌더 단계 E-F").

Same knowledge base and 300 questions as bench/kb_eval.py, and the engine's own path: the sample
bot's NLU decides the rule, `Knowledge.consult` (embedding ranking + cross-encoder judge) decides
whether the knowledge base answers, and `chatcore.generation.Generator` writes the answer through an
OpenAI-compatible server (`mlx_lm.server`, started here for each model).

Configurations: extractive (today) / generative per model x passages given (k) x what happens when a
generated answer is rejected ("extract": show the extractive answer, "fallback": answer nothing; when the
LLM does not answer — a timeout at the engine's LLM_TIMEOUT, an error — both show the extractive answer,
as the engine does). Exact title matches are never generated (the engine shows what the button offered).
Both sides are compared without the engine's "출처: …" line.
Metrics on the test half (the dev half chooses), bootstrap 95% intervals:
  correct  = answerable question, answered, and what the answer says is backed by a right source: at
             least one fact (nouns, numbers, content verb/adjective stems; Kiwi), >= 60% of its facts found in a right source (the
             gold FAQ entry, or the chunk holding the gold evidence) and every number in that source
  wrong    = answered but not correct (answerable), or answered at all (unanswerable)
  recall   = the first metric tried: share of the gold text's facts found in the answer (>= 0.6). It
             marks a short, right answer ("나쁨입니다.") wrong because it drops the rest of the gold
             sentence, and an extractive answer (the gold text itself) right — reported, not used to choose
  unsupported = generated answers with a number not in the passages, BEFORE the check (model quality)
  length   = median characters of answers given; latency = knowledge-base turn p50/p95 incl. generation
             (a generation slower than LLM_TIMEOUT counts as the timeout)
Known limit of `correct`: a negation in the answer must be backed by one in the source, but a source
that negates something the answer states positively can still back it (one sentence of a long chunk).
No LLM grades another LLM: every metric is counted on strings.

Usage: uv run python bench/rag_eval.py   (needs `uv tool install mlx-lm` and the models downloaded)
Writes docs/benchmarks/rag.md; generations are cached in .claude/tmp/rag_cache.json by what was sent
(model, max_tokens, the messages), and judged again on every run (the check may change).
"""

from __future__ import annotations

import hashlib
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
NEGATION = "<부정>"  # a fact of its own (않, 안, 못, 없, 아니, 불가)
NEGATIONS = {("않", "VX"), ("안", "MAG"), ("못", "MAG"), ("못하", "VX"), ("없", "VA"), ("아니", "VCN"), ("불가", "NNG")}


def fact_tokens(text: str, tokenize: Any) -> set[str]:
    """Numbers, nouns and content verb/adjective stems ("나쁘" of 나쁨) of a text, and NEGATION when it
    negates; generic stems (하다, 되다, 있다, 그렇다 ...) say nothing checkable and are left out."""
    from chatcore.generation import numbers

    tokens = tokenize(text)
    words = {t.form.lower() for t in tokens
             if t.tag.startswith(("NNG", "NNP", "SL", "VV", "VA", "XR")) and t.form not in GENERIC_STEMS}
    if any((t.form, t.tag) in NEGATIONS for t in tokens):
        words.add(NEGATION)
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
    exact: bool = False  # an exact title match: the engine shows the entry, never a generation


def backed(answer: str, sources: Sequence[str], tokenize: Any) -> bool:
    """The answer states something (a fact) and what it states is in one right source: >= FACT_BAR of
    its facts and all of its numbers. A negation is one fact (a short flip — "환불되지 않아요" for
    "환불돼요" — fails the share); see unbacked_negation for the rest."""
    from chatcore.generation import numbers

    facts = fact_tokens(answer, tokenize)
    if not facts:
        return False
    for src in sources:
        have = fact_tokens(src, tokenize)
        if len(facts & have) / len(facts) >= FACT_BAR and numbers(answer) <= numbers(src):
            return True
    return False


def unbacked_negation(answer: str, sources: Sequence[str], tokenize: Any) -> bool:
    """The answer negates and no right source does: a flip, or a "no" the model inferred ("km/h는
    지원하지 않습니다" from "m/s로 표시합니다"). Reported, not part of `correct`: reading the cases
    shows both, and only a person can tell them apart."""
    return NEGATION in fact_tokens(answer, tokenize) and not any(NEGATION in fact_tokens(s, tokenize) for s in sources)


def cache_key(model: str, k: int, case: Case) -> str:
    """What was sent: the same key means the same request (prompt, passages, question, limits)."""
    from chatcore.generation import MAX_TOKENS, build_messages

    sent = json.dumps([model, MAX_TOKENS, build_messages(case.q.text, case.passages[:k])], ensure_ascii=False)
    return hashlib.sha256(sent.encode("utf-8")).hexdigest()


def judged(stored: dict[str, Any], passages: Sequence[Any], timeout_ms: float) -> dict[str, Any]:
    """A cached generation ({"raw", "ms"}) judged as the engine would now: slower than the engine's
    timeout -> "timeout", else chatcore.generation.check."""
    from chatcore.generation import check

    if stored["ms"] > timeout_ms:
        return {"text": None, "outcome": "timeout", "ms": timeout_ms}
    outcome = check(stored["raw"], passages)
    return {"text": stored["raw"].strip() if outcome == "generated" else None, "outcome": outcome, "ms": stored["ms"]}


def final_text(case: Case, generated: dict[str, Any] | None, on_reject: str) -> str | None:
    """What the bot says from the knowledge base under a configuration (None: it does not answer)."""
    from chatcore.knowledge import answers_nothing

    if case.extract is None:
        return None
    if generated is None:
        return case.extract
    if generated["text"] is not None:
        return str(generated["text"])
    return None if answers_nothing(generated["outcome"], on_reject) else case.extract


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
        extract = None
        if answer is not None:  # compared without the "출처: …" line (a generated answer gets one too)
            extract = answer.text.removesuffix(f"\n\n출처: {passages[0].source}") if passages[0].kind == "doc" else answer.text
        cases.append(Case(q, n.rule, passages, extract, gold, kb_ms, sources, bool(result and result.exact)))
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


def migrate(cache: dict[str, Any], cases: Sequence[Case]) -> dict[str, Any]:
    """Entries of the first cache ("model|k|question", generated with the prompt of commit 7169529,
    unchanged since) under the request key; their LLM failures are dropped to be tried again."""
    by_question = {c.q.text: c for c in cases if c.passages}
    out = {key: v for key, v in cache.items() if "|" not in key}
    for key, v in cache.items():
        model, _, rest = key.partition("|")
        if not rest or v.get("outcome") in ("timeout", "error"):
            continue
        k, _, question = rest.partition("|")
        if question in by_question:
            out.setdefault(cache_key(model, int(k), by_question[question]), {"raw": v["raw"], "ms": v["ms"]})
    return out


def generate_all(cases: list[Case]) -> dict[str, Any]:  # pragma: no cover - real models
    from chatcore.generation import Generator, LlmClient, LlmConfig

    cache: dict[str, Any] = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    cache = migrate(cache, cases)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    todo = [c for c in cases if c.passages and not c.exact]
    for model in MODELS:
        if all(cache_key(model, k, c) in cache for k in DEPTHS for c in todo):
            continue
        proc = serve(model)
        try:
            gen = {k: Generator(LlmClient(LlmConfig(f"http://127.0.0.1:{PORT}", model, timeout=60)), depth=k)
                   for k in DEPTHS}
            gen[DEPTHS[0]].generate("안녕하세요", todo[0].passages)  # warm up (first call loads the weights)
            for k in DEPTHS:
                for c in todo:
                    key = cache_key(model, k, c)
                    if key in cache:
                        continue
                    g = gen[k].generate(c.q.text, c.passages)
                    if g.outcome not in ("timeout", "error"):  # a failure is tried again next run
                        cache[key] = {"raw": g.raw, "ms": g.ms}
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(model, "k", k, "done", flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    return cache


def generation(c: Case, cache: dict[str, Any], model: str | None, k: int,
               timeout_ms: float) -> dict[str, Any] | None:
    """The judged generation the engine would use for a case, or None (extractive / no generation)."""
    if model is None or not c.passages or c.exact:
        return None
    stored = cache.get(cache_key(model, k, c))
    if stored is None:
        raise BenchError(f"no generation cached for {model} k={k}: {c.q.text}")
    return judged(stored, c.passages[:k], timeout_ms)


def has_unsupported_number(raw: str, passages: Sequence[Any]) -> bool:
    from chatcore.generation import numbers, passage_text

    return bool(numbers(raw) - set().union(*(numbers(passage_text(e)) for e in passages)))


def evaluate(cases: list[Case], cache: dict[str, Any], tokenize: Any, model: str | None, k: int,
             on_reject: str, half: int, timeout_ms: float) -> dict[str, Any]:
    """Metrics of one configuration on one half (model None: extractive)."""
    correct, wrong_ans, wrong_unans, recall, lengths, lat, unsupported, attempts = [], [], [], [], [], [], 0, 0
    negated, timeouts = [], 0
    for c in cases:
        if c.q.half != half or c.q.kind == "block":
            continue
        g = generation(c, cache, model, k, timeout_ms)
        if g is not None:
            attempts += 1
            unsupported += has_unsupported_number(cache[cache_key(str(model), k, c)]["raw"], c.passages[:k])
            timeouts += g["outcome"] == "timeout"
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
            if text is not None:
                negated.append(unbacked_negation(text, c.sources, tokenize))
        else:
            wrong_unans.append(text is not None)
    arr = lambda v: np.array(v, dtype=float)  # noqa: E731
    return {"correct": (float(np.mean(correct)), bootstrap(arr(correct))),
            "wrong": (float(np.mean(wrong_ans)), bootstrap(arr(wrong_ans))),
            "unans_answered": (float(np.mean(wrong_unans)), bootstrap(arr(wrong_unans))),
            "recall": float(np.mean(recall)) if recall else 0.0,
            "unsupported": unsupported / attempts if attempts else 0.0,
            "timeouts": timeouts / attempts if attempts else 0.0,
            "negated": float(np.mean(negated)) if negated else 0.0,
            "median_chars": statistics.median(lengths) if lengths else 0,
            "p50_ms": float(np.percentile(lat, 50)) if lat else 0.0, "p95_ms": float(np.percentile(lat, 95)) if lat else 0.0}


def paired(cases: list[Case], cache: dict[str, Any], tokenize: Any, model: str | None, k: int, on_reject: str,
           half: int, timeout_ms: float, seed: int = 7, n: int = 4000) -> dict[str, Any]:
    """This configuration minus extractive, question by question on one half (answerable questions for
    correct/wrong, unanswerable for answered), with paired bootstrap 95% intervals."""
    def per_question(m: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        corr, wrong, unans, doc_chars = [], [], [], []
        for c in cases:
            if c.q.kind == "block" or c.q.half != half:
                continue
            g = generation(c, cache, m, k, timeout_ms)
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


def disagreements(cases: Sequence[Case], cache: dict[str, Any], tokenize: Any, model: str | None, k: int,
                  on_reject: str, timeout_ms: float) -> list[dict[str, Any]]:
    """Questions (both halves) where the configuration and extractive are scored differently, or its
    answer negates without a negating source: for a person to read."""
    rows = []
    for c in cases:
        if c.q.kind == "block" or c.extract is None:
            continue
        g = generation(c, cache, model, k, timeout_ms)
        text = final_text(c, g, on_reject)
        if c.q.kind == "answerable":
            mine, theirs = (text is not None and backed(text, c.sources, tokenize)), backed(c.extract, c.sources, tokenize)
            negates = text is not None and unbacked_negation(text, c.sources, tokenize)
        else:
            mine, theirs, negates = text is None, False, False  # "right" = not answering
        if mine != theirs or negates:
            rows.append({"half": c.q.half, "kind": c.q.kind, "question": c.q.text, "mine": mine, "theirs": theirs,
                         "negates": negates, "outcome": g["outcome"] if g else "-", "text": text, "extract": c.extract})
    return rows


def passes(r: dict[str, Any], base: dict[str, Any]) -> bool:
    return not failed(r, base)


def failed(r: dict[str, Any], base: dict[str, Any]) -> list[str]:
    """The criteria (docs/engine-core-plan.md E-F) a configuration misses against extractive."""
    out = []
    if r["correct"][0] < base["correct"][0] - CRITERIA["fact_drop"]:
        out.append("맞게 답함")
    if r["wrong"][0] > base["wrong"][0]:
        out.append("틀리게 답함")
    if r["unans_answered"][0] > base["unans_answered"][0]:
        out.append("답 없는 질문에 답함")
    if r["p95_ms"] > CRITERIA["p95_ms"]:
        out.append("p95")
    if r["median_chars"] > CRITERIA["median_chars"]:
        out.append("길이")
    return out


def choose(configs: Sequence[Any], dev: dict[Any, Any]) -> tuple[Any, bool]:
    """The configuration to ship, from the dev half only: the first that passes every criterion
    (smallest model, fewest passages, "extract" before "fallback"), else the one missing the fewest,
    then the highest correct minus wrong. Returns (config, passed)."""
    base = dev[configs[0]]
    for cfg in configs[1:]:
        if passes(dev[cfg], base):
            return cfg, True
    best = min(configs[1:], key=lambda cfg: (len(failed(dev[cfg], base)), -(dev[cfg]["correct"][0] - dev[cfg]["wrong"][0])))
    return best, False


def main() -> int:  # pragma: no cover - real models
    from chatcore.generation import GENERATION_DEPTH, LLM_TIMEOUT
    from chatcore.knowledge import ON_REJECT

    cases, kiwi = build_cases()
    print("answered by the knowledge base:", sum(bool(c.passages) for c in cases), flush=True)
    cache = generate_all(cases)
    tokenize, timeout_ms = kiwi.tokenize, LLM_TIMEOUT * 1000
    configs: list[tuple[str | None, int, str]] = [(None, 1, "-")] + [(m, k, r) for m in MODELS for k in DEPTHS
                                                                  for r in ("extract", "fallback")]
    dev = {cfg: evaluate(cases, cache, tokenize, *cfg, half=0, timeout_ms=timeout_ms) for cfg in configs}
    chosen, passed = choose(configs, dev)  # decided before the test half is computed
    test = {cfg: evaluate(cases, cache, tokenize, *cfg, half=1, timeout_ms=timeout_ms) for cfg in configs}
    pairs = {h: paired(cases, cache, tokenize, *chosen, half=h, timeout_ms=timeout_ms) for h in (0, 1)}
    shipped = (chosen[1], chosen[2]) == (GENERATION_DEPTH, ON_REJECT)
    rows = disagreements(cases, cache, tokenize, *chosen, timeout_ms=timeout_ms)
    REPORT.write_text(report(configs, dev, test, chosen, passed, pairs, cases, shipped, rows), encoding="utf-8")
    print("chosen:", chosen, "passed" if passed else "closest", "shipped" if shipped else "NOT THE ENGINE'S DEFAULTS", flush=True)
    return 0 if shipped else 1


def report(configs: Sequence[tuple[str | None, int, str]], dev: dict[Any, Any], test: dict[Any, Any],
           chosen: tuple[str | None, int, str], passed: bool, pairs: dict[int, Any], cases: Sequence[Case],
           shipped: bool, rows: Sequence[dict[str, Any]]) -> str:  # pragma: no cover
    from chatcore.generation import LLM_TIMEOUT

    def ci(v: tuple[float, tuple[float, float]]) -> str:
        return f"{v[0]:.3f} [{v[1][0]:.3f}–{v[1][1]:.3f}]"

    def name(cfg: tuple[str | None, int, str]) -> str:
        return "추출형(지금)" if cfg[0] is None else f"{cfg[0].split('/')[-1]} · k={cfg[1]} · {cfg[2]}"

    base_dev = dev[configs[0]]
    lines = ["# 생성형 RAG 측정 (E-F)", "", "`uv run python bench/rag_eval.py`가 만든 파일. 손으로 고치지 말 것.", "",
             f"- 질문·지식베이스·검색과 판단은 `docs/benchmarks/kb.md`와 같음. 지식베이스가 답한 질문 {sum(bool(c.passages) for c in cases)}개 중 "
             f"제목이 정확히 같은 {sum(c.exact for c in cases)}개를 뺀 질문에만 생성(엔진도 같음).",
             f"- 엔진과 같은 조건: 생성이 {LLM_TIMEOUT:g}초(`LLM_TIMEOUT`)를 넘으면 시간 초과로 보고 자료를 보여 줌. 양쪽 모두 \"출처: …\" 줄을 빼고 비교.",
             "- 표는 확인용 절반, [괄호]는 부트스트랩 95% 구간. 설정은 개발용 절반만 보고 고른다(확인용 절반은 고른 뒤에 계산).",
             "- **맞게 답함** = 답이 사실(명사·숫자·내용 있는 용언 어간·부정)을 하나 이상 말하고, 그 사실의 60% 이상과 숫자 전부가 맞는 출처(정답 FAQ 항목, 정답 증거가 든 조각)에 있음. "
             "문자열로 세므로 바꿔 말한 맞는 답을 틀렸다고 볼 수 있다 — 아래 '엇갈린 질문'에 채택 설정의 사례를 모두 싣는다.",
             "- **근거 없는 부정** = 답한 것 중 답에 부정(않·안·못·없·아니·불가)이 있는데 맞는 출처 어디에도 부정이 없는 비율. 뒤집힌 답과 모델이 추론한 \"아니요\"가 섞여 있어 사람이 읽어야 구분된다.",
             "- **재현율(참고)** = 정답 문장의 사실 60% 이상이 답에 있음 — 처음 쓴 지표. 짧고 맞는 답(\"나쁨입니다.\")을 틀렸다고 보고 정답 문장을 그대로 내는 추출형만 유리해서 고르는 데 쓰지 않음.",
             "- 생성을 버렸을 때(NO_ANSWER, 근거 없는 숫자, 너무 김, 빈 답): extract = 추출형 답을 보여 줌, fallback = 답하지 않음(폴백). "
             "LLM이 답하지 못했을 때(시간 초과, 오류)는 둘 다 추출형 답을 보여 줌.", "",
             "| 설정 | 맞게 답함 | 틀리게 답함 | 답 없는 질문에 답함 | 재현율(참고) | 근거 없는 부정 | 근거 없는 숫자(검사 전) | 시간 초과 | 답 길이 중앙값 | p50 ms | p95 ms | 개발용 기준 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in configs:
        t = test[cfg]
        miss = failed(dev[cfg], base_dev)
        mark = "—" if cfg[0] is None else ("✅" if not miss else "❌ " + "·".join(miss)) + (" ← 채택" if cfg == chosen else "")
        lines.append(f"| {name(cfg)} | {ci(t['correct'])} | {ci(t['wrong'])} | {ci(t['unans_answered'])} | {t['recall']:.3f} | {t['negated']:.3f} "
                     f"| {t['unsupported']:.3f} | {t['timeouts']:.3f} "
                     f"| {t['median_chars']:.0f} | {t['p50_ms']:.0f} | {t['p95_ms']:.0f} | {mark} |")
    lines += ["", f"기준(개발용 절반, 점추정): 맞게 답함이 추출형보다 {CRITERIA['fact_drop']} 넘게 낮지 않음, 틀리게 답함·답 없는 질문에 답함이 추출형 이하, "
              f"p95 ≤ {CRITERIA['p95_ms']:.0f}ms, 답 길이 중앙값 ≤ {CRITERIA['median_chars']}자. 통과한 것 중 가장 작은 모델·적은 자료·extract 먼저. "
              "통과한 것이 없으면 놓친 기준이 가장 적고, 그다음 (맞게 답함 − 틀리게 답함)이 큰 것.",
              "", "사후 검사 뒤에는 근거 없는 숫자가 나가지 않는다(검사에서 걸러져 위 '버렸을 때' 처리로 감).", "",
              f"## 채택한 설정과 추출형을 같은 질문끼리 비교 ({name(chosen)}, 쌍 부트스트랩 95% 구간)", "",
              "| 절반 | 맞게 답함 차이 | 틀리게 답함 차이 | 답 없는 질문에 답함 (생성 / 추출형) | 문서 답 길이 중앙값 (생성 / 추출형) |",
              "|---|---|---|---|---|"]
    for h, p in pairs.items():
        lines.append(f"| {'개발용' if h == 0 else '확인용'} | {p['correct'][0]:+.3f} [{p['correct'][1][0]:+.3f}, {p['correct'][1][1]:+.3f}] "
                     f"| {p['wrong'][0]:+.3f} [{p['wrong'][1][0]:+.3f}, {p['wrong'][1][1]:+.3f}] "
                     f"| {p['unans'][0]:.3f} / {p['unans'][1]:.3f} | {p['doc_chars'][0]:.0f} / {p['doc_chars'][1]:.0f} |")
    def cell(text: str | None) -> str:
        return "(답하지 않음)" if text is None else " ".join(text.split()).replace("|", "\\|")[:150]

    lines += ["", f"## 엇갈린 질문 ({name(chosen)}, 두 절반, {len(rows)}개)", "",
              "점수가 추출형과 다르거나 근거 없는 부정이 있는 질문 전부. O = 맞게 답함(답 없는 질문은 답하지 않음), ¬ = 근거 없는 부정.", "",
              "| 절반 | 질문 | 생성 | 추출형 | 처리 | 생성 답 | 추출형 답 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {'개발' if r['half'] == 0 else '확인'} | {cell(r['question'])}{' (답 없음)' if r['kind'] != 'answerable' else ''} "
                     f"| {'O' if r['mine'] else 'X'}{' ¬' if r['negates'] else ''} | {'O' if r['theirs'] else 'X'} | {r['outcome']} "
                     f"| {cell(r['text'])} | {cell(r['extract'])} |")
    d = pairs[0]
    lines += ["", "## 결정",
              f"- 개발용 절반에서 {'모든 기준을 통과한' if passed else '기준을 모두 통과한 설정이 없어, 놓친 기준이 가장 적은'} {name(chosen)}을 고른다.",
              f"- 같은 질문끼리 보면(개발용) 맞게 답함 차이의 구간 하한이 {d['correct'][1][0]:+.3f}이다: 추출형보다 최대 {abs(min(0.0, d['correct'][1][0])):.3f} 낮을 가능성을 "
              f"이 표본으로는 배제하지 못한다(기준 {CRITERIA['fact_drop']}보다 넓은 구간 — 질문 수의 한계). 틀리게 답함 차이 구간은 [{d['wrong'][1][0]:+.3f}, {d['wrong'][1][1]:+.3f}].",
              f"- 엔진 기본값: 자료 {chosen[1]}개(`GENERATION_DEPTH`), 생성을 버리면 {chosen[2]}(`ON_REJECT`) — "
              + ("이 결과와 같다." if shipped else "**이 결과와 다르다: 고칠 것.**")
              + " 모델은 운영자가 `--llm-model`로 정하고, 생성은 봇마다 지식베이스 설정으로 켠다(기본 꺼짐)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
