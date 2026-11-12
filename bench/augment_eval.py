"""E-H measurement: do LLM-written block examples and FAQ similar questions help with typos and wording?

docs/engine-core-plan.md "빌더 단계 E-H". The same LLM and screening as the builder's button
(`chatcore.augment`), every `ok` proposal added (a person accepting them all), then the engine's own
path: blocks through deploy training and the interpreter on a fresh session, the knowledge base
through `Knowledge.consult` with the cross-encoder judge.

Test sentences: bench/data/test_independent.yaml (blocks) and bench/data/kb_eval.yaml (knowledge
base), each as written and with ONE rule-made typo (`noise`, seeded). The typos come from Hangul jamo
rules, not from the LLM, so the evaluation does not reward the generator for copying itself.

Usage: uv run python bench/augment_eval.py   (an OpenAI-compatible LLM on 127.0.0.1:8800, or mlx-lm
installed to start Qwen3-8B there). Writes docs/benchmarks/augment.md; proposals are cached in
.claude/tmp/augment_cache.json by what was sent.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.kb_eval import QUESTIONS, BenchError, Item, Question, load_kb, load_questions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs/benchmarks/augment.md"
CACHE = ROOT / ".claude/tmp/augment_cache.json"
SCRATCH = ROOT / ".claude/tmp/augment_eval"
TEST = ROOT / "bench/data/test_independent.yaml"
MODELS = ("mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-14B-4bit")  # the server loads the one asked for
LLM_URL = "http://127.0.0.1:8800"
PER_INTENT, PER_FAQ = 20, 10
SEED = 7
CRITERIA = {"drop": 0.02}  # clean performance, out-of-scope rejection: not worse by more than this
PROBES = (("intent", "오늘 날시 알려줘", "weather"), ("intent", "고마어", "thanks"),
          ("kb", "미새먼지 나쁨은 몇부터예요?", "faq"), ("kb", "미세 먼지 안좋음 기준이 뭐임", "faq"))


# ---------------------------------------------------------------- typos (pure; never the LLM)

_BASE, _JUNG, _JONG = 0xAC00, 21, 28
_VOWEL_SWAP = {1: 5, 5: 1, 3: 7, 7: 3}  # ㅐ↔ㅔ, ㅒ↔ㅖ
_TENSE_CHO = {1: 0, 4: 3, 8: 7, 10: 9, 13: 12}  # ㄲ→ㄱ ㄸ→ㄷ ㅃ→ㅂ ㅆ→ㅅ ㅉ→ㅈ
_TENSE_JONG = {2: 1, 20: 19}  # ㄲ→ㄱ ㅆ→ㅅ (final)
KINDS = ("vowel", "tense", "final", "space", "yo")


def _split(c: str) -> tuple[int, int, int] | None:
    code = ord(c) - _BASE
    if not 0 <= code < 11172:
        return None
    return code // (_JUNG * _JONG), (code % (_JUNG * _JONG)) // _JONG, code % _JONG


def _join(cho: int, jung: int, jong: int) -> str:
    return chr(_BASE + (cho * _JUNG + jung) * _JONG + jong)


def _edits(text: str) -> dict[str, list[str]]:
    """Every sentence one typo away, by kind (none for a sentence without Hangul)."""
    if not any(_split(c) is not None for c in text):
        return {}
    out: dict[str, list[str]] = {k: [] for k in KINDS}
    for i, c in enumerate(text):
        parts = _split(c)
        if parts is None:
            continue
        cho, jung, jong = parts
        put = lambda s, i=i: text[:i] + s + text[i + 1:]  # noqa: E731
        if jung in _VOWEL_SWAP:
            out["vowel"].append(put(_join(cho, _VOWEL_SWAP[jung], jong)))
        if cho in _TENSE_CHO:
            out["tense"].append(put(_join(_TENSE_CHO[cho], jung, jong)))
        if jong in _TENSE_JONG:
            out["tense"].append(put(_join(cho, jung, _TENSE_JONG[jong])))
        if jong:
            out["final"].append(put(_join(cho, jung, 0)))
        if i + 1 < len(text) and _split(text[i + 1]) is not None:
            out["space"].append(text[:i + 1] + " " + text[i + 1:])
    out["space"] += [text[:i] + text[i + 1:] for i, c in enumerate(text) if c == " " and 0 < i < len(text) - 1]
    for i, c in enumerate(text):
        if c == "요" and (i + 1 == len(text) or not ("가" <= text[i + 1] <= "힣")):
            out["yo"].append(text[:i] + "여" + text[i + 1:])
    return {k: v for k, v in out.items() if v}


def noise(text: str, seed: int = SEED) -> tuple[str, str]:
    """(the sentence with one typo, its kind): a kind chosen among those that apply, then a place —
    both from a generator seeded by the sentence, so every run makes the same typos. A sentence with
    no Hangul to change comes back unchanged, kind "none"."""
    edits = _edits(text)
    if not edits:
        return text, "none"
    rng = random.Random(f"{seed}:{text}")
    kind = rng.choice(sorted(edits))
    return rng.choice(edits[kind]), kind


# ---------------------------------------------------------------- proposals (cached)


def cache_key(model: str, messages: list[dict[str, str]]) -> str:
    from chatcore import augment

    sent = json.dumps([model, augment.TEMPERATURE, augment.MAX_TOKENS, messages], ensure_ascii=False)
    return hashlib.sha256(sent.encode("utf-8")).hexdigest()


def proposals(model: str, tasks: list[list[dict[str, str]]], cache: dict[str, Any],
              ask: Callable[[list[dict[str, str]]], list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """Every task's sentences, each task asked once and kept (sampling at temperature 0.7 varies)."""
    out: list[tuple[str, str]] = []
    for messages in tasks:
        key = cache_key(model, messages)
        if key not in cache:
            cache[key] = [list(x) for x in ask(messages)]
        out += [(str(t), str(k)) for t, k in cache[key]]
    return out


def augmented_nlu(nlu: dict[str, Any], added: dict[str, list[str]]) -> dict[str, Any]:
    """The NLU document with `added` examples appended to their intents (a copy)."""
    out = json.loads(json.dumps(nlu))
    for intent, texts in added.items():
        out["intents"][intent]["examples"] = [*out["intents"][intent]["examples"], *texts]
    return out


def augmented_faq(faq: Sequence[dict[str, Any]], added: dict[str, list[str]]) -> list[dict[str, Any]]:
    return [{**f, "similar": [*f.get("similar", []), *added.get(f["id"], [])]} for f in faq]


# ---------------------------------------------------------------- scoring (pure)


def kb_correct(q: Question, answered: Item | None) -> bool:
    return q.gold is not None and answered is not None and q.gold.hit(answered)


def paired(a: Sequence[bool], b: Sequence[bool], n: int = 4000, seed: int = SEED) -> tuple[float, tuple[float, float]]:
    """mean(b - a) and its paired bootstrap 95% interval (the same sentences on both sides)."""
    if len(a) != len(b) or not a:
        raise BenchError("paired samples must be the same, non-empty sentences")
    d = np.asarray(b, float) - np.asarray(a, float)
    boots = d[np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


@dataclass(frozen=True)
class Sentence:
    text: str
    gold: str  # an intent, or "out_of_scope:hard" / "out_of_scope:easy"


def load_block_test(path: Path = TEST) -> list[Sentence]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = [Sentence(t, intent) for intent, texts in raw["in_scope"].items() for t in texts]
    out += [Sentence(t, f"out_of_scope:{level}") for level, texts in raw["out_of_scope"].items() for t in texts]
    if not out:
        raise BenchError(f"{path} has no sentences")
    return out


def block_scores(golds: Sequence[str], decisions: Sequence[tuple[str, str | None]]) -> dict[str, list[bool]]:
    """Per sentence: in scope -> the interpreter started the gold intent; out of scope -> it fell back."""
    out: dict[str, list[bool]] = {"in_scope": [], "oos_hard": [], "oos_easy": []}
    for gold, (rule, top) in zip(golds, decisions, strict=True):
        if gold.startswith("out_of_scope:"):
            out["oos_" + gold.split(":")[1]].append(rule == "fallback")
        else:
            out["in_scope"].append(rule == "intent" and top == gold)
    return out


# ---------------------------------------------------------------- running (real models)


def llm_client(model: str) -> Any:  # pragma: no cover - a real server
    from bench.rag_eval import serve
    from chatcore import augment
    from chatcore.generation import LlmClient, LlmConfig

    try:
        urllib.request.urlopen(f"{LLM_URL}/v1/models", timeout=2)
    except OSError:
        serve(model)  # left running for the rest of the run
    return LlmClient(LlmConfig(LLM_URL, model, augment.TIMEOUT))


def write_bot(directory: Path, nlu: dict[str, Any]) -> Path:  # pragma: no cover - files
    if directory.exists():
        shutil.rmtree(directory)
    shutil.copytree(ROOT / "bots/sample", directory)
    (directory / "nlu.yaml").write_text(yaml.safe_dump(nlu, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return directory


def runtime_for(directory: Path, embedder: Any) -> Any:  # pragma: no cover - real model
    from chatcore.bot.loader import load_bot
    from chatcore.nlu.model import CachedEmbedder
    from chatcore.nlu.train import train
    from chatcore.runtime import build_runtime
    from chatcore.store.memory import InMemoryStore

    artifacts = SCRATCH / "artifacts"
    bot = load_bot(directory)
    if not (artifacts / bot.id / bot.version).exists():
        train(bot, CachedEmbedder(embedder), artifacts)  # deploy training: cross-validated thresholds
    return build_runtime(directory, artifacts, embedder, InMemoryStore())


def decisions(rt: Any, texts: Sequence[str]) -> list[tuple[str, str | None]]:  # pragma: no cover - real model
    from chatcore.dialogue.state import DialogueState

    out = []
    for t in texts:
        info = rt.processor.interpreter.interpret(t, DialogueState(), rt.bot).info
        out.append((info["rule"], info["intents"][0][0] if info.get("intents") else None))
    return out


@dataclass
class Augmented:
    model: str
    stats: dict[str, dict[str, int]]
    added: dict[str, list[str]]  # intent -> examples added
    similar: dict[str, list[str]]  # FAQ id -> similar questions added
    seconds: float


def generate(model: str, nlu: dict[str, Any], base_rt: Any, faq: list[dict[str, Any]], base_index: Any, embedder: Any,
             cache: dict[str, Any]) -> Augmented:  # pragma: no cover - real models
    from chatcore import augment
    from chatcore.nlu.entities import normalize
    from chatcore.nlu.train import PLACEHOLDER, entity_forms, fill

    client = llm_client(model)
    save = lambda: CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")  # noqa: E731
    stats: dict[str, dict[str, int]] = {"intent": {}, "faq": {}}
    forms, rng = entity_forms(base_rt.bot), random.Random(0)
    started = time.perf_counter()
    added: dict[str, list[str]] = {}
    for intent, spec in nlu["intents"].items():
        if intent == "out_of_scope" or spec.get("exact_only"):
            continue
        examples = spec["examples"]
        items = proposals(model, augment.intent_messages(spec.get("label") or intent, examples, PER_INTENT), cache,
                          lambda m: augment.ask(client, m))
        save()

        def conflict(text: str, intent: str = intent) -> str | None:
            top = base_rt.model.parse(fill(text, rng, forms)).intents[0]
            return f"{top.intent} {top.score:.2f}" if top.intent != intent and top.score >= augment.CONFLICT_PROB else None

        screened = augment.screen(items, examples, {p for e in examples for p in PLACEHOLDER.findall(e)}, conflict)
        for p in screened:
            stats["intent"][p.status] = stats["intent"].get(p.status, 0) + 1
        added[intent] = [p.text for p in screened if p.status == "ok"]
        print(model, "intent", intent, len(added[intent]), flush=True)
    similar: dict[str, list[str]] = {}
    for f in faq:
        items = proposals(model, augment.faq_messages(f["question"], f["answer"], [], PER_FAQ), cache,
                          lambda m: augment.ask(client, m))
        save()

        def faq_conflict(text: str, fid: str = f["id"]) -> str | None:
            scores = base_index.faq_scores(embedder.encode([normalize(text)])[0])
            mine = scores.pop(fid, -1.0)
            other, best = max(scores.items(), key=lambda kv: kv[1])
            return other if best > mine + augment.FAQ_MARGIN else None

        screened = augment.screen(items, [f["question"]], set(), faq_conflict)
        for p in screened:
            stats["faq"][p.status] = stats["faq"].get(p.status, 0) + 1
        similar[f["id"]] = [p.text for p in screened if p.status == "ok"][:50]
    print(model, "faq done", flush=True)
    return Augmented(model, stats, added, similar, time.perf_counter() - started)


def main() -> int:  # pragma: no cover - real models
    from bench.kb_eval import nlu_decisions
    from chatcore.knowledge import CrossEncoderJudge, Knowledge, KnowledgeIndex
    from chatcore.nlu.entities import normalize
    from chatcore.nlu.model import SentenceEmbedder

    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Any] = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    embedder = SentenceEmbedder()
    nlu = yaml.safe_load((ROOT / "bots/sample/nlu.yaml").read_text(encoding="utf-8"))
    base_rt = runtime_for(write_bot(SCRATCH / "base", nlu), embedder)
    kb_root = ROOT / "bench/data/kb"
    faq = yaml.safe_load((kb_root / "faq.yaml").read_text(encoding="utf-8"))["faq"]
    docs = [{"id": p.stem, "name": p.name, "text": p.read_text(encoding="utf-8")} for p in sorted((kb_root / "docs").glob("*.md"))]
    base_index = KnowledgeIndex.build({"faq": faq, "documents": docs}, embedder)
    runs = [generate(m, nlu, base_rt, faq, base_index, embedder, cache) for m in MODELS]

    configs: dict[str, tuple[Any, Any]] = {"원래": (base_rt, base_index)}
    for run in runs:
        name = run.model.split("/")[-1]
        rt = runtime_for(write_bot(SCRATCH / name, augmented_nlu(nlu, run.added)), embedder)
        configs[name] = (rt, KnowledgeIndex.build({"faq": augmented_faq(faq, run.similar), "documents": docs}, embedder))

    block = load_block_test()
    noisy_block = [noise(s.text) for s in block]
    golds = [s.gold for s in block]
    blocks = {name: {"clean": block_scores(golds, decisions(rt, [s.text for s in block])),
                     "noisy": block_scores(golds, decisions(rt, [t for t, _ in noisy_block]))}
              for name, (rt, _) in configs.items()}

    items = load_kb()
    by_text = {i.text: i for i in items}
    by_id = {i.id: i for i in items}
    questions = [q for q in load_questions(QUESTIONS, items) if q.kind != "block"]
    noisy_q = [Question(noise(q.text)[0], q.kind, q.gold, q.intent, q.half) for q in questions]
    judge = CrossEncoderJudge()
    kb: dict[str, dict[str, dict[str, list[bool]]]] = {name: {} for name in configs}
    for variant, qs in (("clean", questions), ("noisy", noisy_q)):
        nlu_d, _ = nlu_decisions(qs)  # the original block model on every side: only the index differs
        for name, (_, index) in configs.items():
            knowledge = Knowledge(index, judge, embedder)
            entries = {e.id: e for e in [*index.faq, *index.chunks]}
            rows: dict[str, list[bool]] = {"correct": [], "wrong": [], "unans": []}
            for q, n in zip(qs, nlu_d, strict=True):
                _, answer = knowledge.consult(q.text, n.rule, n.similarity, None)
                item = None
                if answer is not None and answer.items:
                    first = entries[answer.items[0]]
                    item = by_id.get(first.id) if first.kind == "faq" else by_text.get(first.judge_text)
                if q.kind == "answerable":
                    rows["correct"].append(kb_correct(q, item))
                    rows["wrong"].append(answer is not None and not kb_correct(q, item))
                else:
                    rows["unans"].append(answer is not None)
            kb[name][variant] = rows

    probes = []
    for kind, text, _ in PROBES:
        row = [text]
        for rt, index in configs.values():
            if kind == "intent":
                rule, top = decisions(rt, [text])[0]
                row.append(f"{top} ({rule})")
            else:
                found = index.search(text, embedder.encode([normalize(text)])[0], judge)
                row.append(found.answers[0].entry.title if found.answers else "(답 없음)")
        probes.append(row)
    examples = {run.model.split("/")[-1]: run for run in runs}
    REPORT.write_text(report(nlu, examples, blocks, kb, probes, noisy_block), encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))
    return 0


def report(nlu: dict[str, Any], runs: dict[str, Augmented], blocks: dict[str, Any], kb: dict[str, Any],
           probes: list[list[str]], noisy_block: list[tuple[str, str]]) -> str:  # pragma: no cover - formatting
    names = list(blocks)
    base = names[0]

    def cell(a: list[bool], b: list[bool]) -> str:
        if a is b:
            return f"{np.mean(a):.3f}"
        m, (lo, hi) = paired(a, b)
        return f"{np.mean(b):.3f} ({m:+.3f} [{lo:+.3f}, {hi:+.3f}])"

    kinds: dict[str, int] = {}
    for _, k in noisy_block:
        kinds[k] = kinds.get(k, 0) + 1
    lines = ["# LLM 예문·비슷한 질문 채우기 측정 (E-H)", "", "`uv run python bench/augment_eval.py`가 만든 파일. 손으로 고치지 말 것.", "",
             f"- 블록마다 {PER_INTENT}개·FAQ마다 {PER_FAQ}개를 제안받아(오타는 따로 한 번, 나머지 종류는 한 번) 엔진이 거른 뒤 `ok`를 모두 넣음(사람이 전부 받아들인 경우).",
             "- 오타판: 문장마다 규칙으로 오타 하나(LLM과 다른 방식). 종류별 문장 수 " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ".",
             "- 칸: 값 (원래와의 차이 [같은 문장끼리 쌍 부트스트랩 95% 구간]). 블록은 배포 학습 후 실제 해석기, 지식베이스는 `Knowledge.consult`(재순위 판단, 블록 판단은 양쪽 다 원래 모델).", "",
             "## 무엇이 늘었나", "", "| 생성 모델 | 블록 예문 | FAQ 비슷한 질문 | 블록 제안 결과 | FAQ 제안 결과 | 걸린 시간 |", "|---|---|---|---|---|---|"]
    before = sum(len(s["examples"]) for i, s in nlu["intents"].items() if i != "out_of_scope" and not s.get("exact_only"))
    for name, run in runs.items():
        lines.append(f"| {name} | {before} → {before + sum(len(v) for v in run.added.values())} | 0 → {sum(len(v) for v in run.similar.values())} "
                     f"| {run.stats['intent']} | {run.stats['faq']} | {run.seconds:.0f}초 |")
    header = "| | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines += ["", "## 블록 (bench/data/test_independent.yaml)", "", header, sep]
    for variant, label in (("clean", "원문"), ("noisy", "오타판")):
        for key, title in (("in_scope", "범위 안 정확도"), ("oos_hard", "범위 밖(hard) 거절률"), ("oos_easy", "범위 밖(easy) 거절률")):
            a = blocks[base][variant][key]
            lines.append(f"| {label} {title} (n={len(a)}) | " + " | ".join(cell(a, blocks[n][variant][key]) for n in names) + " |")
    lines += ["", "## 지식베이스 (bench/data/kb_eval.yaml)", "", header, sep]
    for variant, label in (("clean", "원문"), ("noisy", "오타판")):
        for key, title in (("correct", "맞게 답함"), ("wrong", "틀리게 답함"), ("unans", "답 없는 질문에 답함")):
            a = kb[base][variant][key]
            lines.append(f"| {label} {title} (n={len(a)}) | " + " | ".join(cell(a, kb[n][variant][key]) for n in names) + " |")
    lines += ["", "## 처음 확인한 문장", "", "| 문장 | " + " | ".join(names) + " |", sep]
    lines += ["| " + " | ".join(row) + " |" for row in probes]
    lines += ["", "## 기준 (미리 정함)", ""]
    for name in names[1:]:
        checks = [
            ("오타판 범위 안 정확도가 오름", paired(blocks[base]["noisy"]["in_scope"], blocks[name]["noisy"]["in_scope"])[1][0] > 0),
            ("오타판 지식베이스 맞게 답함이 오름", paired(kb[base]["noisy"]["correct"], kb[name]["noisy"]["correct"])[1][0] > 0),
            ("원문 범위 안 정확도 −0.02 이내", np.mean(blocks[name]["clean"]["in_scope"]) >= np.mean(blocks[base]["clean"]["in_scope"]) - CRITERIA["drop"]),
            ("원문 지식베이스 맞게 답함 −0.02 이내", np.mean(kb[name]["clean"]["correct"]) >= np.mean(kb[base]["clean"]["correct"]) - CRITERIA["drop"]),
            ("범위 밖(hard) 거절률 −0.02 이내", np.mean(blocks[name]["clean"]["oos_hard"]) >= np.mean(blocks[base]["clean"]["oos_hard"]) - CRITERIA["drop"]),
            ("답 없는 질문에 답함 +0.02 이내", np.mean(kb[name]["clean"]["unans"]) <= np.mean(kb[base]["clean"]["unans"]) + CRITERIA["drop"]),
        ]
        lines.append(f"- **{name}:** " + " · ".join(f"{'✅' if ok else '❌'} {t}" for t, ok in checks))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
