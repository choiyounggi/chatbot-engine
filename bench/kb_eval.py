"""Knowledge-base retrieval measurement (builder plan §3.3, engine stage E-D: "측정 먼저").

Sample knowledge base: bench/data/kb/ (synthetic FAQ + documents for a fictional weather app).
Questions: bench/data/kb_eval.yaml, written by a separate agent from the knowledge base's author:
answerable questions (gold = FAQ ids and/or a document phrase), unanswerable questions (must fall
back), and questions the sample bot's blocks should take.

Two pools, ranked separately because their cosines are not on one scale (FAQ question vs. user
question are both encoded as queries; document chunks are encoded as passages, plan §3.1):
  - retrieval quality (Recall@3, MRR) is measured in the gold's pool;
  - the answer decision: the FAQ pool's top-1 if its cosine passes the FAQ threshold, else the
    document pool's top-1 if it passes the document threshold, else no answer (plan §3.1 step 4:
    decide by the top-1 cosine only, separate thresholds for FAQ and documents).
Where the knowledge base sits among the dialogue rules (plan §3.1 table), on a fresh session:
  rule 3 (a block is sure)  -> the block, unless the FAQ top-1 passes and beats the block's nearest
                               example similarity by OVERRIDE_MARGIN
  rule 5 (clarify)          -> the knowledge base if it answers, else clarify
  rule 6 (fallback)         -> the knowledge base if it answers, else fallback
Thresholds are chosen on the dev half and reported on the test half. Every rate has a bootstrap
95% interval. Latency is measured with the pools padded to LATENCY_CHUNKS document chunks.

Variants: embedding only / BM25 only / hybrid (RRF) / hybrid + cross-encoder rerank.

Usage: uv run python bench/kb_eval.py   (after `make train`; downloads the reranker once)
Writes docs/benchmarks/kb.md.
"""

from __future__ import annotations

import math
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "bench/data/kb"
QUESTIONS = ROOT / "bench/data/kb_eval.yaml"
REPORT = ROOT / "docs/benchmarks/kb.md"
EMBED_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
RERANK_MODEL = "dragonkue/bge-reranker-v2-m3-ko"
QUERY_PROMPT = "query: "  # the NLU's prompt (chatcore.nlu.model.EmbeddingConfig); passages get none

CHUNK_MIN, CHUNK_MAX = 300, 500  # characters (plan §3.1)
OVERLAP_MAX = 150  # the previous chunk's last sentence is repeated when it is at most this long
RRF_K = 60
RERANK_DEPTH = 20  # per pool
OVERRIDE_MARGIN = 0.05  # plan §3.1: FAQ beats a sure block only by this much over its nearest example
GRID = np.round(np.arange(0.30, 0.96, 0.01), 2)
CRITERIA = {"recall3": 0.90, "fallback": 0.90, "kb_takes_block": 0.05, "block_takes_kb": 0.05, "p95_ms": 100.0}
BOOTSTRAP = 2000
SEED = 7
LATENCY_CHUNKS = 10_000
VARIANTS = ("embedding", "bm25", "hybrid", "hybrid+rerank")


class BenchError(RuntimeError):
    pass


# ---------------------------------------------------------------- knowledge base


@dataclass(frozen=True)
class Item:
    id: str  # faq_001 or <doc>#<n>
    pool: str  # "faq" | "doc"
    text: str  # what BM25 and the reranker read
    embed_text: str  # what is embedded (FAQ: the question; chunk: its text)
    doc: str = ""  # file name, for chunks


def sentences(paragraph: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph.strip())]
    return [s for s in parts if s]


def chunk_document(name: str, text: str) -> list[Item]:
    """Whole sentences packed into chunks of CHUNK_MIN..CHUNK_MAX characters within one `##` section,
    each prefixed with "<title> - <section>". A chunk repeats the previous chunk's last sentence when
    that sentence is at most OVERLAP_MAX characters. A single sentence longer than CHUNK_MAX is its own
    chunk (never cut), so a phrase inside one sentence is always inside one chunk."""
    title, section = name, ""
    sections: list[tuple[str, list[str]]] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        heading = [ln for ln in lines if ln.startswith("#")]
        for ln in heading:
            if ln.startswith("# "):
                title = ln[2:].strip()
            elif ln.startswith("## "):
                section = ln[3:].strip()
                sections.append((section, []))
        body = " ".join(ln.strip() for ln in lines if not ln.startswith("#")).strip()
        if body:
            if not sections:
                sections.append(("", []))
            sections[-1][1].extend(sentences(body))
    items: list[Item] = []
    for section, sents in sections:
        head = f"{title} - {section}" if section else title
        current: list[str] = []
        for s in sents:
            if current and len(" ".join([*current, s])) > CHUNK_MAX and len(" ".join(current)) >= CHUNK_MIN // 2:
                items.append(_chunk(name, head, current, len(items)))
                last = current[-1]
                current = [last] if len(last) <= OVERLAP_MAX and len(last) + len(s) < CHUNK_MAX else []
            current.append(s)
        if current:
            items.append(_chunk(name, head, current, len(items)))
    return items


def _chunk(name: str, head: str, sents: list[str], n: int) -> Item:
    body = " ".join(sents)
    text = f"{head}\n{body}"
    return Item(f"{name}#{n}", "doc", text, text, name)


def load_kb(root: Path = KB) -> list[Item]:
    raw = yaml.safe_load((root / "faq.yaml").read_text(encoding="utf-8"))
    faq = raw.get("faq") if isinstance(raw, dict) else None
    if not faq:
        raise BenchError(f"{root / 'faq.yaml'} has no faq entries")
    items = [Item(f["id"], "faq", f"{f['question']}\n{f['answer']}", f["question"]) for f in faq]
    docs = sorted((root / "docs").glob("*.md"))
    if not docs:
        raise BenchError(f"no documents in {root / 'docs'}")
    for d in docs:
        items.extend(chunk_document(d.name, d.read_text(encoding="utf-8")))
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise BenchError("duplicate knowledge-base ids")
    return items


# ---------------------------------------------------------------- questions


@dataclass(frozen=True)
class Gold:
    faq: frozenset[str]
    phrases: tuple[tuple[str, str], ...]  # (doc file, phrase that a right chunk contains)

    def hit(self, item: Item) -> bool:
        if item.pool == "faq":
            return item.id in self.faq
        return any(item.doc == doc and phrase in item.text for doc, phrase in self.phrases)


@dataclass(frozen=True)
class Question:
    text: str
    kind: str  # "answerable" | "unanswerable" | "block"
    gold: Gold | None = None
    intent: str | None = None
    half: int = 0  # 0 = dev, 1 = test


def load_questions(path: Path, items: Sequence[Item]) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    faq_ids = {i.id for i in items if i.pool == "faq"}
    out: list[Question] = []
    for row in raw.get("answerable") or []:
        faq: set[str] = set()
        phrases: list[tuple[str, str]] = []
        for g in row["gold"]:
            if isinstance(g, str):
                if g not in faq_ids:
                    raise BenchError(f"unknown FAQ id {g} for {row['q']!r}")
                faq.add(g)
            else:
                phrases.append((g["doc"], g["evidence"]))
        gold = Gold(frozenset(faq), tuple(phrases))
        if not any(gold.hit(i) for i in items):
            raise BenchError(f"no knowledge-base item matches the gold of {row['q']!r}")
        out.append(Question(row["q"], "answerable", gold=gold))
    out += [Question(q, "unanswerable") for q in raw.get("unanswerable") or []]
    out += [Question(row["q"], "block", intent=row["intent"]) for row in raw.get("blocks") or []]
    if not out:
        raise BenchError(f"{path} has no questions")
    return split_halves(out)


def split_halves(questions: list[Question], seed: int = SEED) -> list[Question]:
    """Half 0 (dev) / 1 (test), alternating within each kind after a seeded shuffle."""
    rng = random.Random(seed)
    out: list[Question] = []
    for kind in ("answerable", "unanswerable", "block"):
        group = [q for q in questions if q.kind == kind]
        rng.shuffle(group)
        out += [Question(q.text, q.kind, q.gold, q.intent, i % 2) for i, q in enumerate(group)]
    return out


# ---------------------------------------------------------------- ranking


class BM25:
    """Okapi BM25 over pre-tokenized documents (k1=1.5, b=0.75)."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float64)
        self.avg = float(self.lengths.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings.setdefault(term, []).append((i, tf))

    def scores(self, query: Sequence[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        if not self.n or not self.avg:
            return out
        for term in set(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                out[i] += idf * tf * (self.k1 + 1) / norm
        return out


def rrf(*rankings: Sequence[int], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion of index lists (best first); ties keep the first ranking's order."""
    score: dict[int, float] = {}
    first: dict[int, int] = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + r + 1)
            first.setdefault(idx, len(first))
    return sorted(score, key=lambda i: (-score[i], first[i]))


def order(scores: np.ndarray) -> list[int]:
    return [int(i) for i in np.argsort(-scores, kind="stable")]


# ---------------------------------------------------------------- decisions


@dataclass(frozen=True)
class PoolResult:
    ranking: list[int]  # indices into the pool's items, best first
    cosines: np.ndarray  # cosine of every pool item to the question


@dataclass(frozen=True)
class Nlu:
    rule: str  # the interpreter's rule on a fresh session: intent | clarify | fallback | rerun | system | slot
    intent: str | None  # the block it would start (rule "intent")
    similarity: float  # nearest in-scope example cosine


def kb_answer(faq: PoolResult, doc: PoolResult, t_faq: float, t_doc: float) -> tuple[str, int] | None:
    """(pool, index) the knowledge base answers with, or None."""
    if faq.ranking and faq.cosines[faq.ranking[0]] >= t_faq:
        return "faq", faq.ranking[0]
    if doc.ranking and doc.cosines[doc.ranking[0]] >= t_doc:
        return "doc", doc.ranking[0]
    return None


def final(nlu: Nlu, faq: PoolResult, doc: PoolResult, t_faq: float, t_doc: float) -> tuple[str, tuple[str, int] | None]:
    """What the bot does: ("kb", answer) | ("block", None) | ("clarify", None) | ("fallback", None)."""
    answer = kb_answer(faq, doc, t_faq, t_doc)
    if nlu.rule == "intent":
        if answer is not None and answer[0] == "faq" and faq.cosines[answer[1]] >= nlu.similarity + OVERRIDE_MARGIN:
            return "kb", answer
        return "block", None
    if nlu.rule in ("clarify", "fallback"):
        return ("kb", answer) if answer is not None else (nlu.rule, None)
    return "block", None  # rerun / system / slot: the knowledge base is not consulted


@dataclass
class Outcome:
    """Per-question booleans for one threshold pair (arrays align with the question list)."""

    correct: np.ndarray  # answerable: the knowledge base answered with a gold item
    wrong: np.ndarray  # answerable: answered with a non-gold item
    to_block: np.ndarray  # answerable: a block took it
    fallback: np.ndarray  # unanswerable: nothing answered (fallback or clarify)
    kb_takes_block: np.ndarray  # block question: the knowledge base answered


def outcomes(questions: Sequence[Question], nlu: Sequence[Nlu], faq: Sequence[PoolResult], doc: Sequence[PoolResult],
             pools: Mapping[str, Sequence[Item]], t_faq: float, t_doc: float) -> Outcome:
    n = len(questions)
    o = Outcome(*(np.zeros(n, dtype=bool) for _ in range(5)))
    for i, q in enumerate(questions):
        what, answer = final(nlu[i], faq[i], doc[i], t_faq, t_doc)
        if q.kind == "answerable":
            assert q.gold is not None
            if what == "kb":
                assert answer is not None
                good = q.gold.hit(pools[answer[0]][answer[1]])
                o.correct[i], o.wrong[i] = good, not good
            o.to_block[i] = what == "block"
        elif q.kind == "unanswerable":
            o.fallback[i] = what in ("fallback", "clarify")
        else:
            o.kb_takes_block[i] = what == "kb"
    return o


# report name -> (Outcome field, question kind it is a rate over)
METRICS = {"correct": ("correct", "answerable"), "wrong": ("wrong", "answerable"),
           "block_takes_kb": ("to_block", "answerable"), "fallback": ("fallback", "unanswerable"),
           "kb_takes_block": ("kb_takes_block", "block")}


def _values(o: Outcome, questions: Sequence[Question], half: int | None, name: str) -> np.ndarray:
    attr, kind = METRICS[name]
    mask = np.array([q.kind == kind and (half is None or q.half == half) for q in questions], dtype=bool)
    if not mask.any():
        raise BenchError(f"no {kind} questions in half {half}")
    return np.asarray(getattr(o, attr), dtype=float)[mask]


def rates(o: Outcome, questions: Sequence[Question], half: int | None) -> dict[str, float]:
    return {name: float(_values(o, questions, half, name).mean()) for name in METRICS}


def rates_ci(o: Outcome, questions: Sequence[Question], half: int | None) -> dict[str, tuple[float, tuple[float, float]]]:
    out = {}
    for name in METRICS:
        v = _values(o, questions, half, name)
        out[name] = (float(v.mean()), bootstrap(v))
    return out


def choose_thresholds(evaluate: Callable[[float, float], dict[str, float]],
                      grid: Sequence[float] = tuple(GRID)) -> tuple[float, float, dict[str, float]] | None:
    """On the dev half: among threshold pairs with fallback >= CRITERIA and kb_takes_block <= CRITERIA,
    the one with the most correct answers, then the fewest wrong ones, then the highest thresholds.
    None if no pair meets both constraints."""
    best: tuple[tuple[float, float, float, float], float, float, dict[str, float]] | None = None
    for t_faq in grid:
        for t_doc in grid:
            r = evaluate(float(t_faq), float(t_doc))
            if r["fallback"] < CRITERIA["fallback"] or r["kb_takes_block"] > CRITERIA["kb_takes_block"]:
                continue
            key = (r["correct"], -r["wrong"], float(t_faq), float(t_doc))
            if best is None or key > best[0]:
                best = (key, float(t_faq), float(t_doc), r)
    return None if best is None else (best[1], best[2], best[3])


def recall_mrr(gold: Gold, faq_items: Sequence[Item], doc_items: Sequence[Item],
               faq: PoolResult, doc: PoolResult) -> tuple[bool, float]:
    """Recall@3 and reciprocal rank in the gold's pool (the better pool when gold is in both)."""
    best_rank: int | None = None
    for items, pool in ((faq_items, faq), (doc_items, doc)):
        for r, idx in enumerate(pool.ranking):
            if gold.hit(items[idx]):
                best_rank = r if best_rank is None else min(best_rank, r)
                break
    if best_rank is None:
        return False, 0.0
    return best_rank < 3, 1.0 / (best_rank + 1)


def bootstrap(values: np.ndarray, n: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling questions."""
    if len(values) == 0:
        raise BenchError("empty sample")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- models (real; not unit-tested)


class Tokenizer:  # pragma: no cover - Kiwi
    """Content morphemes for BM25: nouns, verb/adjective stems, roots, foreign words, numbers."""

    KEEP = ("NN", "VV", "VA", "XR", "SL", "SN", "SH", "MAG")

    def __init__(self) -> None:
        from kiwipiepy import Kiwi

        self.kiwi = Kiwi()

    def __call__(self, text: str) -> list[str]:
        return [t.form.lower() for t in self.kiwi.tokenize(text) if t.tag.startswith(self.KEEP)]


class Encoder:  # pragma: no cover - the real model
    """The NLU's embedding model loaded once; `prefix` is the prompt ("query: " or "" for passages)."""

    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        mps = torch.backends.mps.is_available()
        self.model = SentenceTransformer(EMBED_MODEL, device="mps" if mps else "cpu",
                                         model_kwargs={"dtype": torch.float16 if mps else torch.float32})

    def __call__(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        from chatcore.nlu.entities import normalize

        vecs = self.model.encode([prefix + normalize(t) for t in texts], normalize_embeddings=True, batch_size=64)
        return np.asarray(vecs, dtype=np.float32)


class Reranker:  # pragma: no cover - the real model
    def __init__(self) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(RERANK_MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")

    def __call__(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        return np.asarray(self.model.predict([(query, t) for t in texts], batch_size=32), dtype=np.float64)


@dataclass
class Pool:  # pragma: no cover - built from the real models
    items: list[Item]
    vectors: np.ndarray
    bm25: BM25

    def rank(self, variant: str, qvec: np.ndarray, qtokens: Sequence[str], query: str,
             reranker: Reranker | None) -> PoolResult:
        cos = self.vectors @ qvec
        emb = order(cos)
        if variant == "embedding":
            ranking = emb
        else:
            lex = order(self.bm25.scores(qtokens))
            if variant == "bm25":
                ranking = lex
            else:
                ranking = rrf(emb, lex)
                if variant == "hybrid+rerank":
                    if reranker is None:
                        raise BenchError("no reranker")
                    head = ranking[:RERANK_DEPTH]
                    scores = reranker(query, [self.items[i].text for i in head])
                    ranking = [head[i] for i in order(scores)] + ranking[RERANK_DEPTH:]
        return PoolResult(ranking, cos)


def build_pool(items: list[Item], encoder: Encoder, tokenize: Tokenizer, prefix: str) -> Pool:  # pragma: no cover
    return Pool(items, encoder([i.embed_text for i in items], prefix), BM25([tokenize(i.text) for i in items]))


def nlu_decisions(questions: Sequence[Question]) -> tuple[list[Nlu], Any]:  # pragma: no cover - real model
    """The sample bot's interpreter on a fresh session for every question; also its embedder."""
    from chatcore.dialogue.state import DialogueState
    from chatcore.nlu.model import SentenceEmbedder
    from chatcore.runtime import build_runtime
    from chatcore.store.memory import InMemoryStore

    embedder = SentenceEmbedder()
    rt = build_runtime(ROOT / "bots/sample", ROOT / "artifacts", embedder, InMemoryStore())
    scenario_of = {sc.trigger.intent: sid for sid, sc in rt.bot.scenarios.items() if sc.trigger is not None}
    out = []
    for q in questions:
        interp = rt.processor.interpreter.interpret(q.text, DialogueState(), rt.bot)
        info = interp.info
        top = info["intents"][0][0] if info.get("intents") else None
        out.append(Nlu(info["rule"], scenario_of.get(top) if info["rule"] == "intent" else None,
                       float(info.get("similarity", 0.0))))
    return out, embedder


def filler_chunks(items: Sequence[Item], n: int, seed: int = SEED) -> list[Item]:
    """`n` chunks of shuffled document sentences, for latency only (never scored)."""
    pool = [s for i in items if i.pool == "doc" for s in sentences(i.text.split("\n", 1)[-1])]
    if not pool:
        raise BenchError("no document sentences to pad with")
    rng = random.Random(seed)
    out = []
    for k in range(n):
        body: list[str] = []
        while len(" ".join(body)) < CHUNK_MIN:
            body.append(rng.choice(pool))
        out.append(Item(f"filler#{k}", "doc", " ".join(body), " ".join(body), "filler"))
    return out


def main() -> int:  # pragma: no cover - runs the real models
    items = load_kb()
    questions = load_questions(QUESTIONS, items)
    faq_items = [i for i in items if i.pool == "faq"]
    doc_items = [i for i in items if i.pool == "doc"]
    print(f"knowledge base: {len(faq_items)} FAQ, {len(doc_items)} chunks; questions: "
          f"{Counter(q.kind for q in questions)}", flush=True)

    nlu, nlu_embedder = nlu_decisions(questions)
    print("nlu rules:", Counter(n.rule for n in nlu), flush=True)
    encoder, tokenize = Encoder(), Tokenizer()
    reranker = Reranker()
    faq_pool = build_pool(faq_items, encoder, tokenize, QUERY_PROMPT)
    doc_pool = build_pool(doc_items, encoder, tokenize, "")
    qvecs = encoder([q.text for q in questions], QUERY_PROMPT)
    nlu_vecs = nlu_embedder.encode([q.text for q in questions])
    reuse_gap = float(np.max(np.abs(qvecs - nlu_vecs)))  # the NLU's vector is reused (plan §3.1 step 2)
    qtokens = [tokenize(q.text) for q in questions]
    pools = {"faq": faq_items, "doc": doc_items}

    results: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        faq_r = [faq_pool.rank(variant, qvecs[i], qtokens[i], q.text, reranker) for i, q in enumerate(questions)]
        doc_r = [doc_pool.rank(variant, qvecs[i], qtokens[i], q.text, reranker) for i, q in enumerate(questions)]
        ans = [i for i, q in enumerate(questions) if q.kind == "answerable"]
        rm = [recall_mrr(questions[i].gold, faq_items, doc_items, faq_r[i], doc_r[i]) for i in ans]  # type: ignore[arg-type]
        recall = np.array([r for r, _ in rm], dtype=float)
        mrr = np.array([m for _, m in rm])
        ans_test = np.array([questions[i].half == 1 for i in ans])

        def at_dev(t_faq: float, t_doc: float, fr: list[PoolResult] = faq_r, dr: list[PoolResult] = doc_r) -> dict[str, float]:
            return rates(outcomes(questions, nlu, fr, dr, pools, t_faq, t_doc), questions, 0)

        chosen = choose_thresholds(at_dev)
        entry: dict[str, Any] = {
            "recall3": (float(recall[ans_test].mean()), bootstrap(recall[ans_test])),
            "mrr": (float(mrr[ans_test].mean()), bootstrap(mrr[ans_test])),
            "recall3_all": float(recall.mean()), "mrr_all": float(mrr.mean()), "chosen": chosen,
        }
        if chosen is not None:
            entry["test"] = rates_ci(outcomes(questions, nlu, faq_r, doc_r, pools, chosen[0], chosen[1]), questions, 1)
        results[variant] = entry
        print(variant, {k: v for k, v in entry.items() if k in ("recall3", "mrr", "chosen")}, flush=True)

    # latency with LATENCY_CHUNKS document chunks; the question's vector comes from the NLU (not timed)
    padded = doc_items + filler_chunks(doc_items, LATENCY_CHUNKS - len(doc_items))
    big = build_pool(padded, encoder, tokenize, "")
    latency: dict[str, list[float]] = {}
    for variant in VARIANTS:
        times = []
        for _ in range(3):
            for i, q in enumerate(questions):
                start = time.perf_counter()
                toks = tokenize(q.text)
                faq_pool.rank(variant, qvecs[i], toks, q.text, reranker)
                big.rank(variant, qvecs[i], toks, q.text, reranker)
                times.append((time.perf_counter() - start) * 1000)
        latency[variant] = times
        print(variant, "p95 ms", round(float(np.percentile(times, 95)), 1), flush=True)

    REPORT.write_text(report(results, latency, items, questions, nlu, reuse_gap), encoding="utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    return 0


def passes(entry: dict[str, Any], p95: float) -> bool:
    if entry.get("chosen") is None:
        return False
    t = entry["test"]
    return (entry["recall3"][0] >= CRITERIA["recall3"] and t["fallback"][0] >= CRITERIA["fallback"]
            and t["kb_takes_block"][0] <= CRITERIA["kb_takes_block"]
            and t["block_takes_kb"][0] <= CRITERIA["block_takes_kb"] and p95 <= CRITERIA["p95_ms"])


def report(results: dict[str, dict[str, Any]], latency: dict[str, list[float]], items: Sequence[Item],
           questions: Sequence[Question], nlu: Sequence[Nlu], reuse_gap: float) -> str:  # pragma: no cover
    def pct(x: float) -> str:
        return f"{x:.3f}"

    def with_ci(v: tuple[float, tuple[float, float]]) -> str:
        return f"{v[0]:.3f} [{v[1][0]:.3f}–{v[1][1]:.3f}]"

    kinds = Counter((q.kind, q.half) for q in questions)
    lines = [
        "# 지식베이스 검색 측정 (E-D, 빌더 계획 §3.3)",
        "",
        "`uv run python bench/kb_eval.py`가 만든 파일. 손으로 고치지 말 것.",
        "",
        f"- 지식베이스: FAQ {sum(i.pool == 'faq' for i in items)}개, 문서 조각 {sum(i.pool == 'doc' for i in items)}개 "
        f"(`bench/data/kb/`, 합성).",
        f"- 질문: 답 있음 {kinds[('answerable', 0)]}+{kinds[('answerable', 1)]}, 답 없음 "
        f"{kinds[('unanswerable', 0)]}+{kinds[('unanswerable', 1)]}, 블록 {kinds[('block', 0)]}+{kinds[('block', 1)]} "
        "(개발용+확인용). 임계값은 개발용 절반에서 고르고 아래 표는 확인용 절반 값이다. [괄호]는 부트스트랩 95% 구간.",
        f"- NLU(샘플 봇, 새 세션) 규칙 분포: {dict(Counter(n.rule for n in nlu))}. NLU 벡터 재사용 확인: "
        f"하네스 인코딩과의 최대 차이 {reuse_gap:.4f}.",
        "",
        "| 방식 | Recall@3 | MRR | FAQ 기준 | 문서 기준 | 맞게 답함 | 틀리게 답함 | 블록이 가져감 | 답 없음→폴백 | 지식베이스가 블록 질문 가져감 | p95 ms | 통과 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, e in results.items():
        p95 = float(np.percentile(latency[variant], 95))
        if e["chosen"] is None:
            lines.append(f"| {variant} | {with_ci(e['recall3'])} | {with_ci(e['mrr'])} | — | — | 개발용에서 조건을 만족하는 임계값 없음 "
                         f"| | | | | {p95:.1f} | ❌ |")
            continue
        t = e["test"]
        lines.append(f"| {variant} | {with_ci(e['recall3'])} | {with_ci(e['mrr'])} | {e['chosen'][0]:.2f} | {e['chosen'][1]:.2f} "
                     f"| {with_ci(t['correct'])} | {with_ci(t['wrong'])} | {with_ci(t['block_takes_kb'])} "
                     f"| {with_ci(t['fallback'])} | {with_ci(t['kb_takes_block'])} | {p95:.1f} | {'✅' if passes(e, p95) else '❌'} |")
    lines += [
        "",
        f"기준: Recall@3 ≥ {CRITERIA['recall3']}, 답 없는 질문 폴백 ≥ {CRITERIA['fallback']}, 가로채기 각각 ≤ "
        f"{CRITERIA['kb_takes_block']}, p95 ≤ {CRITERIA['p95_ms']:.0f}ms (문서 조각 {LATENCY_CHUNKS}개, 질문 벡터는 NLU 것을 재사용해 "
        "시간에서 뺌). 두 절반 전체의 Recall@3/MRR(참고): "
        + ", ".join(f"{v} {pct(e['recall3_all'])}/{pct(e['mrr_all'])}" for v, e in results.items()) + ".",
        "",
        "판단 규칙: FAQ 1위가 FAQ 기준 이상이면 그 답, 아니면 문서 1위가 문서 기준 이상이면 그 조각, 아니면 답하지 않음. "
        f"블록이 확실한 질문은 FAQ 1위가 블록의 가장 가까운 예문 유사도보다 {OVERRIDE_MARGIN} 이상 높을 때만 지식베이스. "
        "답 없는 질문은 폴백이나 되묻기로 끝나면 성공. 값을 기다리는 중의 규칙은 새 세션 측정이라 다루지 않는다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
