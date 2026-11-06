"""Phase 0 benchmark: pick the v1 intent-classification setup (no generative LLM).

Two families are compared on bots/sample/nlu.yaml:

- n-gram: {expand | delex} x {char | morph | both} TF-IDF x {LogisticRegression |
  sigmoid-calibrated LinearSVC} x C{1,10}.
- embedding: a frozen sentence-embedding model (+ optionally the n-gram TF-IDF
  features) with a LogisticRegression head, plus an out-of-scope gate on the
  max cosine similarity to in-scope training examples.

Cross-validation is grouped by near-duplicate template clusters (character
n-grams, plus sentence-embedding similarity when available) and repeated over
several seeds. Every setup gets the same two-dimensional out-of-scope rule
(class probability + max cosine to in-scope training rows). Thresholds are
cross-fitted: chosen on one half of the template groups, scored on the other.
The held-out out-of-scope set never chooses thresholds; it does decide
pass/fail, so its 95% interval is reported. Writes docs/benchmarks/phase0.md;
exits 0 only if some setup meets every criterion.

Usage: uv run python bench/phase0.py [--only-embeddings]
"""

from __future__ import annotations

import itertools
import platform
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import kiwipiepy
import numpy as np
import scipy.sparse as sp
import sklearn
import yaml
from kiwipiepy import Kiwi
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize as l2_normalize
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
BOT_NLU = ROOT / "bots/sample/nlu.yaml"
OOS_EVAL = ROOT / "bench/data/out_of_scope_eval.yaml"
REPORT = ROOT / "docs/benchmarks/phase0.md"

OOS = "out_of_scope"
PLACEHOLDER = "{location}"
ENT_TOKEN = "ENTLOC"
SEED = 0
SEEDS = (0, 1, 2, 3, 4)  # repeated CV
N_SPLITS = 5
PER_TEMPLATE = 3  # expansions per template
CAP_RATIO = 2  # expanded rows per intent <= CAP_RATIO * templates
GROUP_SIMILARITY = 0.75  # char-ngram cosine at/above which two templates share a CV group
MARGIN = 0.1  # ambiguity margin for "clarify"
THRESHOLDS = [round(t, 2) for t in np.arange(0.0, 0.96, 0.05)]
SIM_THRESHOLDS = [0.0] + [round(t, 2) for t in np.arange(0.30, 0.99, 0.01)]
LATENCY_SAMPLES = 400
LATENCY_ROUNDS = 3
SEMANTIC_SIMILARITY = 0.90  # complete-link embedding cosine for near-duplicate groups
GROUPING_EMBEDDING = "arctic"  # model used to build the shared groups / OOS overlap check
NEAR_OOS = 0.70  # held-out OOS sentences this close to an OOS training row are "near"
BOOTSTRAP = 1000

CRITERIA = {"macro_f1": 0.90, "oos_reject": 0.80, "p95_ms": 20.0, "train_s": 10.0}


class BenchError(ValueError):
    pass


# ---------------------------------------------------------------- data

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def load_bot(path: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Return (intent -> templates, location surface forms longest-first)."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    intents = {k: [normalize(e) for e in v["examples"]] for k, v in (doc.get("intents") or {}).items()}
    if OOS not in intents:
        raise BenchError(f"{path} needs an {OOS!r} intent")
    empty = [k for k, v in intents.items() if not v]
    if empty:
        raise BenchError(f"intents without examples: {empty}")
    values = doc["entities"]["location"]["values"]
    forms = {normalize(f) for canon, syns in values.items() for f in [canon, *syns]}
    if not forms:
        raise BenchError("location entity has no values")
    return intents, sorted(forms, key=lambda f: (-len(f), f))


def load_oos_eval(path: Path) -> list[str]:
    exs = [normalize(str(e)) for e in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("examples") or []]
    if not exs:
        raise BenchError(f"{path} has no examples")
    return exs


def fill(template: str, rng: random.Random, forms: list[str]) -> str:
    """Replace every placeholder with a random surface form."""
    if PLACEHOLDER in template and not forms:
        raise BenchError(f"no values to fill {template!r}")
    out = template
    while PLACEHOLDER in out:
        out = out.replace(PLACEHOLDER, rng.choice(forms), 1)
    return out


def expand(templates: list[str], forms: list[str], seed: int = SEED) -> list[str]:
    """Training rows for one intent: every template once, then extra fillings
    round-robin until CAP_RATIO * len(templates) rows."""
    rng = random.Random(seed)
    per_template: list[list[str]] = []
    for t in templates:
        if PLACEHOLDER not in t:
            per_template.append([t])
            continue
        k = min(PER_TEMPLATE, len(forms))
        fills = list(dict.fromkeys(fill(t, rng, forms) for _ in range(k * 3)))[:k]
        per_template.append(fills)
    limit = CAP_RATIO * len(templates)
    rows = [variants[0] for variants in per_template]
    for depth in range(1, PER_TEMPLATE):
        for variants in per_template:
            if len(rows) >= limit:
                return rows
            if depth < len(variants):
                rows.append(variants[depth])
    return rows


def near_duplicate_groups(templates: list[str], threshold: float = GROUP_SIMILARITY) -> list[int]:
    """Cluster templates whose char 2-3-gram cosine similarity >= threshold
    (single-link). Raw term frequency (no idf), so a pair's similarity does not
    depend on the rest of the corpus. Returns a cluster id per template."""
    if not templates:
        return []
    texts = [t.replace(PLACEHOLDER, " ") for t in templates]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), use_idf=False)
    sim = cosine_similarity(vec.fit_transform(texts))
    parent = list(range(len(templates)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(sim >= threshold, k=1))):
        parent[root(int(i))] = root(int(j))
    ids: dict[int, int] = {}
    return [ids.setdefault(root(i), len(ids)) for i in range(len(templates))]


def semantic_groups(vectors: np.ndarray, threshold: float = SEMANTIC_SIMILARITY) -> list[int]:
    """Complete-link clusters of L2-normalized vectors (cosine >= threshold
    between every pair in a cluster, so no chaining)."""
    if len(vectors) == 0:
        return []
    if len(vectors) == 1:
        return [0]
    model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="complete", distance_threshold=1 - threshold)
    return [int(g) for g in model.fit_predict(vectors)]


def combine_groups(*labelings: list[int]) -> list[int]:
    """Connected components of 'same label in any labeling'."""
    n = len(labelings[0]) if labelings else 0
    if any(len(lab) != n for lab in labelings):
        raise BenchError("labelings differ in length")
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for lab in labelings:
        first: dict[int, int] = {}
        for i, g in enumerate(lab):
            if g in first:
                parent[root(i)] = root(first[g])
            else:
                first[g] = i
    ids: dict[int, int] = {}
    return [ids.setdefault(root(i), len(ids)) for i in range(n)]


def template_groups(intents: dict[str, list[str]], embed=None) -> list[int]:
    """Global group id per template (intents in sorted order). Templates of
    different intents never share a group."""
    groups: list[int] = []
    for intent in sorted(intents):
        ts = intents[intent]
        local = near_duplicate_groups(ts)
        if embed is not None:
            local = combine_groups(local, semantic_groups(embed([t.replace(PLACEHOLDER, "서울") for t in ts])))
        offset = max(groups, default=-1) + 1
        groups += [offset + g for g in local]
    return groups


class Delexicalizer:
    """Replace location surface forms with ENT_TOKEN, only where the match
    starts and ends on Kiwi token boundaries (부산은 → ENTLOC은, but a form
    buried inside a longer token is left alone)."""

    def __init__(self, forms: list[str], kiwi: Kiwi) -> None:
        if not forms:
            raise BenchError("no entity forms")
        self.pattern = re.compile("|".join(re.escape(f) for f in forms))  # forms are longest-first
        self.kiwi = kiwi

    def __call__(self, text: str) -> str:
        matches = list(self.pattern.finditer(text))
        if not matches:
            return text
        tokens = self.kiwi.tokenize(text)
        starts = {t.start for t in tokens}
        ends = {t.start + t.len for t in tokens}
        out, last = [], 0
        for m in matches:
            if m.start() in starts and m.end() in ends:
                out += [text[last:m.start()], ENT_TOKEN]
                last = m.end()
        out.append(text[last:])
        return "".join(out)


# ---------------------------------------------------------------- embeddings

@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    model_id: str
    prompt: str  # prepended to every text (classification is symmetric)
    device: str
    dtype: str


EMBEDDINGS = {
    s.key: s
    for s in (
        EmbeddingSpec("e5s", "intfloat/multilingual-e5-small", "query: ", "cpu", "float32"),
        EmbeddingSpec("e5b", "intfloat/multilingual-e5-base", "query: ", "mps", "float16"),
        EmbeddingSpec("arctic", "dragonkue/snowflake-arctic-embed-l-v2.0-ko", "query: ", "mps", "float16"),
        EmbeddingSpec("qwen3", "Qwen/Qwen3-Embedding-0.6B",
                      "Instruct: Classify the intent of a Korean chatbot user message\nQuery:", "mps", "float16"),
    )
}


class Embedder:
    """L2-normalized sentence embeddings with an in-memory cache (the cache is
    for the benchmark's repeated folds; `encode_fresh` bypasses it)."""

    def __init__(self, spec: EmbeddingSpec) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.model = SentenceTransformer(spec.model_id, device=spec.device, model_kwargs={"dtype": getattr(torch, spec.dtype)})
        self.cache: dict[str, np.ndarray] = {}

    def encode_fresh(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise BenchError("no texts to embed")
        vecs = self.model.encode([self.spec.prompt + t for t in texts], normalize_embeddings=True, batch_size=64)
        return np.asarray(vecs, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            self.cache.update(zip(missing, self.encode_fresh(missing)))
        return np.vstack([self.cache[t] for t in texts])


# ---------------------------------------------------------------- models

class KiwiMorphs:
    """Analyzer: morpheme unigrams + bigrams as 'form/TAG'."""

    def __init__(self, kiwi: Kiwi) -> None:
        self.kiwi = kiwi

    def __call__(self, text: str) -> list[str]:
        toks = [f"{t.form}/{t.tag}" for t in self.kiwi.tokenize(text)]
        return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


@dataclass(frozen=True)
class Setup:
    mode: str  # expand | delex
    features: str  # char | morph | both | emb:<key> | emb:<key>+both
    clf: str  # logreg | svc
    C: float = 1.0

    @property
    def name(self) -> str:
        return f"{self.mode}/{self.features}/{self.clf}/C{self.C:g}"

    @property
    def embedding(self) -> str | None:
        return self.features.split(":", 1)[1].split("+")[0] if self.features.startswith("emb:") else None


NGRAM_SETUPS = [
    Setup(m, f, c, C)
    for m, f, c, C in itertools.product(("expand", "delex"), ("char", "morph", "both"), ("logreg", "svc"), (1.0, 10.0))
]
EMBEDDING_SETUPS = [
    Setup("expand", f"emb:{k}{extra}", "logreg", C)
    for k, extra, C in itertools.product(EMBEDDINGS, ("", "+both"), (1.0, 10.0))
]


def ngram_features(features: str, kiwi: Kiwi):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True)
    morph = TfidfVectorizer(analyzer=KiwiMorphs(kiwi), sublinear_tf=True)
    return {"char": char, "morph": morph, "both": FeatureUnion([("char", char), ("morph", morph)])}[features]


def make_classifier(setup: Setup):
    if setup.clf == "logreg":
        return LogisticRegression(C=setup.C, max_iter=5000, class_weight="balanced", random_state=SEED)
    svc = LinearSVC(C=setup.C, class_weight="balanced", random_state=SEED)
    return CalibratedClassifierCV(svc, method="sigmoid", cv=3)


class IntentModel:
    """fit / predict / classes_ for both families. predict also returns the
    max cosine similarity to in-scope training rows: in embedding space when
    the setup has an embedding, otherwise in TF-IDF space."""

    def __init__(self, setup: Setup, kiwi: Kiwi, embedder: Embedder | None, fresh: bool = False) -> None:
        if (setup.embedding is None) != (embedder is None):
            raise BenchError(f"{setup.name}: embedder mismatch")
        self.setup = setup
        self.embedder = embedder
        self.fresh = fresh  # True → never use the embedding cache (timing runs)
        hybrid = setup.features.endswith("+both")
        self.ngrams = ngram_features("both", kiwi) if hybrid else (None if embedder else ngram_features(setup.features, kiwi))
        self.clf = make_classifier(setup)
        self.index: np.ndarray | sp.csr_matrix | None = None  # in-scope training rows for similarity

    def _embed(self, texts: list[str]) -> np.ndarray:
        assert self.embedder is not None
        return self.embedder.encode_fresh(texts) if self.fresh else self.embedder.encode(texts)

    def _features(self, texts: list[str], fit: bool):
        parts = []
        if self.ngrams is not None:
            parts.append(self.ngrams.fit_transform(texts) if fit else self.ngrams.transform(texts))
        emb = None
        if self.embedder is not None:
            emb = self._embed(texts)
            parts.append(sp.csr_matrix(emb))
        X = parts[0] if len(parts) == 1 else sp.hstack(parts, format="csr")
        return X, (emb if emb is not None else l2_normalize(X))

    def fit(self, texts: list[str], labels: list[str]) -> "IntentModel":
        if not texts:
            raise BenchError("no training texts")
        X, space = self._features(texts, fit=True)
        self.clf.fit(X, labels)
        keep = np.array([lab != OOS for lab in labels])
        if not keep.any():
            raise BenchError("no in-scope training rows")
        self.index = space[keep]
        return self

    @property
    def classes_(self) -> np.ndarray:
        return self.clf.classes_

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(class probabilities, max cosine to in-scope training rows)."""
        if not texts:
            raise BenchError("no texts to score")
        if self.index is None:
            raise BenchError("model is not fitted")
        X, space = self._features(texts, fit=False)
        probs = self.clf.predict_proba(X)
        sims = space @ self.index.T
        sims = np.asarray(sims.max(axis=1).todense()).ravel() if sp.issparse(sims) else sims.max(axis=1)
        return probs, sims


def training_rows(setup: Setup, intents: dict[str, list[str]], forms: list[str], delex: Delexicalizer) -> tuple[list[str], list[str]]:
    """Delex rows go through the same delexicalizer as runtime input."""
    texts: list[str] = []
    labels: list[str] = []
    for i, (intent, templates) in enumerate(sorted(intents.items())):
        if setup.mode == "expand":
            rows = expand(templates, forms, seed=SEED + i)
        else:
            rows = [delex(t.replace(PLACEHOLDER, ENT_TOKEN)) for t in templates]
        texts += rows
        labels += [intent] * len(rows)
    return texts, labels


def prepare(setup: Setup, text: str, delex: Delexicalizer) -> str:
    """Runtime input path: normalize, then delexicalize in delex mode."""
    text = normalize(text)
    return delex(text) if setup.mode == "delex" else text


# ---------------------------------------------------------------- metrics

def decide_idx(probs: np.ndarray, oos_idx: int, threshold: float, sims: np.ndarray | None = None, sim_threshold: float = 0.0) -> np.ndarray:
    """Class index per row: the top class, or oos_idx when its probability is
    below threshold or the row is farther than sim_threshold from every
    in-scope training row."""
    reject = probs.max(axis=1) < threshold
    if sims is not None:
        reject |= sims < sim_threshold
    return np.where(reject, oos_idx, probs.argmax(axis=1))


def decide(probs: np.ndarray, classes: np.ndarray, threshold: float, sims: np.ndarray | None = None, sim_threshold: float = 0.0) -> list[str]:
    """String form of decide_idx; classes must include out_of_scope."""
    if OOS not in classes:
        raise BenchError("classes need out_of_scope")
    oos_idx = int(np.flatnonzero(classes == OOS)[0])
    return [str(classes[i]) for i in decide_idx(probs, oos_idx, threshold, sims, sim_threshold)]


def macro_f1_idx(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    """sklearn-equivalent macro-F1 over labels present in y or p."""
    if len(y) == 0:
        raise BenchError("empty evaluation set")
    tp = np.bincount(y[y == p], minlength=n_classes).astype(float)
    denom = np.bincount(y, minlength=n_classes) + np.bincount(p, minlength=n_classes)
    present = denom > 0
    return float(np.mean(2 * tp[present] / denom[present]))


def oos_recall(y_true: list[str], y_pred: list[str]) -> float:
    rows = [p for t, p in zip(y_true, y_pred) if t == OOS]
    if not rows:
        raise BenchError("no out_of_scope rows to measure")
    return sum(p == OOS for p in rows) / len(rows)


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    sim_threshold: float
    macro_f1: float
    oos_recall: float
    constrained: bool  # False → no threshold pair reached the OOS recall target


def choose_threshold(y_true: list[str], probs: np.ndarray, classes: np.ndarray, min_oos_recall: float,
                     sims: np.ndarray | None = None) -> ThresholdChoice:
    """Highest macro-F1 among (probability, similarity) threshold pairs whose
    OOS recall reaches the target; if none does, the highest macro-F1 overall.
    Ties → lowest thresholds."""
    if not y_true:
        raise BenchError("empty evaluation set")
    if OOS not in classes:
        raise BenchError("classes need out_of_scope")
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[t] for t in y_true])
    oos_idx = index[OOS]
    oos_rows = y == oos_idx
    if not oos_rows.any():
        raise BenchError("no out_of_scope rows to measure")
    best_ok: ThresholdChoice | None = None
    best_any: ThresholdChoice | None = None
    for t, s in itertools.product(THRESHOLDS, SIM_THRESHOLDS if sims is not None else [0.0]):
        pred = decide_idx(probs, oos_idx, t, sims, s)
        f1 = macro_f1_idx(y, pred, len(classes))
        rec = float(np.mean(pred[oos_rows] == oos_idx))
        cand = ThresholdChoice(t, s, f1, rec, rec >= min_oos_recall)
        if cand.constrained and (best_ok is None or f1 > best_ok.macro_f1 + 1e-12):
            best_ok = cand
        if best_any is None or f1 > best_any.macro_f1 + 1e-12:
            best_any = cand
    assert best_any is not None
    return best_ok or ThresholdChoice(best_any.threshold, best_any.sim_threshold, best_any.macro_f1, best_any.oos_recall, False)


def clarify_stats(y_true: list[str], y_pred: list[str], probs: np.ndarray, classes: np.ndarray) -> tuple[float, float]:
    """(share of accepted in-scope inputs that would trigger 'clarify',
    share of those whose true intent is in the top 2)."""
    order = np.argsort(-probs, axis=1)
    accepted = clarify = hit = 0
    for r, (truth, pred) in enumerate(zip(y_true, y_pred)):
        if truth == OOS or pred == OOS:
            continue
        accepted += 1
        if probs[r, order[r, 0]] - probs[r, order[r, 1]] < MARGIN:
            clarify += 1
            hit += truth in (classes[order[r, 0]], classes[order[r, 1]])
    return (clarify / accepted if accepted else 0.0), (hit / clarify if clarify else 1.0)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n <= 0:
        raise BenchError("empty sample")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def split_halves(groups: list[int], labels: list[str], seed: int = SEED) -> dict[int, int]:
    """Assign each group to half 0 or 1, alternating within each intent."""
    rng = random.Random(seed)
    by_intent: dict[str, list[int]] = {}
    for g, lab in zip(groups, labels):
        if g not in by_intent.setdefault(lab, []):
            by_intent[lab].append(g)
    half: dict[int, int] = {}
    for lab in sorted(by_intent):
        gs = sorted(by_intent[lab])
        rng.shuffle(gs)
        for k, g in enumerate(gs):
            half[g] = k % 2
    return half


@dataclass
class Result:
    setup: Setup
    choice: ThresholdChoice  # thresholds chosen on all out-of-fold rows (deployment values)
    f1_mean: float  # cross-fitted thresholds, mean over seeds
    f1_std: float
    oos_recall_cv: float  # cross-fitted
    oos_reject: float  # held-out set
    oos_reject_ci: tuple[float, float]
    oos_reject_far: float | None  # held-out sentences not near any OOS training row
    clarify_rate: float
    clarify_top2: float
    p95_ms: float  # worst round
    p95_rounds: list[float]
    cold_ms: float
    train_s: float
    per_intent_f1: dict[str, float]
    confusions: list[tuple[str, str, int]]
    oos_accepted: list[tuple[str, str, float]] = field(default_factory=list)
    rows: dict = field(default_factory=dict, repr=False)  # y, pred, group per pooled row (for bootstrap)

    @property
    def passed(self) -> bool:
        return (self.f1_mean >= CRITERIA["macro_f1"] and self.oos_reject >= CRITERIA["oos_reject"]
                and self.p95_ms <= CRITERIA["p95_ms"] and self.train_s <= CRITERIA["train_s"])


def evaluate(setup: Setup, intents: dict[str, list[str]], forms: list[str], oos_eval: list[str], kiwi: Kiwi,
             embedder: Embedder | None = None, groups: list[int] | None = None,
             far_mask: np.ndarray | None = None) -> Result:
    delex = Delexicalizer(forms, kiwi)
    templates = [(intent, t) for intent in sorted(intents) for t in intents[intent]]
    labels = [i for i, _ in templates]
    groups = template_groups(intents) if groups is None else groups
    if len(groups) != len(templates):
        raise BenchError("groups do not match templates")

    seed_of: list[int] = []
    tmpl_of: list[int] = []
    prob_rows: list[np.ndarray] = []
    sim_rows: list[np.ndarray] = []
    classes: np.ndarray | None = None
    for seed in SEEDS:
        rng = random.Random(seed)
        concrete = [fill(t, rng, forms) for _, t in templates]  # what a user would type
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for train_idx, test_idx in splitter.split(concrete, labels, groups=groups):
            fold: dict[str, list[str]] = {}
            for i in train_idx:
                fold.setdefault(labels[i], []).append(templates[i][1])
            model = IntentModel(setup, kiwi, embedder).fit(*training_rows(setup, fold, forms, delex))
            if classes is None:
                classes = model.classes_
            elif list(classes) != list(model.classes_):
                raise BenchError("a CV fold is missing an intent")
            probs, sims = model.predict([prepare(setup, concrete[i], delex) for i in test_idx])
            prob_rows.append(probs)
            sim_rows.append(sims)
            seed_of += [seed] * len(test_idx)
            tmpl_of += [int(i) for i in test_idx]
    assert classes is not None
    order = np.lexsort((np.array(tmpl_of), np.array(seed_of)))  # stable row order across setups
    seed_arr = np.array(seed_of)[order]
    tmpl_arr = np.array(tmpl_of)[order]
    probs_all = np.vstack(prob_rows)[order]
    sims_all = np.concatenate(sim_rows)[order]
    index = {c: i for i, c in enumerate(classes)}
    oos_idx = index[OOS]
    y_all = np.array([index[labels[t]] for t in tmpl_arr])
    y_true_all = [labels[t] for t in tmpl_arr]
    group_arr = np.array([groups[t] for t in tmpl_arr])

    # Deployment thresholds: all out-of-fold rows.
    choice = choose_threshold(y_true_all, probs_all, classes, CRITERIA["oos_reject"], sims_all)
    # Reported scores: thresholds cross-fitted across template-group halves.
    half_of = split_halves(groups, labels)
    half_arr = np.array([half_of[g] for g in group_arr])
    pred_all = np.empty_like(y_all)
    for h in (0, 1):
        pick, other = half_arr == h, half_arr != h
        c = choose_threshold([y_true_all[i] for i in np.flatnonzero(other)], probs_all[other], classes,
                             CRITERIA["oos_reject"], sims_all[other])
        pred_all[pick] = decide_idx(probs_all[pick], oos_idx, c.threshold, sims_all[pick], c.sim_threshold)
    seed_f1 = [macro_f1_idx(y_all[seed_arr == s], pred_all[seed_arr == s], len(classes)) for s in SEEDS]
    oos_rows = y_all == oos_idx
    pred_names = [str(classes[i]) for i in pred_all]
    clarify_rate, clarify_top2 = clarify_stats(y_true_all, pred_names, probs_all, classes)
    names = sorted(set(y_true_all))
    per_intent = dict(zip(names, f1_score(y_true_all, pred_names, labels=names, average=None, zero_division=0)))
    confusions = Counter((t, p) for t, p in zip(y_true_all, pred_names) if t != p).most_common(5)

    # Full model with uncached embeddings: training time, cold first call, held-out OOS, latency.
    start = time.perf_counter()
    full = IntentModel(setup, kiwi, embedder, fresh=True).fit(*training_rows(setup, intents, forms, delex))
    train_s = time.perf_counter() - start
    rng = random.Random(SEED)
    pool = [fill(t, rng, forms) for _, t in templates] + oos_eval
    samples = rng.sample(pool, min(LATENCY_SAMPLES, len(pool)))
    t0 = time.perf_counter()
    full.predict([prepare(setup, samples[0], delex)])
    cold_ms = (time.perf_counter() - t0) * 1000

    oos_probs, oos_sims = full.predict([prepare(setup, s, delex) for s in oos_eval])
    rejected = decide_idx(oos_probs, int(np.flatnonzero(full.classes_ == OOS)[0]), choice.threshold, oos_sims, choice.sim_threshold) \
        == int(np.flatnonzero(full.classes_ == OOS)[0])
    accepted = [(s, str(full.classes_[oos_probs[r].argmax()]), float(oos_probs[r].max()))
                for r, s in enumerate(oos_eval) if not rejected[r]]
    far = None
    if far_mask is not None:
        if len(far_mask) != len(oos_eval) or not far_mask.any():
            raise BenchError("far_mask must cover the held-out set and select something")
        far = float(rejected[far_mask].mean())

    for s in samples[:20]:  # warm-up
        full.predict([prepare(setup, s, delex)])
    p95_rounds = []
    for _ in range(LATENCY_ROUNDS):
        timings = []
        for s in samples:
            t1 = time.perf_counter()
            full.predict([prepare(setup, s, delex)])
            timings.append((time.perf_counter() - t1) * 1000)
        p95_rounds.append(float(np.percentile(timings, 95)))

    return Result(
        setup=setup, choice=choice,
        f1_mean=float(np.mean(seed_f1)), f1_std=float(np.std(seed_f1)),
        oos_recall_cv=float(np.mean(pred_all[oos_rows] == oos_idx)),
        oos_reject=float(rejected.mean()), oos_reject_ci=wilson(int(rejected.sum()), len(oos_eval)), oos_reject_far=far,
        clarify_rate=clarify_rate, clarify_top2=clarify_top2,
        p95_ms=max(p95_rounds), p95_rounds=p95_rounds, cold_ms=cold_ms, train_s=train_s,
        per_intent_f1={k: float(v) for k, v in per_intent.items()},
        confusions=[(t, p, n) for (t, p), n in confusions], oos_accepted=accepted,
        rows={"y": y_all, "pred": pred_all, "group": group_arr, "n_classes": len(classes)},
    )


def paired_bootstrap(a: Result, b: Result, n: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    """macro-F1(a) - macro-F1(b) on the same rows, resampling template groups.
    Returns (observed difference, 2.5th percentile, 97.5th percentile)."""
    ra, rb = a.rows, b.rows
    if not ra or not np.array_equal(ra["y"], rb["y"]) or not np.array_equal(ra["group"], rb["group"]):
        raise BenchError("results are not on the same rows")
    k = ra["n_classes"]
    observed = macro_f1_idx(ra["y"], ra["pred"], k) - macro_f1_idx(rb["y"], rb["pred"], k)
    uniq = np.unique(ra["group"])
    rows_of = {g: np.flatnonzero(ra["group"] == g) for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([rows_of[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        diffs.append(macro_f1_idx(ra["y"][idx], ra["pred"][idx], k) - macro_f1_idx(rb["y"][idx], rb["pred"][idx], k))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return observed, float(lo), float(hi)


def recommend(results: list[Result]) -> Result | None:
    passed = [r for r in results if r.passed]
    return max(passed, key=lambda r: (round(r.f1_mean, 3), r.oos_reject, -r.p95_ms)) if passed else None


def kiwi_speed(texts: list[str]) -> tuple[float, float]:
    """(model load seconds, sentences per second single-threaded)."""
    if not texts:
        raise BenchError("no texts for the Kiwi speed test")
    t0 = time.perf_counter()
    kiwi = Kiwi()
    load_s = time.perf_counter() - t0
    batch = (texts * (2000 // len(texts) + 1))[:2000]
    t0 = time.perf_counter()
    for s in batch:
        kiwi.tokenize(s)
    return load_s, len(batch) / (time.perf_counter() - t0)


# ---------------------------------------------------------------- report

def machine() -> str:
    cpu = platform.processor()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return f"{cpu} ({platform.machine()}), {platform.system()} {platform.mac_ver()[0] or platform.release()}"


def render(results: list[Result], best: Result | None, intents: dict[str, list[str]], forms: list[str], oos_eval: list[str],
           kiwi: tuple[float, float], load_s: dict[str, float], n_groups: int, n_near: int,
           comparisons: list[tuple[str, str, tuple[float, float, float]]]) -> str:
    import sentence_transformers
    import torch

    c = CRITERIA
    emb_lines = [f"  - `{s.key}` = {s.model_id} ({s.device} {s.dtype}, 프롬프트 `{s.prompt!r}`, 로드 {load_s.get(s.key, float('nan')):.1f}초)"
                 for s in EMBEDDINGS.values()]
    lines = [
        "# Phase 0 — NLU 베이스라인 결과",
        "",
        "`uv run python bench/phase0.py`가 생성한 파일이다. 손으로 고치지 말 것.",
        "",
        f"- 환경: {machine()}, Python {platform.python_version()}, scikit-learn {sklearn.__version__}, kiwipiepy {kiwipiepy.__version__}, torch {torch.__version__}, sentence-transformers {sentence_transformers.__version__}",
        f"- 데이터: 의도 {len(intents)}개(범위 밖 포함), 템플릿 {sum(map(len, intents.values()))}개 → 거의 같은 템플릿끼리 묶어 {n_groups}묶음(글자 n-gram 유사도 ≥ {GROUP_SIMILARITY} 또는 `{GROUPING_EMBEDDING}` 임베딩 complete-link ≥ {SEMANTIC_SIMILARITY}), 지역 표기 {len(forms)}개",
        f"- 처음 보는 범위 밖 평가 문장 {len(oos_eval)}개. 그중 {n_near}개는 학습용 범위 밖 문장과 `{GROUPING_EMBEDDING}` 유사도 ≥ {NEAR_OOS}이라 '먼 문장만' 열에서 뺐다.",
        f"- 교차검증: 묶음 단위 {N_SPLITS}-fold StratifiedGroupKFold × 시드 {len(SEEDS)}개. 평가 문장은 템플릿의 `{{location}}`에 무작위 지역을 넣은 것.",
        f"- 범위 밖 규칙(모든 조합 동일): `out_of_scope` 클래스가 1위이거나, 1위 확률 < 확률 임계값이거나, 범위 안 학습 문장과의 최대 코사인 < 유사도 임계값이면 거절. 유사도 공간은 임베딩 조합은 임베딩, n-gram 조합은 TF-IDF.",
        f"- 임계값: 교차검증 예측에서 범위 밖 재현율 ≥ {c['oos_reject']}를 만족하는 쌍 중 macro-F1 최대(없으면 ⚠). **표의 macro-F1은 교차 적합 값**이다. 템플릿 묶음을 둘로 나눠 한쪽에서 고른 임계값으로 다른 쪽을 쟀다. 표의 임계값은 전체 예측으로 고른 배포용 초기값이다.",
        f"- 처음 보는 범위 밖 문장은 임계값 선택에는 쓰지 않는다. 다만 통과 판정에는 쓰므로 95% 신뢰구간(Wilson)을 함께 적는다.",
        f"- 기준: macro-F1 ≥ {c['macro_f1']}, 처음 보는 범위 밖 거절률 ≥ {c['oos_reject']}, 추론 p95 ≤ {c['p95_ms']}ms({LATENCY_ROUNDS}회 측정 중 최악값), 학습(임베딩 포함, 모델 로드 제외) ≤ {c['train_s']}초",
        f"- Kiwi: 모델 로드 {kiwi[0]:.2f}초, 단일 스레드 {kiwi[1]:,.0f}문장/초",
        "- 임베딩 모델:",
        *emb_lines,
        "",
        "## 조합별 결과",
        "",
        "| 조합 | macro-F1 (교차 적합, 평균±표준편차) | 확률 임계값 | 유사도 임계값 | 범위 밖 재현율(CV) | 처음 보는 범위 밖 거절률 [95% CI] | 먼 문장만 | 되묻기 비율 | p95 ms (최악/회차) | 첫 호출 ms | 학습 s | 통과 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: -r.f1_mean):
        warn = "" if r.choice.constrained else " ⚠"
        far = "-" if r.oos_reject_far is None else f"{r.oos_reject_far:.2f}"
        rounds = "/".join(f"{x:.1f}" for x in r.p95_rounds)
        lines.append(
            f"| `{r.setup.name}` | {r.f1_mean:.3f} ± {r.f1_std:.3f} | {r.choice.threshold:.2f}{warn} | {r.choice.sim_threshold:.2f} | {r.oos_recall_cv:.2f} "
            f"| {r.oos_reject:.2f} [{r.oos_reject_ci[0]:.2f}, {r.oos_reject_ci[1]:.2f}] | {far} | {r.clarify_rate:.2f} "
            f"| {r.p95_ms:.1f} ({rounds}) | {r.cold_ms:.1f} | {r.train_s:.2f} | {'✅' if r.passed else '❌'} |"
        )
    lines += ["", "## 조합 간 차이 (묶음 단위 paired bootstrap, 교차 적합 macro-F1)", "",
              "| A | B | A − B | 95% 구간 |", "|---|---|---|---|"]
    lines += [f"| `{a}` | `{b}` | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] |" for a, b, (d, lo, hi) in comparisons] or ["| - | - | - | - |"]
    lines.append("")
    if best is None:
        shown = max(results, key=lambda r: r.f1_mean)
        lines += ["## 결론", "", "**기준을 모두 넘은 조합이 없다.**", "",
                  f"아래 상세는 macro-F1 평균이 가장 높은 `{shown.setup.name}` 기준이다.", ""]
    else:
        shown = best
        lines += [
            "## 결론 (v1 기본값)", "",
            f"- 조합: `{best.setup.name}`",
            f"- `intent_threshold` 초기값: {best.choice.threshold:.2f}",
            f"- 범위 밖 유사도 임계값 초기값: {best.choice.sim_threshold:.2f} (모델마다 척도가 달라 모델을 바꾸면 다시 보정)",
            f"- `ambiguity_margin` 초기값: {MARGIN} (되묻기 비율 {best.clarify_rate:.2f}, 그중 정답이 2위 안 {best.clarify_top2:.2f})",
            "",
        ]
    lines += ["## 의도별 F1", "", "| 의도 | F1 |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in sorted(shown.per_intent_f1.items(), key=lambda kv: kv[1])]
    lines += ["", "## 가장 많이 헷갈린 쌍 (시드 합산)", "", "| 정답 | 예측 | 건수 |", "|---|---|---|"]
    lines += [f"| {t} | {p} | {n} |" for t, p, n in shown.confusions] or ["| - | - | 0 |"]
    lines += ["", f"## 거절하지 못한 처음 보는 범위 밖 문장 ({len(shown.oos_accepted)}개)", "", "| 문장 | 예측 의도 | 확신도 |", "|---|---|---|"]
    lines += [f"| {s} | {p} | {conf:.2f} |" for s, p, conf in shown.oos_accepted] or ["| - | - | - |"]
    lines += [
        "", "## 알려진 한계", "",
        "- 평가 문장은 템플릿에 지역을 무작위로 넣어 만들기 때문에 조사가 어색한 문장(예: \"화성시이\")이 섞인다.",
        "- 거의 같은 템플릿만 같은 묶음으로 묶는다. \"뭐 할 수 있어?\"와 \"할 줄 아는 게 뭐야\" 같은 말바꿈은 학습과 평가에 나뉘어 들어간다. 실제 사용자 입력도 빌더가 쓴 예문의 말바꿈이라는 가정이다.",
        f"- 묶음과 '먼 문장' 판정에 `{GROUPING_EMBEDDING}` 임베딩을 쓰므로, 같은 모델 조합에는 오히려 불리하게(보수적으로) 작용한다.",
        "- 조합 40개 중 최고값을 고르므로 선택 편향이 남는다. 위 bootstrap 구간으로 차이가 우연인지 판단할 것.",
        "- 지연은 단일 요청 기준이다. 동시 요청 지연은 Phase 5에서 잰다.",
        "- `delex`는 지역명과 같은 일반 명사(\"경기\", \"화성\")도 바꾼다. 엔진에서는 Phase 3의 '필요한 엔티티만 채택' 규칙으로 처리한다.",
        "- 정확히 일치하는 예문을 먼저 처리하는 규칙은 이 수치에 반영되지 않는다.",
        "- 처음 보는 범위 밖 문장 100개와 잡담 의도 보강 예문은 이 프로젝트를 위해 새로 쓴 문장이다. 실제 사용자 분포와 다를 수 있다.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    intents, forms = load_bot(BOT_NLU)
    oos_eval = load_oos_eval(OOS_EVAL)
    kiwi = Kiwi()
    setups = EMBEDDING_SETUPS if "--only-embeddings" in argv else NGRAM_SETUPS + EMBEDDING_SETUPS

    # Shared groups and the near/far split, built once with one embedding model.
    grouper = Embedder(EMBEDDINGS[GROUPING_EMBEDDING])
    groups = template_groups(intents, embed=grouper.encode)
    near = (grouper.encode(oos_eval) @ grouper.encode(intents[OOS]).T).max(axis=1) >= NEAR_OOS
    far_mask = ~near
    del grouper

    results: list[Result] = []
    load_s: dict[str, float] = {}
    embedders: dict[str, Embedder] = {}
    for setup in setups:
        key = setup.embedding
        if key is not None and key not in embedders:
            embedders.clear()  # keep at most one model in memory
            t0 = time.perf_counter()
            embedders[key] = Embedder(EMBEDDINGS[key])
            load_s[key] = time.perf_counter() - t0
        r = evaluate(setup, intents, forms, oos_eval, kiwi, embedders.get(key) if key else None, groups, far_mask)
        print(f"{setup.name:34} F1={r.f1_mean:.3f}±{r.f1_std:.3f} t={r.choice.threshold:.2f} s={r.choice.sim_threshold:.2f}{'' if r.choice.constrained else '!'} "
              f"oosCV={r.oos_recall_cv:.2f} oos={r.oos_reject:.2f} far={r.oos_reject_far:.2f} p95={r.p95_ms:.1f}ms cold={r.cold_ms:.0f}ms train={r.train_s:.2f}s {'PASS' if r.passed else 'fail'}", flush=True)
        results.append(r)
    best = recommend(results)

    ranked = sorted(results, key=lambda r: -r.f1_mean)
    anchor = best or ranked[0]
    others = [r for r in ranked if r is not anchor]
    picks = others[:2] + [max((r for r in results if r.setup.embedding is None), key=lambda r: r.f1_mean)]
    comparisons = []
    for other in {r.setup.name: r for r in picks}.values():  # Result is unhashable; dedupe by name
        comparisons.append((anchor.setup.name, other.setup.name, paired_bootstrap(anchor, other)))

    texts = [normalize(t.replace(PLACEHOLDER, "서울")) for ts in intents.values() for t in ts]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(results, best, intents, forms, oos_eval, kiwi_speed(texts), load_s,
                             len(set(groups)), int(near.sum()), comparisons), encoding="utf-8")
    print(f"report → {REPORT.relative_to(ROOT)}; recommended: {best.setup.name if best else 'NONE'}")
    return 0 if best else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
