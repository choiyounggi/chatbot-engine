"""Train and evaluate a bot's intent classifier; save it under artifacts/<bot>/<version>/.

    uv run python -m chatcore.nlu.train bots/sample [--out artifacts] [--min-f1 0.90]

Steps: data check -> template expansion -> grouped cross-validation with the
bot's own thresholds (per-intent F1 report) -> fit on everything -> save ->
determinism check. Exits 1 if macro-F1 is below --min-f1.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import shutil
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from chatcore.bot.loader import BotError, load_bot, validate
from chatcore.bot.schema import BotDefinition, DictionaryEntity
from chatcore.nlu.entities import normalize
from chatcore.nlu.model import (
    OOS,
    CachedEmbedder,
    Embedder,
    EmbeddingConfig,
    FloatArray,
    IntentClassifier,
    SentenceEmbedder,
    Thresholds,
)

PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")
PER_TEMPLATE = 3  # fillings per template
CAP_RATIO = 2  # rows per intent <= CAP_RATIO * templates (keeps entity-rich intents from dominating)
GROUP_SIMILARITY = 0.75  # char n-gram cosine for "near-duplicate" templates (kept on one side of a split)
FOLDS = 5
SEED = 0

# Builder's 5-step similarity dial (bot/schema.py Settings.similarity_level): a target out-of-scope
# rejection rate, searched for by _search_thresholds over this grid.
LEVEL_TARGETS: dict[str, float] = {
    "low": 0.80, "normal": 0.90, "somewhat_high": 0.95, "high": 0.97, "very_high": 0.99,
}
INTENT_GRID = tuple(round(x, 2) for x in np.arange(0.0, 0.90 + 1e-9, 0.05))
SIMILARITY_GRID = tuple(round(x, 2) for x in np.arange(0.0, 0.95 + 1e-9, 0.01))

# Measured on the sample bot with the real embedding model (dragonkue/snowflake-arctic-embed-l-
# v2.0-ko, docs/engine-core-plan.md "빌더 단계 E-B4"): of the bot's 36 in-scope intent pairs, the
# top 9 (0.762-0.946, led by temperature<->weather and bye<->greeting) are genuinely related or
# share templated phrasing; similarity falls off smoothly from there (0.70 at rank 11, 0.54 at
# rank 23), so 0.75 flags the clearly confusable pairs without flagging most of the 36.
CONFUSABLE_THRESHOLD = 0.75


class TrainingError(ValueError):
    pass


# ---------------------------------------------------------------- data


def entity_forms(bot: BotDefinition) -> dict[str, list[str]]:
    """Surface forms (values + synonyms) per dictionary entity."""
    return {eid: sorted({f for c, syns in ent.values.items() for f in [c, *syns]})
            for eid, ent in bot.entities.items() if isinstance(ent, DictionaryEntity)}


def classifier_intents(bot: BotDefinition) -> dict[str, list[str]]:
    """Examples of the intents trained into the classifier and cross-validated: every intent
    except those marked exact_only (bot/schema.py Intent.exact_only), which only ever match by
    exact text or keyword match."""
    return {k: v.examples for k, v in bot.intents.items() if not v.exact_only}


def fill(template: str, rng: random.Random, forms: dict[str, list[str]]) -> str:
    def sub(m: re.Match[str]) -> str:
        if m.group(1) not in forms:
            raise TrainingError(f"placeholder {{{m.group(1)}}} is not a dictionary entity: {template!r}")
        return rng.choice(forms[m.group(1)])

    return normalize(PLACEHOLDER.sub(sub, template))


def expand(templates: Sequence[str], forms: dict[str, list[str]], seed: int = SEED) -> list[str]:
    """Every template once, then extra fillings round-robin up to CAP_RATIO x templates."""
    rng = random.Random(seed)
    variants: list[list[str]] = []
    for t in templates:
        if not PLACEHOLDER.search(t):
            variants.append([normalize(t)])
            continue
        tries = [fill(t, rng, forms) for _ in range(PER_TEMPLATE * 3)]
        variants.append(list(dict.fromkeys(tries))[:PER_TEMPLATE])
    rows = [v[0] for v in variants]
    limit = CAP_RATIO * len(templates)
    for depth in range(1, PER_TEMPLATE):
        for v in variants:
            if len(rows) >= limit:
                return rows
            if depth < len(v):
                rows.append(v[depth])
    return rows


def training_rows(intents: dict[str, list[str]], forms: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for i, name in enumerate(sorted(intents)):
        rows = expand(intents[name], forms, seed=SEED + i)
        texts += rows
        labels += [name] * len(rows)
    return texts, labels


def near_duplicate_groups(templates: Sequence[str], threshold: float = GROUP_SIMILARITY) -> list[int]:
    """Single-link clusters of templates by char 2-3-gram cosine (raw tf)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not templates:
        return []
    texts = [PLACEHOLDER.sub(" ", t) for t in templates]
    sim = cosine_similarity(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), use_idf=False).fit_transform(texts))
    parent = list(range(len(templates)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(sim >= threshold, k=1)), strict=True):
        parent[root(int(i))] = root(int(j))
    ids: dict[int, int] = {}
    return [ids.setdefault(root(i), len(ids)) for i in range(len(templates))]


# ---------------------------------------------------------------- evaluation


CLARIFY = "(clarify)"  # the interpreter would ask "A or B?" instead of starting a scenario


def _decide_batch(probs: np.ndarray, sims: np.ndarray, classes: Sequence[str], intent_threshold: float,
                   similarity_threshold: float, ambiguity_margin: float) -> list[str]:
    """The interpreter's rules 3/5/6, vectorized, for one explicit threshold triple (used by the
    grid search below; `decide()` is the same rule with a bot's configured thresholds)."""
    if len(probs) == 0:
        return []
    classes_arr = np.array(classes)
    order = np.argsort(-probs, axis=1)
    rows = np.arange(len(probs))
    top_idx = order[:, 0]
    top_class = classes_arr[top_idx]
    top_prob = probs[rows, top_idx]
    rejected = (top_class == OOS) | (top_prob < intent_threshold) | (sims < similarity_threshold)
    out = np.where(rejected, OOS, top_class)
    if order.shape[1] > 1:
        second_idx = order[:, 1]
        second_class = classes_arr[second_idx]
        second_prob = probs[rows, second_idx]
        ambiguous = (~rejected) & (second_class != OOS) & (top_prob - second_prob < ambiguity_margin)
        out = np.where(ambiguous, CLARIFY, out)
    return [str(x) for x in out]


def decide(probs: np.ndarray, sims: np.ndarray, classes: Sequence[str], bot: BotDefinition) -> list[str]:
    """The interpreter's rules 3/5/6 as one label per input: the accepted intent,
    CLARIFY when the top two in-scope intents are within ambiguity_margin, or
    out_of_scope when rejected. (Rules 1/2/4 depend on dialogue state.)"""
    s = bot.settings
    return _decide_batch(probs, sims, classes, s.intent_threshold, s.similarity_threshold, s.ambiguity_margin)


@dataclass(frozen=True)
class ConfusablePair:
    """Two in-scope intents whose closest filled example pair is this similar (builder plan §2.3:
    "의도 간 헷갈림 경고"); a hint to add distinguishing examples, not a training-time error."""

    a: str
    b: str
    similarity: float
    example_a: str
    example_b: str


def confusable_intents(bot: BotDefinition, embedder: Embedder, threshold: float = CONFUSABLE_THRESHOLD
                        ) -> list[ConfusablePair]:
    """In-scope intent pairs (every intent except out_of_scope) whose closest example pair (one
    random filling per template) has cosine similarity >= threshold, sorted by similarity
    descending, at most 10. Independent of cross-validation: works for exact_only intents and for
    test builds (one example per intent) too."""
    forms = entity_forms(bot)
    rng = random.Random(SEED)
    names = sorted(n for n in bot.intents if n != OOS)
    filled = {name: [(t, fill(t, rng, forms)) for t in bot.intents[name].examples] for name in names}
    texts = [concrete for name in names for _, concrete in filled[name]]
    if not texts or len(names) < 2:
        return []
    vectors = embedder.encode(texts)
    spans: dict[str, slice] = {}
    offset = 0
    for name in names:
        spans[name] = slice(offset, offset + len(filled[name]))
        offset += len(filled[name])

    pairs: list[ConfusablePair] = []
    for i, a in enumerate(names):
        va = vectors[spans[a]]
        for b in names[i + 1:]:
            vb = vectors[spans[b]]
            sim = va @ vb.T
            ia, ib = divmod(int(np.argmax(sim)), sim.shape[1])
            best = float(sim[ia, ib])
            if best >= threshold:
                pairs.append(ConfusablePair(a, b, best, filled[a][ia][0], filled[b][ib][0]))
    return sorted(pairs, key=lambda p: -p.similarity)[:10]


FoldPrediction = tuple[FloatArray, FloatArray, tuple[str, ...], list[str]]  # probs, sims, classes, true labels


def _search_thresholds(folds: Sequence[FoldPrediction], level: str, ambiguity_margin: float
                        ) -> tuple[Thresholds, str | None]:
    """The (intent_threshold, similarity_threshold) pair that best meets `level`'s target
    out-of-scope rejection, reusing the cross-validation predictions `evaluate` already made (no
    second CV pass). Selection: highest in-scope recall among pairs whose rejection >= target;
    ties -> higher rejection, then lower similarity_threshold, then lower intent_threshold.
    Nothing meets the target -> the pair with the highest rejection, plus a warning."""
    target = LEVEL_TARGETS[level]
    best_meeting: tuple[float, float, float, float] | None = None
    meeting_pair: tuple[float, float] | None = None
    best_fallback: tuple[float, float, float, float] | None = None
    fallback_pair = (INTENT_GRID[0], SIMILARITY_GRID[0])
    fallback_rejection = 0.0

    for it in INTENT_GRID:
        for st in SIMILARITY_GRID:
            y_true: list[str] = []
            y_pred: list[str] = []
            for probs, sims, classes, true in folds:
                y_true += true
                y_pred += _decide_batch(probs, sims, classes, it, st, ambiguity_margin)
            oos_rows = [p for t, p in zip(y_true, y_pred, strict=True) if t == OOS]
            in_rows = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if t != OOS]
            rejection = (sum(p == OOS for p in oos_rows) / len(oos_rows)) if oos_rows else 0.0
            recall = (sum(p == t for t, p in in_rows) / len(in_rows)) if in_rows else 0.0

            fallback_key = (rejection, recall, -st, -it)
            if best_fallback is None or fallback_key > best_fallback:
                best_fallback, fallback_pair, fallback_rejection = fallback_key, (it, st), rejection

            if rejection >= target:
                meeting_key = (recall, rejection, -st, -it)
                if best_meeting is None or meeting_key > best_meeting:
                    best_meeting, meeting_pair = meeting_key, (it, st)

    if meeting_pair is not None:
        it, st = meeting_pair
        return Thresholds(it, st, ambiguity_margin, level), None
    it, st = fallback_pair
    warning = (f"similarity_level {level!r} targets out-of-scope rejection >= {target:.2f}; no threshold "
               f"pair reached it, using the highest achieved ({fallback_rejection:.2f})")
    return Thresholds(it, st, ambiguity_margin, level), warning


@dataclass
class Report:
    bot_id: str
    bot_version: str
    embedding_model: str
    rows: int
    templates: int
    groups: int
    macro_f1: float
    per_intent_f1: dict[str, float]
    oos_recall: float | None  # out-of-scope rejection rate, with `thresholds` applied
    clarify_rate: float  # share of in-scope inputs that would get "A or B?"
    in_scope_recall: float | None  # share of in-scope rows decided as their true intent
    confusions: list[tuple[str, str, int]]
    thresholds: Thresholds
    confusable: list[ConfusablePair]
    determinism_max_diff: float | None = None
    warnings: list[str] = field(default_factory=list)
    train_seconds: float = 0.0


def evaluate(bot: BotDefinition, embedder: Embedder) -> Report:
    from collections import Counter

    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    intents = classifier_intents(bot)
    if len(intents) < 2:
        raise TrainingError("need at least two intents to train (exact_only intents do not count)")
    forms = entity_forms(bot)
    templates = [(name, t) for name in sorted(intents) for t in intents[name]]
    labels = [name for name, _ in templates]
    groups: list[int] = []
    for name in sorted(intents):
        offset = max(groups, default=-1) + 1
        groups += [offset + g for g in near_duplicate_groups(intents[name])]
    small = [n for n, c in Counter(labels).items() if c < FOLDS]
    if small:
        raise TrainingError(f"intents need at least {FOLDS} examples for evaluation: {small}")

    rng = random.Random(SEED)
    concrete = [fill(t, rng, forms) for _, t in templates]
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds: list[FoldPrediction] = []
    for train_idx, test_idx in splitter.split(concrete, labels, groups=groups):
        fold: dict[str, list[str]] = {}
        for i in train_idx:
            fold.setdefault(labels[i], []).append(templates[i][1])
        rows, row_labels = training_rows(fold, forms)
        clf = IntentClassifier.fit(embedder.encode(rows), row_labels)
        probs, sims = clf.predict(embedder.encode([concrete[i] for i in test_idx]))
        folds.append((probs, sims, clf.classes, [labels[i] for i in test_idx]))

    extra_warnings: list[str] = []
    level = bot.settings.similarity_level
    if level is not None and OOS not in intents:
        extra_warnings.append(f"similarity_level {level!r} is set but the bot has no out_of_scope "
                               "examples to search thresholds against; using the configured thresholds")
        level = None
    if level is not None:
        thresholds, search_warning = _search_thresholds(folds, level, bot.settings.ambiguity_margin)
        if search_warning is not None:
            extra_warnings.append(search_warning)
    else:
        s = bot.settings
        thresholds = Thresholds(s.intent_threshold, s.similarity_threshold, s.ambiguity_margin, None)

    y_true: list[str] = []
    y_pred: list[str] = []
    for probs, sims, classes, true in folds:
        y_true += true
        y_pred += _decide_batch(probs, sims, classes, thresholds.intent_threshold,
                                 thresholds.similarity_threshold, thresholds.ambiguity_margin)

    names = sorted(set(y_true))  # CLARIFY is never a true label: it only counts as a miss
    per = f1_score(y_true, y_pred, labels=names, average=None, zero_division=0)
    in_scope = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if t != OOS]
    oos_rows = [p for t, p in zip(y_true, y_pred, strict=True) if t == OOS]
    return Report(
        bot_id=bot.id, bot_version=bot.version, embedding_model=embedder.model_id,
        rows=len(training_rows(intents, forms)[0]), templates=len(templates), groups=len(set(groups)),
        macro_f1=float(f1_score(y_true, y_pred, labels=names, average="macro", zero_division=0)),
        per_intent_f1={n: float(v) for n, v in zip(names, per, strict=True)},
        oos_recall=(sum(p == OOS for p in oos_rows) / len(oos_rows)) if oos_rows else None,
        clarify_rate=(sum(p == CLARIFY for t, p in in_scope) / len(in_scope)) if in_scope else 0.0,
        in_scope_recall=(sum(p == t for t, p in in_scope) / len(in_scope)) if in_scope else None,
        confusions=[(t, p, n) for (t, p), n in Counter((t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p)
                    .most_common(5)],
        thresholds=thresholds,
        confusable=confusable_intents(bot, embedder),
        warnings=validate(bot).warnings + extra_warnings + [
            f"keyword match: {' '.join(k)!r} is shared by {', '.join(owners)}; it matches none of them"
            for k, owners in keyword_collisions(bot)],
    )


def keyword_collisions(bot: BotDefinition) -> list[tuple[str, list[str]]]:
    """With keyword_match on: core-morpheme keys more than one intent's examples reduce to
    ("배송 조회해요" / "배송 조회할래요"), as (key words, intents). Such a key matches none of them."""
    if not bot.settings.keyword_match:
        return []
    from chatcore.nlu.keyword import KeywordMatcher
    from chatcore.nlu.model import keyword_keys

    keys = sorted(keyword_keys(bot, KeywordMatcher()).items())
    return [(" ".join(k), owners) for k, owners in keys if len(owners) > 1]


# ---------------------------------------------------------------- train + save


def train(bot: BotDefinition, embedder: Embedder, out_root: Path, fresh: Embedder | None = None,
          min_f1: float = 0.0) -> tuple[Path | None, Report]:
    """Evaluate; if macro-F1 >= min_f1, fit on all data and save to
    out_root/<bot>/<version>/ (returned), else save nothing (None). `fresh`
    (an uncached embedder) is used for the determinism check when given."""
    report = evaluate(bot, embedder)
    if report.macro_f1 < min_f1:
        return None, report
    start = time.perf_counter()
    rows, clf = _fit(bot, embedder)
    report.train_seconds = time.perf_counter() - start
    if fresh is not None:
        sample = rows[:16]
        report.determinism_max_diff = float(np.abs(fresh.encode(sample) - fresh.encode(sample)).max())
    return _save(clf, out_root / bot.id / bot.version, {**asdict(report), "embedding_prompt": embedder.prompt}), report


def fit_only(bot: BotDefinition, embedder: Embedder, out_root: Path) -> Path:
    """Fit on all data without evaluating (no cross-validation, no F1 bar) and save to
    out_root/<bot>/<version>/. For builder test builds: works with one example per intent."""
    if len(classifier_intents(bot)) < 2:
        raise TrainingError("need at least two intents to train (exact_only intents do not count)")
    _, clf = _fit(bot, embedder)
    s = bot.settings
    thresholds = Thresholds(s.intent_threshold, s.similarity_threshold, s.ambiguity_margin, None)
    meta: dict[str, object] = {"bot_id": bot.id, "bot_version": bot.version, "embedding_model": embedder.model_id,
            "embedding_prompt": embedder.prompt, "macro_f1": None, "test_build": True,
            "thresholds": asdict(thresholds)}
    return _save(clf, out_root / bot.id / bot.version, meta)


def _fit(bot: BotDefinition, embedder: Embedder) -> tuple[list[str], IntentClassifier]:
    rows, labels = training_rows(classifier_intents(bot), entity_forms(bot))
    vectors = embedder.encode(rows)
    try:
        return rows, IntentClassifier.fit(vectors, labels)
    except ValueError as exc:  # e.g. only out_of_scope examples: the bot's data, not a bug
        raise TrainingError(str(exc)) from exc


def _save(clf: IntentClassifier, directory: Path, meta: dict[str, object]) -> Path:
    """Write a sibling directory and swap it in, so a reader (a deploy, a server start) never
    sees half-written files when the same version is trained again."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}")
    staging.mkdir()
    old = directory.with_name(f".{directory.name}.old.{uuid.uuid4().hex}")
    try:
        clf.save(staging / "classifier.npz")
        meta = {**meta, "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
        (staging / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if directory.exists():
            directory.rename(old)
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)
    return directory


def format_report(report: Report, directory: Path | None) -> str:
    th = report.thresholds
    lines = [
        f"bot {report.bot_id}@{report.bot_version}  model {report.embedding_model}",
        f"templates {report.templates} ({report.groups} near-duplicate groups) -> training rows {report.rows}",
        f"thresholds: intent={th.intent_threshold:.2f} similarity={th.similarity_threshold:.2f} "
        f"margin={th.ambiguity_margin:.2f} (level={th.level})",
        f"cross-validated macro-F1 {report.macro_f1:.3f}, clarify rate {report.clarify_rate:.2f}"
        + (f", out-of-scope recall {report.oos_recall:.2f}" if report.oos_recall is not None else "")
        + (f", in-scope recall {report.in_scope_recall:.2f}" if report.in_scope_recall is not None else ""),
        "per-intent F1:",
        *[f"  {name:20} {f1:.3f}" for name, f1 in sorted(report.per_intent_f1.items(), key=lambda kv: kv[1])],
        "most confused (true -> predicted):",
        *([f"  {t} -> {p}: {n}" for t, p, n in report.confusions] or ["  none"]),
        "confusable intents:",
        *([f"  {c.a} <-> {c.b}: {c.similarity:.2f} ({c.example_a!r} / {c.example_b!r})" for c in report.confusable]
          or ["  none"]),
        *[f"warning: {w}" for w in report.warnings],
        f"final fit {report.train_seconds:.1f}s; saved to {directory}" if directory else "NOT SAVED",
    ]
    if report.determinism_max_diff is not None:
        lines.append(f"determinism: same input encoded twice, max |diff| = {report.determinism_max_diff:.2e}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("bot", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-f1", type=float, default=0.90)
    args = parser.parse_args(argv)
    try:
        bot = load_bot(args.bot)
    except BotError as exc:
        print(exc, file=sys.stderr)
        return 2
    fresh = SentenceEmbedder(EmbeddingConfig())
    try:
        directory, report = train(bot, CachedEmbedder(fresh), args.out, fresh=fresh, min_f1=args.min_f1)
    except TrainingError as exc:
        print(f"cannot train: {exc}", file=sys.stderr)
        return 2
    print(format_report(report, directory))
    if directory is None:
        print(f"FAIL: macro-F1 {report.macro_f1:.3f} < {args.min_f1}; model not saved", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
