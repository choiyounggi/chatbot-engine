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
from chatcore.nlu.model import OOS, CachedEmbedder, Embedder, EmbeddingConfig, IntentClassifier, SentenceEmbedder

PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")
PER_TEMPLATE = 3  # fillings per template
CAP_RATIO = 2  # rows per intent <= CAP_RATIO * templates (keeps entity-rich intents from dominating)
GROUP_SIMILARITY = 0.75  # char n-gram cosine for "near-duplicate" templates (kept on one side of a split)
FOLDS = 5
SEED = 0


class TrainingError(ValueError):
    pass


# ---------------------------------------------------------------- data


def entity_forms(bot: BotDefinition) -> dict[str, list[str]]:
    """Surface forms (values + synonyms) per dictionary entity."""
    return {eid: sorted({f for c, syns in ent.values.items() for f in [c, *syns]})
            for eid, ent in bot.entities.items() if isinstance(ent, DictionaryEntity)}


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


def decide(probs: np.ndarray, sims: np.ndarray, classes: Sequence[str], bot: BotDefinition) -> list[str]:
    """The interpreter's rules 3/5/6 as one label per input: the accepted intent,
    CLARIFY when the top two in-scope intents are within ambiguity_margin, or
    out_of_scope when rejected. (Rules 1/2/4 depend on dialogue state.)"""
    s = bot.settings
    out = []
    for p, sim in zip(probs, sims, strict=True):
        order = np.argsort(-p)
        top, second = int(order[0]), int(order[1]) if len(order) > 1 else None
        if classes[top] == OOS or p[top] < s.intent_threshold or sim < s.similarity_threshold:
            out.append(OOS)
        elif second is not None and classes[second] != OOS and p[top] - p[second] < s.ambiguity_margin:
            out.append(CLARIFY)
        else:
            out.append(classes[top])
    return out


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
    oos_recall: float | None
    clarify_rate: float  # share of in-scope inputs that would get "A or B?"
    confusions: list[tuple[str, str, int]]
    determinism_max_diff: float | None = None
    warnings: list[str] = field(default_factory=list)
    train_seconds: float = 0.0


def evaluate(bot: BotDefinition, embedder: Embedder) -> Report:
    from collections import Counter

    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    intents = {k: v.examples for k, v in bot.intents.items()}
    if len(intents) < 2:
        raise TrainingError("need at least two intents")
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
    y_true: list[str] = []
    y_pred: list[str] = []
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for train_idx, test_idx in splitter.split(concrete, labels, groups=groups):
        fold: dict[str, list[str]] = {}
        for i in train_idx:
            fold.setdefault(labels[i], []).append(templates[i][1])
        rows, row_labels = training_rows(fold, forms)
        clf = IntentClassifier.fit(embedder.encode(rows), row_labels)
        probs, sims = clf.predict(embedder.encode([concrete[i] for i in test_idx]))
        y_pred += decide(probs, sims, clf.classes, bot)
        y_true += [labels[i] for i in test_idx]
    names = sorted(set(y_true))  # CLARIFY is never a true label: it only counts as a miss
    per = f1_score(y_true, y_pred, labels=names, average=None, zero_division=0)
    in_scope = [p for t, p in zip(y_true, y_pred, strict=True) if t != OOS]
    oos_rows = [p for t, p in zip(y_true, y_pred, strict=True) if t == OOS]
    return Report(
        bot_id=bot.id, bot_version=bot.version, embedding_model=embedder.model_id,
        rows=len(training_rows(intents, forms)[0]), templates=len(templates), groups=len(set(groups)),
        macro_f1=float(f1_score(y_true, y_pred, labels=names, average="macro", zero_division=0)),
        per_intent_f1={n: float(v) for n, v in zip(names, per, strict=True)},
        oos_recall=(sum(p == OOS for p in oos_rows) / len(oos_rows)) if oos_rows else None,
        clarify_rate=(sum(p == CLARIFY for p in in_scope) / len(in_scope)) if in_scope else 0.0,
        confusions=[(t, p, n) for (t, p), n in Counter((t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p)
                    .most_common(5)],
        warnings=validate(bot).warnings,
    )


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
    if len(bot.intents) < 2:
        raise TrainingError("need at least two intents")
    _, clf = _fit(bot, embedder)
    meta: dict[str, object] = {"bot_id": bot.id, "bot_version": bot.version, "embedding_model": embedder.model_id,
            "embedding_prompt": embedder.prompt, "macro_f1": None, "test_build": True}
    return _save(clf, out_root / bot.id / bot.version, meta)


def _fit(bot: BotDefinition, embedder: Embedder) -> tuple[list[str], IntentClassifier]:
    rows, labels = training_rows({k: v.examples for k, v in bot.intents.items()}, entity_forms(bot))
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
    lines = [
        f"bot {report.bot_id}@{report.bot_version}  model {report.embedding_model}",
        f"templates {report.templates} ({report.groups} near-duplicate groups) -> training rows {report.rows}",
        f"cross-validated macro-F1 {report.macro_f1:.3f}, clarify rate {report.clarify_rate:.2f}"
        + (f", out-of-scope recall {report.oos_recall:.2f}" if report.oos_recall is not None else ""),
        "per-intent F1:",
        *[f"  {name:20} {f1:.3f}" for name, f1 in sorted(report.per_intent_f1.items(), key=lambda kv: kv[1])],
        "most confused (true -> predicted):",
        *([f"  {t} -> {p}: {n}" for t, p, n in report.confusions] or ["  none"]),
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
