"""Sentence embeddings, the intent classifier, and the runtime NLU model.

Phase 0 chose a frozen sentence embedding + LogisticRegression (docs/benchmarks/
phase0.md). The trained classifier is stored as plain arrays (npz + json), not
pickle, so loading a model never runs code from the file.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from chatcore.bot.schema import BotDefinition
from chatcore.nlu.entities import EntityExtractor, EntityMatch, bare, normalize
from chatcore.nlu.keyword import KeywordMatcher

OOS = "out_of_scope"
FloatArray = NDArray[np.float32]

# ---------------------------------------------------------------- embeddings


class Embedder(Protocol):
    model_id: str
    prompt: str  # prepended to every text; part of the model's identity

    def encode(self, texts: Sequence[str]) -> FloatArray:
        """L2-normalized float32 rows, one per text."""
        ...


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    prompt: str = "query: "


class SentenceEmbedder:  # pragma: no cover - needs the real model; exercised by `make train`
    """sentence-transformers on the Apple GPU (MPS, fp16) when available, else CPU (fp32).
    Not thread-safe on MPS (pytorch#197805): call it from one thread (Phase 5 batches requests)."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        mps = torch.backends.mps.is_available()
        config = config or EmbeddingConfig()
        self.config = config
        self.model_id = config.model_id
        self.prompt = config.prompt
        self.device = "mps" if mps else "cpu"
        self._model = SentenceTransformer(config.model_id, device=self.device,
                                          model_kwargs={"dtype": torch.float16 if mps else torch.float32})

    def encode(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = self._model.encode([self.config.prompt + t for t in texts], normalize_embeddings=True, batch_size=64)
        return np.asarray(vecs, dtype=np.float32)


class CachedEmbedder:
    """Remembers embeddings by text (training re-encodes the same rows many times)."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.prompt = inner.prompt
        self._cache: dict[str, FloatArray] = {}

    def encode(self, texts: Sequence[str]) -> FloatArray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, self.inner.encode(missing), strict=True))
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([self._cache[t] for t in texts])


# ---------------------------------------------------------------- classifier


@dataclass(frozen=True)
class IntentClassifier:
    classes: tuple[str, ...]
    coef: FloatArray  # (n_classes or 1, dim)
    intercept: FloatArray
    index: FloatArray  # in-scope training embeddings, for the similarity gate

    @classmethod
    def fit(cls, embeddings: FloatArray, labels: Sequence[str], c: float = 10.0, seed: int = 0) -> IntentClassifier:
        from sklearn.linear_model import LogisticRegression

        if len(embeddings) != len(labels) or len(embeddings) == 0:
            raise ValueError("need one label per embedding and at least one row")
        if len(set(labels)) < 2:
            raise ValueError("need at least two intents to train")
        clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=seed).fit(embeddings, labels)
        keep = np.array([lab != OOS for lab in labels])
        if not keep.any():
            raise ValueError("need at least one in-scope example")
        return cls(tuple(str(x) for x in clf.classes_), clf.coef_.astype(np.float32),
                   clf.intercept_.astype(np.float32), embeddings[keep].astype(np.float32))

    def predict(self, embeddings: FloatArray) -> tuple[FloatArray, FloatArray]:
        """(class probabilities like LogisticRegression.predict_proba, max cosine to in-scope rows)."""
        scores = embeddings @ self.coef.T + self.intercept
        if self.coef.shape[0] == 1:  # binary: sklearn models P(classes[1]) with a sigmoid
            p1 = 1.0 / (1.0 + np.exp(-scores[:, 0]))
            probs = np.stack([1.0 - p1, p1], axis=1)
        else:
            z = scores - scores.max(axis=1, keepdims=True)
            e = np.exp(z)
            probs = e / e.sum(axis=1, keepdims=True)
        sims = (embeddings @ self.index.T).max(axis=1)
        return probs.astype(np.float32), sims.astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, classes=np.array(self.classes), coef=self.coef, intercept=self.intercept, index=self.index)

    @classmethod
    def load(cls, path: Path) -> IntentClassifier:
        with np.load(path, allow_pickle=False) as data:
            return cls(tuple(str(x) for x in data["classes"]), data["coef"], data["intercept"], data["index"])


# ---------------------------------------------------------------- runtime model


@dataclass(frozen=True)
class IntentScore:
    intent: str
    score: float


@dataclass(frozen=True)
class NluResult:
    text: str
    intents: tuple[IntentScore, ...]  # best first; exact matches have score 1.0
    similarity: float  # max cosine to an in-scope training example (1.0 for exact matches)
    entities: tuple[EntityMatch, ...]
    matched_by: str  # "exact" | "keyword" | "classifier"

    @property
    def top(self) -> IntentScore:
        return self.intents[0]

    @property
    def second(self) -> IntentScore | None:
        return self.intents[1] if len(self.intents) > 1 else None

    def to_log(self) -> dict[str, Any]:
        return {
            "intents": [[s.intent, round(s.score, 4)] for s in self.intents[:3]],
            "similarity": round(self.similarity, 4),
            "matched_by": self.matched_by,
            "entities": [{"entity": e.entity, "value": e.value, "text": e.text} for e in self.entities],
        }


class ModelMismatch(ValueError):
    """The stored model was trained for another bot version or embedding model."""


@dataclass(frozen=True)
class Thresholds:
    """The OOS-decision thresholds a model was evaluated at (bot/schema.py Settings.similarity_level,
    docs/engine-core-plan.md "빌더 단계 E13"). `level` is None for the document's own numbers
    (no level asked, or deploy training found no out_of_scope examples to search against)."""

    intent_threshold: float
    similarity_threshold: float
    ambiguity_margin: float
    level: str | None = None


def with_thresholds(bot: BotDefinition, thresholds: Thresholds) -> BotDefinition:
    """`bot` with its settings' thresholds replaced (the version is the document's content hash,
    so it is unaffected by this in-memory override)."""
    settings = bot.settings.model_copy(update={
        "intent_threshold": thresholds.intent_threshold,
        "similarity_threshold": thresholds.similarity_threshold,
        "ambiguity_margin": thresholds.ambiguity_margin,
    })
    return bot.model_copy(update={"settings": settings})


@dataclass
class NluModel:
    bot: BotDefinition
    classifier: IntentClassifier
    embedder: Embedder
    extractor: EntityExtractor
    today: Callable[[], dt.date] = dt.date.today
    _exact: dict[str, str] = field(init=False, default_factory=dict)
    _keyword: dict[tuple[str, ...], str] = field(init=False, default_factory=dict)
    _matcher: KeywordMatcher | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        for intent, spec in self.bot.intents.items():
            for example in spec.examples:
                if "{" not in example:  # templates are not literal sentences
                    self._exact.setdefault(bare(example), intent)
        if self.bot.settings.keyword_match:
            self._matcher = KeywordMatcher()
            for intent, spec in self.bot.intents.items():
                for example in spec.examples:
                    if "{" not in example:
                        key = self._matcher.morphemes(example)
                        if key:
                            self._keyword.setdefault(key, intent)

    def parse(self, text: str) -> NluResult:
        entities = tuple(self.extractor.extract(text, self.today()))
        exact = self._exact.get(bare(text))
        if exact is not None:
            return NluResult(text, (IntentScore(exact, 1.0),), 1.0, entities, "exact")
        if self._matcher is not None:
            key = self._matcher.morphemes(text)
            keyword = self._keyword.get(key) if key else None
            if keyword is not None:
                return NluResult(text, (IntentScore(keyword, 1.0),), 1.0, entities, "keyword")
        probs, sims = self.classifier.predict(self.embedder.encode([normalize(text)]))  # as in training
        order = np.argsort(-probs[0])
        intents = tuple(IntentScore(self.classifier.classes[i], float(probs[0, i])) for i in order)
        return NluResult(text, intents, float(sims[0]), entities, "classifier")

    @staticmethod
    def thresholds(meta: dict[str, Any], bot: BotDefinition) -> Thresholds:
        """The thresholds the stored model was evaluated at: meta.json's `thresholds` (deploy
        training, docs/engine-core-plan.md "빌더 단계 E13") when present, else the bot document's
        own settings (test builds, and models trained before this field existed)."""
        raw = meta.get("thresholds")
        if not isinstance(raw, dict):
            s = bot.settings
            return Thresholds(s.intent_threshold, s.similarity_threshold, s.ambiguity_margin, None)
        return Thresholds(float(raw["intent_threshold"]), float(raw["similarity_threshold"]),
                           float(raw["ambiguity_margin"]), raw.get("level"))

    @classmethod
    def load(cls, bot: BotDefinition, directory: Path, embedder: Embedder, extractor: EntityExtractor | None = None,
             today: Callable[[], dt.date] = dt.date.today) -> NluModel:
        meta = cls.check(bot, directory, embedder)
        served = with_thresholds(bot, cls.thresholds(meta, bot))
        return cls(served, IntentClassifier.load(directory / "classifier.npz"), embedder,
                   extractor or EntityExtractor(bot), today)

    @staticmethod
    def check(bot: BotDefinition, directory: Path, embedder: Embedder) -> dict[str, Any]:
        """The model's meta.json if `directory` holds a model trained for this bot version and
        embedder; ModelMismatch otherwise. Cheap: nothing is loaded."""
        meta_file = directory / "meta.json"
        if not meta_file.is_file() or not (directory / "classifier.npz").is_file():
            raise ModelMismatch(f"no trained model in {directory}; run `make train`")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("bot_id") != bot.id or meta.get("bot_version") != bot.version:
            raise ModelMismatch(f"model in {directory} is for {meta.get('bot_id')}@{meta.get('bot_version')}, "
                                f"not {bot.id}@{bot.version}; run `make train`")
        if meta.get("embedding_model") != embedder.model_id or meta.get("embedding_prompt") != embedder.prompt:
            raise ModelMismatch(f"model was trained with {meta.get('embedding_model')} / prompt "
                                f"{meta.get('embedding_prompt')!r}, not {embedder.model_id} / {embedder.prompt!r}")
        return dict(meta)
