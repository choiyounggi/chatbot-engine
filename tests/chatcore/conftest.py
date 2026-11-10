import hashlib
import threading
from pathlib import Path

import numpy as np
import pytest
from kiwipiepy import Kiwi

from chatcore.bot.loader import load_bot
from chatcore.bot.schema import BotDefinition
from chatcore.dialogue.values import ActionRegistry, StubAction

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "bots/sample"
SAMPLE_ACTIONS = ("get_weather", "get_temperature", "now")


_KIWIS: dict[frozenset, tuple[Kiwi, threading.Lock]] = {}
_KIWIS_GUARD = threading.Lock()


class PooledKiwi:
    """Test stand-in for kiwipiepy.Kiwi. A real Kiwi() takes ~0.3 s here and several times that on
    CI, and the suite builds the same bot dozens of times, so extractors that add the same user
    words share one real instance (created on first use, one lock per instance)."""

    def __init__(self) -> None:
        self._words: list[tuple] = []
        self._real: tuple[Kiwi, threading.Lock] | None = None

    def add_user_word(self, *args) -> bool:
        assert self._real is None, "words must be added before the first tokenize"
        self._words.append(args)
        return True

    def tokenize(self, text: str, **kwargs):
        if self._real is None:
            key = frozenset(self._words)
            with _KIWIS_GUARD:
                if key not in _KIWIS:
                    kiwi = Kiwi()
                    for args in self._words:
                        kiwi.add_user_word(*args)
                    _KIWIS[key] = (kiwi, threading.Lock())
            self._real = _KIWIS[key]
        kiwi, lock = self._real
        with lock:
            return kiwi.tokenize(text, **kwargs)


@pytest.fixture(autouse=True)
def pooled_kiwi(monkeypatch):
    monkeypatch.setattr("chatcore.nlu.entities.Kiwi", PooledKiwi)


@pytest.fixture(scope="session")
def sample_bot() -> BotDefinition:
    return load_bot(SAMPLE, actions=SAMPLE_ACTIONS)


@pytest.fixture
def stubs() -> dict[str, StubAction]:
    return {
        "get_weather": StubAction({"condition": "맑음"}),
        "get_temperature": StubAction({"celsius": 21}),
        "now": StubAction({"time": "오후 3시 5분"}),
    }


@pytest.fixture
def registry(stubs) -> ActionRegistry:
    reg = ActionRegistry()
    for name, stub in stubs.items():
        reg.register(name, stub)
    return reg


class FakeEmbedder:
    """Deterministic stand-in for a sentence-embedding model: hashed char 1-3-grams, L2-normalized."""

    model_id = "fake-ngram-256"
    prompt = ""

    def __init__(self):
        self.calls = 0
        self.last = None

    def encode(self, texts):
        self.calls += 1
        self.last = list(texts)
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for r, t in enumerate(texts):
            for n in (1, 2, 3):
                for i in range(len(t) - n + 1):
                    out[r, int(hashlib.md5(t[i:i + n].encode()).hexdigest(), 16) % 256] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


@pytest.fixture(scope="session")
def fake_embedder():
    return FakeEmbedder()
