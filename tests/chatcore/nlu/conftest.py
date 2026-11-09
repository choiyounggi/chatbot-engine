import hashlib

import numpy as np
import pytest

from chatcore.nlu.entities import EntityExtractor


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


@pytest.fixture(scope="session")
def extractor(sample_bot):
    return EntityExtractor(sample_bot)
