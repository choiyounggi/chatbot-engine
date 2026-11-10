"""A deterministic stand-in for the sentence-embedding model: hashed character
1-3-grams, L2-normalized. No model download, no GPU, instant.

For tests, CI and demos only (`python -m chatcore.api --embedder hashing`). It
matches surface overlap, not meaning: never use it to serve real users.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class HashingEmbedder:
    model_id = "hashing-ngram-256"
    prompt = ""
    dims = 256

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for r, text in enumerate(texts):
            for n in (1, 2, 3):
                for i in range(len(text) - n + 1):
                    out[r, int(hashlib.md5(text[i:i + n].encode()).hexdigest(), 16) % self.dims] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-9)).astype(np.float32)
