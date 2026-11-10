"""Serving helpers: one worker thread for the embedding model, with batching.

PyTorch MPS can crash when two threads run tensor ops at once
(pytorch/pytorch#197805), so every model call goes through one worker thread.
Requests waiting while the model runs are batched into its next call
(max_wait_ms=0: no artificial wait; measured in docs/benchmarks/concurrency.md).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future

import numpy as np
from numpy.typing import NDArray

from chatcore.nlu.model import Embedder

log = logging.getLogger(__name__)

Item = tuple[str, Future[NDArray[np.float32]]]


class MicroBatcher:
    """Runs `fn(list[str]) -> array` on a single worker thread, batching
    requests that arrive within max_wait_ms (at most max_batch per call).

    Two lanes: `submit(text)` (chat) and `submit(text, bulk=True)` (training).
    When both wait they take turns, one batch each, and bulk batches hold at most
    bulk_batch texts: a chat request waits for one small bulk batch at most, and
    training still finishes under a steady stream of chat (strict priority would
    starve it). Lanes are never mixed in one call."""

    def __init__(self, fn: Callable[[list[str]], NDArray[np.float32]], max_batch: int = 16,
                 max_wait_ms: float = 2.0, keepalive_ms: float | None = None,
                 keepalive_text: str = "keepalive", bulk_batch: int = 4) -> None:
        """keepalive_ms: when idle that long, call fn([keepalive_text]) on the worker thread
        to keep the GPU clocked up (Apple Silicon slows the first request after a pause)."""
        if (max_batch < 1 or bulk_batch < 1 or max_wait_ms < 0
                or (keepalive_ms is not None and keepalive_ms <= 0)):
            raise ValueError("max_batch and bulk_batch must be >= 1, max_wait_ms >= 0 and keepalive_ms > 0")
        self.keepalive = keepalive_ms / 1000 if keepalive_ms is not None else None
        self.keepalive_text = keepalive_text
        self.keepalive_calls = 0
        self.keepalive_failures = 0
        self.fn, self.max_batch, self.bulk_batch, self.max_wait = fn, max_batch, bulk_batch, max_wait_ms / 1000
        self.calls = 0
        self._closed = False
        self._cond = threading.Condition()
        self._lanes: tuple[deque[Item], deque[Item]] = (deque(), deque())  # (chat, bulk)
        self._last_chat = False  # whether the previous batch came from the chat lane
        self._thread = threading.Thread(target=self._run, name="embedder", daemon=True)
        self._thread.start()

    def submit(self, text: str, bulk: bool = False) -> Future[NDArray[np.float32]]:
        fut: Future[NDArray[np.float32]] = Future()
        with self._cond:
            if self._closed:
                raise RuntimeError("batcher is closed")
            self._lanes[1 if bulk else 0].append((text, fut))
            self._cond.notify()
        return fut

    def __call__(self, texts: list[str], bulk: bool = False) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("no texts")
        return np.vstack([f.result() for f in [self.submit(t, bulk) for t in texts]])

    def close(self) -> None:
        """Finish queued requests, then stop the worker."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify()
        self._thread.join(timeout=30)
        if self._thread.is_alive():  # still draining a long backlog: do not leave callers waiting forever
            with self._cond:
                for lane in self._lanes:
                    while lane:
                        lane.popleft()[1].set_exception(RuntimeError("batcher closed"))

    def _take(self) -> list[Item] | None:
        """The next batch, [] for a keepalive tick, None to stop. Called with the lock held."""
        chat, bulk = self._lanes
        while not chat and not bulk:
            if self._closed:
                return None
            if not self._cond.wait(timeout=self.keepalive) and not chat and not bulk and not self._closed:
                return []  # idle for keepalive seconds
        if not chat or (bulk and self._last_chat):
            self._last_chat = False
            return [bulk.popleft() for _ in range(min(self.bulk_batch, len(bulk)))]
        self._last_chat = True
        batch = [chat.popleft() for _ in range(min(self.max_batch, len(chat)))]
        deadline = time.perf_counter() + self.max_wait
        while len(batch) < self.max_batch and not self._closed:
            if chat:
                batch.append(chat.popleft())
                continue
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            self._cond.wait(timeout=remaining)
        return batch

    def _run(self) -> None:
        while True:
            with self._cond:
                batch = self._take()
            if batch is None:
                return
            if not batch:  # only with keepalive: idle long enough, warm the model
                try:
                    self.fn([self.keepalive_text])
                    self.keepalive_calls += 1
                except Exception:  # a failed warm-up must not kill the worker; make it visible
                    self.keepalive_failures += 1
                    log.debug("keepalive call failed", exc_info=True)
                continue
            live = [(t, f) for t, f in batch if f.set_running_or_notify_cancel()]  # skip cancelled callers
            if not live:
                continue
            self.calls += 1
            try:
                out = np.asarray(self.fn([t for t, _ in live]))
                if len(out) != len(live):
                    raise RuntimeError(f"fn returned {len(out)} rows for {len(live)} texts")
            except Exception as exc:  # deliver the failure to every waiting caller
                for _, fut in live:
                    fut.set_exception(exc)
                continue
            for (_, fut), row in zip(live, out, strict=True):
                fut.set_result(row)


class BatchedEmbedder:
    """An Embedder whose `encode` runs on a MicroBatcher's single worker thread."""

    def __init__(self, inner: Embedder, max_batch: int = 16, max_wait_ms: float = 0.0,
                 keepalive_ms: float | None = None, bulk_batch: int = 4) -> None:
        # bulk_batch 4: chat p95 46 ms while training under 4 nonstop sessions; 8 gave 55-58 ms
        # (docs/benchmarks/train_latency.md)
        self.model_id = inner.model_id
        self.prompt = inner.prompt
        self._batcher = MicroBatcher(lambda texts: inner.encode(texts), max_batch=max_batch,
                                     max_wait_ms=max_wait_ms, keepalive_ms=keepalive_ms, keepalive_text="안녕하세요",
                                     bulk_batch=bulk_batch)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._batcher(list(texts))

    def bulk(self) -> Embedder:
        """The same model in the low-priority lane: for training, so chat stays fast meanwhile."""
        return _BulkLane(self)

    def close(self) -> None:
        self._batcher.close()


class _BulkLane:
    def __init__(self, owner: BatchedEmbedder) -> None:
        self.model_id, self.prompt, self._owner = owner.model_id, owner.prompt, owner

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._owner._batcher(list(texts), bulk=True)
