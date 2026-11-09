"""Serving helpers: one worker thread for the embedding model, with batching.

PyTorch MPS can crash when two threads run tensor ops at once
(pytorch/pytorch#197805), so every model call goes through one worker thread.
Requests waiting while the model runs are batched into its next call
(max_wait_ms=0: no artificial wait; measured in docs/benchmarks/concurrency.md).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future

import numpy as np
from numpy.typing import NDArray

from chatcore.nlu.model import Embedder

log = logging.getLogger(__name__)


class MicroBatcher:
    """Runs `fn(list[str]) -> array` on a single worker thread, batching
    requests that arrive within max_wait_ms (at most max_batch per call)."""

    def __init__(self, fn: Callable[[list[str]], NDArray[np.float32]], max_batch: int = 16,
                 max_wait_ms: float = 2.0, keepalive_ms: float | None = None,
                 keepalive_text: str = "keepalive") -> None:
        """keepalive_ms: when idle that long, call fn([keepalive_text]) on the worker thread
        to keep the GPU clocked up (Apple Silicon slows the first request after a pause)."""
        if max_batch < 1 or max_wait_ms < 0 or (keepalive_ms is not None and keepalive_ms <= 0):
            raise ValueError("max_batch must be >= 1, max_wait_ms >= 0 and keepalive_ms > 0")
        self.keepalive = keepalive_ms / 1000 if keepalive_ms is not None else None
        self.keepalive_text = keepalive_text
        self.keepalive_calls = 0
        self.keepalive_failures = 0
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1000
        self.calls = 0
        self._closed = False
        self._lock = threading.Lock()  # orders submit() against close()
        self._queue: queue.Queue[tuple[str, Future[NDArray[np.float32]]] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="embedder", daemon=True)
        self._thread.start()

    def submit(self, text: str) -> Future[NDArray[np.float32]]:
        fut: Future[NDArray[np.float32]] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("batcher is closed")
            self._queue.put((text, fut))
        return fut

    def __call__(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("no texts")
        return np.vstack([f.result() for f in [self.submit(t) for t in texts]])

    def close(self) -> None:
        """Finish queued requests, then stop the worker."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)  # nothing can be enqueued after the sentinel
        self._thread.join(timeout=30)

    def _run(self) -> None:
        stopping = False
        while not stopping:
            try:
                item = self._queue.get(timeout=self.keepalive)
            except queue.Empty:  # only with keepalive: idle long enough, warm the model
                try:
                    self.fn([self.keepalive_text])
                    self.keepalive_calls += 1
                except Exception:  # a failed warm-up must not kill the worker; make it visible
                    self.keepalive_failures += 1
                    log.debug("keepalive call failed", exc_info=True)
                continue
            if item is None:
                return
            batch = [item]
            deadline = time.perf_counter() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - time.perf_counter()
                try:
                    nxt = self._queue.get(timeout=remaining) if remaining > 0 else self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    stopping = True  # serve this batch, then exit
                    break
                batch.append(nxt)
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
                 keepalive_ms: float | None = None) -> None:
        self.model_id = inner.model_id
        self.prompt = inner.prompt
        self._batcher = MicroBatcher(lambda texts: inner.encode(texts), max_batch=max_batch,
                                     max_wait_ms=max_wait_ms, keepalive_ms=keepalive_ms, keepalive_text="안녕하세요")

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return self._batcher(list(texts))

    def close(self) -> None:
        self._batcher.close()
