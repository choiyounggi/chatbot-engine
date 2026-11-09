"""Concurrency benchmark for the embedding model on Apple Silicon (MPS).

PyTorch MPS can crash when two Python threads run tensor ops at once
(pytorch/pytorch#197805), so every model call goes through one worker thread.
`MicroBatcher` adds dynamic micro-batching on that thread (requests that
arrive within `max_wait_ms` share one forward pass, like TEI / Triton /
Ray Serve batching). This script compares serialized vs batched latency at
several concurrency levels, and checks in a subprocess whether unsynchronized
multi-threaded calls still crash on the installed torch.

Usage: uv run python bench/concurrency.py
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs/benchmarks/concurrency.md"
LEVELS = (1, 4, 8, 16)
RATES = (20, 60, 120)  # open-loop Poisson arrival rates, req/s
IDLE_GAPS = (0.0, 0.2, 1.0)  # seconds of silence before each single request
IDLE_REQUESTS = 30
REQUESTS_PER_LEVEL = 240


class MicroBatcher:
    """Runs `fn(list[str]) -> array` on a single worker thread, batching
    requests that arrive within max_wait_ms (at most max_batch per call)."""

    def __init__(self, fn: Callable[[list[str]], np.ndarray], max_batch: int = 16, max_wait_ms: float = 2.0) -> None:
        if max_batch < 1 or max_wait_ms < 0:
            raise ValueError("max_batch must be >= 1 and max_wait_ms >= 0")
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1000
        self.calls = 0
        self._closed = False
        self._lock = threading.Lock()  # orders submit() against close()
        self._queue: queue.Queue[tuple[str, Future] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="embedder", daemon=True)
        self._thread.start()

    def submit(self, text: str) -> Future:
        fut: Future = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("batcher is closed")
            self._queue.put((text, fut))
        return fut

    def __call__(self, texts: list[str]) -> np.ndarray:
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
            item = self._queue.get()
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


def load_test(batcher: MicroBatcher, texts: list[str], concurrency: int, total: int) -> list[float]:
    """`concurrency` client threads issue `total` single-text requests; returns per-request ms."""
    if concurrency < 1 or total < 1 or not texts:
        raise ValueError("need concurrency >= 1, total >= 1 and texts")
    latencies: list[float] = []
    lock = threading.Lock()
    counter = iter(range(total))

    def client() -> None:
        while True:
            with lock:
                i = next(counter, None)
            if i is None:
                return
            t0 = time.perf_counter()
            batcher.submit(texts[i % len(texts)]).result()
            ms = (time.perf_counter() - t0) * 1000
            with lock:
                latencies.append(ms)

    threads = [threading.Thread(target=client) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return latencies


def open_loop(batcher: MicroBatcher, texts: list[str], rate: float, total: int, seed: int = 0) -> list[float]:
    """Poisson arrivals at `rate` req/s regardless of completions; returns ms
    from each request's scheduled arrival to its completion."""
    if rate <= 0 or total < 1 or not texts:
        raise ValueError("need rate > 0, total >= 1 and texts")
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1 / rate, size=total))
    done: list[float] = []
    lock = threading.Lock()
    all_done = threading.Event()
    start = time.perf_counter()

    def finish(scheduled: float) -> Callable[[Future], None]:
        def cb(_: Future) -> None:
            with lock:
                done.append((time.perf_counter() - scheduled) * 1000)
                if len(done) == total:
                    all_done.set()
        return cb

    for i, at in enumerate(arrivals):
        delay = start + at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        batcher.submit(texts[i % len(texts)]).add_done_callback(finish(start + at))
    if not all_done.wait(timeout=120):
        raise RuntimeError("open-loop requests did not finish")
    return done


def idle_latency(encode: Callable[[list[str]], np.ndarray], texts: list[str], gap: float, n: int = IDLE_REQUESTS) -> list[float]:
    """ms per single request when each one follows `gap` seconds of silence
    (a chat user typing). Apple Silicon lowers GPU/CPU clocks while idle."""
    if gap < 0 or n < 1 or not texts:
        raise ValueError("need gap >= 0, n >= 1 and texts")
    out = []
    for i in range(n):
        if gap:
            time.sleep(gap)
        t0 = time.perf_counter()
        encode([texts[i % len(texts)]])
        out.append((time.perf_counter() - t0) * 1000)
    return out


UNSAFE_PROBE = r"""
import threading, torch
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko", device="mps", model_kwargs={"dtype": torch.float16})
def work():
    for _ in range(40):
        m.encode(["query: 서울 날씨 어때?"])
ts = [threading.Thread(target=work) for _ in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print("NO_CRASH")
"""


def unsafe_probe(runs: int = 3) -> list[int]:
    """Exit codes of unsynchronized 4-thread MPS encode runs (0 = survived, -9 = hung)."""
    codes = []
    for _ in range(runs):
        try:
            r = subprocess.run([sys.executable, "-c", UNSAFE_PROBE], capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            codes.append(-9)  # hung (deadlock)
            continue
        codes.append(r.returncode if "NO_CRASH" in r.stdout or r.returncode != 0 else -1)
    return codes


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, str(ROOT))
    from bench.phase0 import EMBEDDINGS, load_bot

    spec = EMBEDDINGS["arctic"]
    model = SentenceTransformer(spec.model_id, device=spec.device, model_kwargs={"dtype": getattr(torch, spec.dtype)})

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode([spec.prompt + t for t in texts], normalize_embeddings=True, batch_size=64)

    intents, _ = load_bot(ROOT / "bots/sample/nlu.yaml")
    texts = [t.replace("{location}", "서울") for exs in intents.values() for t in exs]
    for size in range(1, 17):  # compile every batch shape before timing
        encode(texts[:size])
    rows, open_rows = [], []
    modes = (
        ("직렬(배치 없음)", {"max_batch": 1, "max_wait_ms": 0.0}),
        ("즉시 배칭(최대 16, 대기 0ms: 실행 중 쌓인 요청만 묶음)", {"max_batch": 16, "max_wait_ms": 0.0}),
        ("마이크로배칭(최대 16, 대기 2ms)", {"max_batch": 16, "max_wait_ms": 2.0}),
    )
    for mode, kwargs in modes:
        batcher = MicroBatcher(encode, **kwargs)
        load_test(batcher, texts, 1, 20)  # warm-up
        for level in LEVELS:
            calls0 = batcher.calls
            t0 = time.perf_counter()
            lat = load_test(batcher, texts, level, REQUESTS_PER_LEVEL)
            wall = time.perf_counter() - t0
            rows.append((mode, level, np.percentile(lat, 50), np.percentile(lat, 95), np.percentile(lat, 99),
                         REQUESTS_PER_LEVEL / wall, REQUESTS_PER_LEVEL / max(batcher.calls - calls0, 1)))
            print(f"{mode} c={level}: p50={rows[-1][2]:.1f} p95={rows[-1][3]:.1f} p99={rows[-1][4]:.1f} ms, {rows[-1][5]:.0f} req/s, avg batch {rows[-1][6]:.1f}", flush=True)
        for rate in RATES:
            lat = open_loop(batcher, texts, rate, REQUESTS_PER_LEVEL)
            open_rows.append((mode, rate, np.percentile(lat, 50), np.percentile(lat, 95), np.percentile(lat, 99)))
            print(f"{mode} open-loop {rate} req/s: p50={open_rows[-1][2]:.1f} p95={open_rows[-1][3]:.1f} p99={open_rows[-1][4]:.1f} ms", flush=True)
        batcher.close()
    idle_rows = []
    for gap in IDLE_GAPS:
        lat = idle_latency(encode, texts, gap)
        idle_rows.append((gap, np.percentile(lat, 50), np.percentile(lat, 95)))
        print(f"idle gap {gap*1000:.0f}ms: p50={idle_rows[-1][1]:.1f} p95={idle_rows[-1][2]:.1f} ms", flush=True)
    codes = unsafe_probe()
    print("unsafe multi-thread probe exit codes:", codes)

    lines = [
        "# 임베딩 동시 요청 측정",
        "",
        "`uv run python bench/concurrency.py`가 생성한 파일이다. 손으로 고치지 말 것.",
        "",
        f"- 모델 `{spec.model_id}` ({spec.device} {spec.dtype}), torch {torch.__version__}. 임베딩만 잰 값이다(분류기 연산 < 1ms 별도).",
        f"- 수준마다 요청 {REQUESTS_PER_LEVEL}개. 모든 모델 호출은 워커 스레드 하나에서 실행한다. 측정 전 배치 크기 1~16을 한 번씩 실행해 MPS 첫 실행 지연을 뺐다.",
        "",
        "## 닫힌 루프 — 클라이언트 스레드 수 = 동시 요청 수",
        "",
        "",
        "| 방식 | 동시 요청 | p50 ms | p95 ms | p99 ms | 처리량 req/s | 평균 배치 크기 |",
        "|---|---|---|---|---|---|---|",
        *[f"| {m} | {c} | {a:.1f} | {b:.1f} | {d:.1f} | {e:.0f} | {f:.1f} |" for m, c, a, b, d, e, f in rows],
        "",
        "## 유휴 후 단일 요청 — 채팅 사용자처럼 요청 사이에 쉬는 경우",
        "",
        f"요청 {IDLE_REQUESTS}개를 하나씩, 매번 정해진 시간만큼 쉰 뒤 보냈다. 맥은 쉬는 동안 GPU·CPU 클럭을 내리므로 연속 요청보다 느리다. 실제 채팅의 단일 요청 지연은 이 표가 가깝다.",
        "",
        "| 쉬는 시간 | p50 ms | p95 ms |",
        "|---|---|---|",
        *[f"| {g*1000:.0f}ms | {a:.1f} | {b:.1f} |" for g, a, b in idle_rows],
        "",
        "## 열린 루프(푸아송 도착) — 응답을 기다리지 않고 정해진 평균 속도로 요청이 들어올 때",
        "",
        "지연 = 요청 도착 예정 시각부터 응답 완료까지(큐 대기 포함).",
        "",
        "| 방식 | 도착률 req/s | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|---|",
        *[f"| {m} | {r} | {a:.1f} | {b:.1f} | {d:.1f} |" for m, r, a, b, d in open_rows],
        "",
        "닫힌 루프(위 표)는 클라이언트가 응답 직후 다시 요청하므로 배치가 꽉 차기 쉽다. 실제 도착에 가까운 값은 이 표다.",
        "",
        "## 스레드 동기화 없이 4개 스레드가 동시에 encode (별도 프로세스, 3회)",
        "",
        f"- 종료 코드: {codes} (0 = 정상 종료, 134/139 = Metal assertion/SIGSEGV로 죽음)",
        "- pytorch/pytorch#197805: 여러 스레드의 동시 MPS 연산이 프로세스를 죽일 수 있다. 고친 커밋(c8bb607)은 2026-09-23 main에 들어갔고, 정식 릴리스 포함 여부는 확인하지 못했다. 결과와 상관없이 엔진은 단일 워커로 직렬화한다.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report → {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
