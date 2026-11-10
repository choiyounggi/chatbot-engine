"""Chat latency while the builder trains a version (builder plan E-A, criterion 1):
4 concurrent chat sessions over HTTP, real NLU model on MPS, p95 <= 50 ms while a
deployable training job runs.

Three conditions, each with a fresh server and model:
  - no training (baseline)
  - training through the chat lane (before E-A: one first-come queue)
  - training through the bulk lane (E-A: lanes take turns, bulk batches of 4)

Weather actions use a fake network. Artifacts go to a scratch copy that is removed.
Writes docs/benchmarks/train_latency.md; exits 1 if the bulk-lane condition misses.

Usage: uv run python bench/train_latency.py   (after `make train`)
       uv run python bench/train_latency.py --experiment 8 4 2   (compare bulk batch sizes)
"""

from __future__ import annotations

import copy
import platform
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chatcore.api import create_app  # noqa: E402
from chatcore.bot.loader import load_bot  # noqa: E402
from chatcore.bot.repository import InMemoryRepository  # noqa: E402
from chatcore.conversations import synthetic_actions  # noqa: E402
from chatcore.host import BotHost  # noqa: E402
from chatcore.nlu.model import SentenceEmbedder  # noqa: E402
from chatcore.serving import BatchedEmbedder  # noqa: E402
from chatcore.store.memory import InMemoryStore  # noqa: E402

REPORT = ROOT / "docs/benchmarks/train_latency.md"
SCRATCH = ROOT / ".claude/tmp/bench-train-latency"
SENTENCES = ["서울 날씨 어때", "지금 몇 시야", "안녕하세요 반가워요", "부산 기온 알려줘", "고마워요",
             "제주도 날씨 좀 알려줄래", "너는 뭘 할 수 있어?", "대구 지금 비 와?", "오늘 기온 몇 도야", "잘 있어"]
CONCURRENCY, BASELINE_PER_CLIENT = 4, 60
BUDGET_MS = 50.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def measure(condition: str, bulk_batch: int = 4) -> tuple[list[float], float | None]:
    """(chat latencies in ms, training seconds or None)."""
    version = load_bot(ROOT / "bots/sample").version
    artifacts = SCRATCH / f"{condition}-{bulk_batch}-{time.monotonic_ns()}"
    shutil.copytree(ROOT / "artifacts/sample" / version, artifacts / "sample" / version)
    embedder = BatchedEmbedder(SentenceEmbedder(), bulk_batch=bulk_batch)
    bulk = embedder.bulk() if condition == "bulk" else None
    host = BotHost(InMemoryRepository(), InMemoryStore(), embedder, artifacts, actions=synthetic_actions,
                   bulk_embedder=bulk)
    host.seed(ROOT / "bots/sample")  # reuses the copied model
    runtime = host.runtime("sample")
    assert runtime is not None
    runtime.model.parse("워밍업")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(host), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"

    def post(client: httpx.Client, session: str, text: str) -> float:
        t0 = time.perf_counter()
        r = client.post(f"{base}/v1/bots/sample/sessions/{session}/messages", json={"text": text})
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text}")
        return ms

    with httpx.Client(timeout=30) as client:
        for i in range(10):
            post(client, f"warm{i}", SENTENCES[i % len(SENTENCES)])
        training = condition != "none"
        if training:  # a new draft (different label -> new version) so training really runs
            draft = copy.deepcopy(host.repository.get_draft("sample"))
            assert draft is not None
            draft["label"] = f"bench {condition}"
            host.save_draft("sample", draft)
            job = client.post(f"{base}/v1/builder/bots/sample/versions").json()

    done = threading.Event()
    latencies: list[float] = []
    lock = threading.Lock()

    def worker(w: int) -> None:
        with httpx.Client(timeout=30) as client:
            i = 0
            while (not done.is_set()) if training else i < BASELINE_PER_CLIENT:
                ms = post(client, f"c{w}-{i}", SENTENCES[(w + i) % len(SENTENCES)])
                with lock:
                    latencies.append(ms)
                i += 1

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(CONCURRENCY)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    seconds = None
    if training:
        with httpx.Client(timeout=30) as client:
            while True:
                state = client.get(f"{base}/v1/builder/jobs/{job['id']}").json()
                if state["status"] in ("done", "failed"):
                    break
                time.sleep(0.1)
        seconds = time.perf_counter() - t0
        if state["status"] != "done":
            raise RuntimeError(f"training failed: {state}")
        done.set()
    for t in threads:
        t.join()
    server.should_exit = True
    embedder.close()
    return latencies, seconds


def experiment(sizes: list[str]) -> int:
    """`--experiment 8 4 2`: the bulk-lane condition per bulk batch size, no report."""
    try:
        for size in map(int, sizes):
            v, s = measure("bulk", size)
            print(f"bulk_batch={size}: n={len(v)} p50 {np.percentile(v, 50):.1f} p95 {np.percentile(v, 95):.1f} "
                  f"max {max(v):.1f} ms, training {s:.1f}s", flush=True)
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--experiment":
        return experiment(sys.argv[2:])
    names = {"none": "학습 없음 (기준)", "chat": "학습 중, 같은 줄 (E-A 전: 먼저 온 순서)",
             "bulk": "학습 중, 학습 전용 줄 (E-A: 번갈아 처리, 학습 묶음 4)"}
    rows = []
    try:
        for condition in ("none", "chat", "bulk"):
            latencies, seconds = measure(condition)
            rows.append((condition, latencies, seconds))
            print(f"{names[condition]}: n={len(latencies)} p50 {np.percentile(latencies, 50):.1f} "
                  f"p95 {np.percentile(latencies, 95):.1f} max {max(latencies):.1f} ms"
                  + (f", training {seconds:.1f}s" if seconds else ""))
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    bulk_p95 = np.percentile(rows[2][1], 95)
    ok = bulk_p95 <= BUDGET_MS
    lines = [
        "# 학습 중 대화 지연 (빌더 계획 E-A)", "",
        "`uv run python bench/train_latency.py`가 생성한 파일이다. 손으로 고치지 말 것.", "",
        f"- 환경: {platform.machine()} macOS {platform.mac_ver()[0]}, Python {platform.python_version()}. "
        "실제 서버(uvicorn)에 HTTP로 동시 4세션이 쉬지 않고 대화한다. NLU 모델은 MPS 단일 워커.",
        "- 학습: 빌더 API `POST /versions`(배포용 학습: 교차검증 + 적합)가 끝날 때까지 잰다. 날씨 액션은 가짜 네트워크.",
        f"- 기준: 학습 중 동시 4세션 p95 ≤ {BUDGET_MS:.0f}ms (엔진 지연 기준과 같음).",
        f"- 판정: {'통과' if ok else '**기준 미달**'} (학습 전용 줄 p95 {bulk_p95:.1f}ms).", "",
        "| 경우 | 요청 수 | p50 ms | p95 ms | 최대 ms | 학습 시간 |", "|---|---|---|---|---|---|",
        *[f"| {names[c]} | {len(v)} | {np.percentile(v, 50):.1f} | {np.percentile(v, 95):.1f} | {max(v):.1f} | "
          f"{f'{s:.1f}초' if s else '-'} |" for c, v, s in rows],
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {REPORT.relative_to(ROOT)}; {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
