"""End-to-end HTTP latency of the real server (real NLU model on MPS), Phase 5 criterion:
p95 <= 50 ms for single requests after idle gaps and for 4 concurrent sessions.

Weather actions use a fake network so only the engine + NLU is measured.
Writes docs/benchmarks/api_latency.md; exits 1 if a criterion fails.

Usage: uv run python bench/api_latency.py   (after `make train`)
"""

from __future__ import annotations

import platform
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
from chatcore.conversations import synthetic_actions  # noqa: E402
from chatcore.host import StaticHost  # noqa: E402
from chatcore.nlu.model import SentenceEmbedder  # noqa: E402
from chatcore.runtime import build_runtime  # noqa: E402
from chatcore.serving import BatchedEmbedder  # noqa: E402
from chatcore.store.memory import InMemoryStore  # noqa: E402

REPORT = ROOT / "docs/benchmarks/api_latency.md"
SENTENCES = ["서울 날씨 어때", "지금 몇 시야", "안녕하세요 반가워요", "부산 기온 알려줘", "고마워요",
             "제주도 날씨 좀 알려줄래", "너는 뭘 할 수 있어?", "대구 지금 비 와?", "오늘 기온 몇 도야", "잘 있어"]
IDLE_GAP, IDLE_N = 0.5, 40
CONCURRENCY, PER_CLIENT = 4, 40
BUDGET_MS = 50.0
KEEPALIVE_MS = 20.0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def measure(keepalive_ms: float | None) -> list[tuple[str, list[float]]]:
    embedder = BatchedEmbedder(SentenceEmbedder(), keepalive_ms=keepalive_ms)
    rt = build_runtime(ROOT / "bots/sample", ROOT / "artifacts", embedder, InMemoryStore(), actions=synthetic_actions())
    rt.model.parse("워밍업")
    port = free_port()
    app = create_app(StaticHost({"sample": rt}, rt.processor.store))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}/v1/bots/sample/sessions"

    def post(client: httpx.Client, session: str, text: str) -> float:
        t0 = time.perf_counter()
        r = client.post(f"{base}/{session}/messages", json={"text": text})
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text}")
        return ms

    with httpx.Client(timeout=10) as client:
        for i in range(10):  # warm up HTTP and every code path
            post(client, f"warm{i}", SENTENCES[i % len(SENTENCES)])
        idle = []
        for i in range(IDLE_N):
            time.sleep(IDLE_GAP)
            idle.append(post(client, f"idle{i}", SENTENCES[i % len(SENTENCES)]))

    concurrent: list[float] = []
    lock = threading.Lock()

    def worker(w: int) -> None:
        with httpx.Client(timeout=10) as client:
            for i in range(PER_CLIENT):
                ms = post(client, f"c{w}-{i}", SENTENCES[(w + i) % len(SENTENCES)])
                with lock:
                    concurrent.append(ms)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(CONCURRENCY)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    server.should_exit = True
    embedder.close()
    tag = "GPU 깨워 두기 끔 (기본)" if keepalive_ms is None else f"GPU 깨워 두기 {keepalive_ms:.0f}ms"
    return [(f"유휴 후 단일 요청 (0.5초 간격), {tag}", idle, keepalive_ms is None),
            (f"동시 {CONCURRENCY}개 세션, {tag}", concurrent, keepalive_ms is None)]


def main() -> int:
    rows = measure(None) + measure(KEEPALIVE_MS)
    ok = all(np.percentile(v, 95) <= BUDGET_MS for _, v, default in rows if default)
    ok_keepalive = all(np.percentile(v, 95) <= BUDGET_MS for _, v, default in rows if not default)
    lines = [
        "# HTTP API 지연 (Phase 5)", "",
        "`uv run python bench/api_latency.py`가 생성한 파일이다. 손으로 고치지 말 것.", "",
        f"- 환경: {platform.machine()} macOS {platform.mac_ver()[0]}, Python {platform.python_version()}. "
        f"실제 서버(uvicorn)에 HTTP로 요청했다. NLU 모델은 MPS에서 단일 워커 + 대기 0ms 배칭으로 돌렸다.",
        "- 날씨 액션은 가짜 네트워크를 쓴다(엔진과 NLU만 잰다). 실제 Open-Meteo 호출은 네트워크 시간이 더해진다.",
        f"- 기준: 두 경우 모두 p95 ≤ {BUDGET_MS:.0f}ms (계획 Phase 5, 2026-11-09 합의)",
        f"- GPU 깨워 두기: 쉬는 동안 {KEEPALIVE_MS:.0f}ms마다 모델을 한 번 호출한다(`make serve KEEP_WARM_MS=...`). GPU를 계속 쓴다.",
        f"- 판정: 기본 설정 {'통과' if ok else '**기준 미달**'}, GPU 깨워 두기 {'통과' if ok_keepalive else '**기준 미달**'}. "
        f"p95는 요청 {IDLE_N}개 중 두 번째로 느린 값 근처라 실행마다 수 ms씩 흔들린다.", "",
        "| 경우 | 요청 수 | p50 ms | p95 ms | 최대 ms | 기준 |", "|---|---|---|---|---|---|",
        *[f"| {name} | {len(v)} | {np.percentile(v, 50):.1f} | {np.percentile(v, 95):.1f} | {max(v):.1f} | "
          f"{'✅' if np.percentile(v, 95) <= BUDGET_MS else '❌'} |" for name, v, _ in rows],
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, v, _ in rows:
        print(f"{name}: p50 {np.percentile(v, 50):.1f} p95 {np.percentile(v, 95):.1f} max {max(v):.1f} ms")
    print(f"report -> {REPORT.relative_to(ROOT)}; default {'PASS' if ok else 'FAIL'}, "
          f"keep-warm {'PASS' if ok_keepalive else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
