import threading
import time

import numpy as np
import pytest

from bench.concurrency import MicroBatcher, load_test
from bench.eval_independent import resolve_pair
from bench.phase0 import BenchError

CLASSES = np.array(["bye", "greeting", "out_of_scope", "weather"])


def probs(bye=0.0, greeting=0.0, oos=0.0, weather=0.0):
    return np.array([bye, greeting, oos, weather])


def fake_encode(texts):
    time.sleep(0.005)
    return np.array([[len(t), ord(t[0])] for t in texts], dtype=float)


# ---------------------------------------------------------------- resolve_pair

@pytest.mark.parametrize(
    "label, p, turn, expected",
    [
        ("bye", probs(bye=0.55, greeting=0.40), 0, "greeting"),  # first turn: ambiguous → greeting
        ("bye", probs(bye=0.95, greeting=0.02), 0, "bye"),  # clear goodbye even on turn 0
        ("greeting", probs(bye=0.40, greeting=0.55), 3, "bye"),  # later: ambiguous → bye
        ("greeting", probs(bye=0.05, greeting=0.90), 3, "greeting"),  # clear re-greeting mid-chat
        ("weather", probs(weather=0.9), 0, "weather"),  # other labels untouched
        ("out_of_scope", probs(bye=0.5, greeting=0.5), 2, "out_of_scope"),
    ],
)
def test_resolve_pair_uses_position_with_margin(label, p, turn, expected):
    assert resolve_pair(label, p, CLASSES, turn) == expected


def test_resolve_pair_boundary_margin_is_inclusive():
    assert resolve_pair("bye", probs(bye=0.75, greeting=0.25), CLASSES, 0) == "bye"


def test_resolve_pair_errors():
    with pytest.raises(BenchError):
        resolve_pair("bye", probs(bye=1.0), CLASSES, -1)
    with pytest.raises(BenchError):
        resolve_pair("bye", np.array([1.0, 0.0]), np.array(["bye", "weather"]), 0)


# ---------------------------------------------------------------- MicroBatcher

def test_batcher_returns_rows_in_request_order():
    b = MicroBatcher(fake_encode)
    try:
        out = b(["a", "bb", "ccc"])
        np.testing.assert_array_equal(out[:, 0], [1, 2, 3])
    finally:
        b.close()


def test_batcher_groups_concurrent_requests_on_one_thread():
    seen_threads = set()

    def fn(texts):
        seen_threads.add(threading.get_ident())
        return fake_encode(texts)

    b = MicroBatcher(fn, max_batch=16, max_wait_ms=20)
    try:
        lat = load_test(b, ["가", "나", "다"], concurrency=8, total=48)
        assert len(lat) == 48
        assert b.calls < 48  # batching happened
        assert len(seen_threads) == 1  # every model call on the worker thread
    finally:
        b.close()


def test_serialized_batcher_never_batches():
    b = MicroBatcher(fake_encode, max_batch=1, max_wait_ms=0)
    try:
        load_test(b, ["가"], concurrency=4, total=12)
        assert b.calls == 12
    finally:
        b.close()


def test_batcher_propagates_errors_to_every_caller():
    def boom(texts):
        raise RuntimeError("model failed")

    b = MicroBatcher(boom)
    try:
        futures = [b.submit("가"), b.submit("나")]
        for f in futures:
            with pytest.raises(RuntimeError, match="model failed"):
                f.result(timeout=5)
    finally:
        b.close()


def test_batcher_rejects_wrong_row_count():
    b = MicroBatcher(lambda texts: np.zeros((len(texts) + 1, 2)))
    try:
        with pytest.raises(RuntimeError, match="rows"):
            b.submit("가").result(timeout=5)
    finally:
        b.close()


def test_batcher_validation_and_closed_state():
    with pytest.raises(ValueError):
        MicroBatcher(fake_encode, max_batch=0)
    b = MicroBatcher(fake_encode)
    with pytest.raises(ValueError):
        b([])
    b.close()
    with pytest.raises(RuntimeError):
        b.submit("가")


def test_load_test_validates_arguments():
    b = MicroBatcher(fake_encode)
    try:
        with pytest.raises(ValueError):
            load_test(b, [], 1, 1)
    finally:
        b.close()


def test_close_finishes_queued_requests_and_rejects_new_ones():
    b = MicroBatcher(fake_encode, max_batch=2, max_wait_ms=0)
    futures = [b.submit(t) for t in ["가", "나", "다", "라"]]
    b.close()
    assert all(f.result(timeout=5) is not None for f in futures)
    with pytest.raises(RuntimeError, match="closed"):
        b.submit("마")
    b.close()  # idempotent


def test_cancelled_request_does_not_break_its_batch():
    gate = threading.Event()

    def slow(texts):
        gate.wait(5)
        return fake_encode(texts)

    b = MicroBatcher(slow, max_batch=8, max_wait_ms=0)
    try:
        first = b.submit("가")  # occupies the worker
        cancelled, kept = b.submit("나"), b.submit("다")
        assert cancelled.cancel()
        gate.set()
        assert first.result(timeout=5)[0] == 1 and kept.result(timeout=5)[0] == 1
    finally:
        b.close()


def test_open_loop_measures_every_request():
    from bench.concurrency import open_loop

    b = MicroBatcher(fake_encode, max_batch=8, max_wait_ms=0)
    try:
        lat = open_loop(b, ["가", "나"], rate=200, total=30)
        assert len(lat) == 30 and min(lat) > 0
        with pytest.raises(ValueError):
            open_loop(b, ["가"], rate=0, total=1)
    finally:
        b.close()


def test_idle_latency_waits_between_requests():
    from bench.concurrency import idle_latency

    calls = []
    t0 = time.perf_counter()
    lat = idle_latency(lambda xs: calls.append(xs) or np.zeros((1, 2)), ["가"], gap=0.01, n=3)
    assert len(lat) == 3 and len(calls) == 3
    assert time.perf_counter() - t0 >= 0.03
    with pytest.raises(ValueError):
        idle_latency(fake_encode, ["가"], gap=-1)
