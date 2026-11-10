import threading

import pytest

from chatcore.jobs import JobFailed, JobRunner


def wait_done(runner, job_id):
    seen = -1
    for _ in range(100):
        job, seen = runner.wait(job_id, seen, timeout=0.1)
        if job["status"] in ("done", "failed"):
            return job
    raise AssertionError("job did not finish")


def wait_running(runner, job_id):
    seen = -1
    for _ in range(100):
        job, seen = runner.wait(job_id, seen, timeout=0.1)
        if job["status"] == "running":
            return
    raise AssertionError("job did not start")


def test_jobs_run_in_order_and_report_results():
    runner = JobRunner()
    order = []
    a = runner.submit("version", "a", lambda: order.append("a") or {"n": 1})
    b = runner.submit("version", "b", lambda: order.append("b") or {"n": 2})
    assert wait_done(runner, a.id)["result"] == {"n": 1} and wait_done(runner, b.id)["result"] == {"n": 2}
    assert order == ["a", "b"]
    runner.close()


def test_a_queued_job_of_the_same_kind_and_bot_is_reused():
    runner = JobRunner()
    gate = threading.Event()
    running = runner.submit("version", "a", lambda: gate.wait(5) and {})
    wait_running(runner, running.id)
    calls = []
    first = runner.submit("test_build", "a", lambda: calls.append(1) or {})
    again = runner.submit("test_build", "a", lambda: calls.append(2) or {})  # still queued: same job
    other = runner.submit("test_build", "b", lambda: {})
    assert again.id == first.id and other.id != first.id
    same_as_running = runner.submit("version", "a", lambda: {})  # a running job is never reused
    assert same_as_running.id != running.id
    gate.set()
    for job in (running, first, other, same_as_running):
        wait_done(runner, job.id)
    assert calls == [1]
    runner.close()


def test_failures_keep_their_detail_but_unexpected_errors_do_not_leak():
    runner = JobRunner()

    def expected():
        raise JobFailed("below_bar", {"detail": "macro-F1 0.5"})

    def bug():
        raise RuntimeError("secret path /etc/x")

    one, two = runner.submit("version", "a", expected), runner.submit("version", "b", bug)
    assert wait_done(runner, one.id)["error"] == {"kind": "below_bar", "detail": "macro-F1 0.5"}
    failed = wait_done(runner, two.id)
    assert failed["status"] == "failed" and "secret" not in str(failed) and failed["finished_at"]
    runner.close()


def test_wait_returns_on_change_or_timeout_and_unknown_jobs_are_none():
    runner = JobRunner()
    gate = threading.Event()
    job = runner.submit("version", "a", lambda: gate.wait(5) and {})
    seen = -1
    for _ in range(100):  # wait until the worker picked it up
        current, seen = runner.wait(job.id, seen, timeout=0.1)
        if current["status"] == "running":
            break
    still, same = runner.wait(job.id, seen, timeout=0.05)  # nothing changes while the gate is shut
    assert still["status"] == "running" and same == seen
    gate.set()
    assert wait_done(runner, job.id)["status"] == "done"
    assert runner.wait("0" * 32, -1, timeout=0.01) is None and runner.get("0" * 32) is None
    runner.close()


def test_finished_jobs_are_forgotten_first_and_unfinished_ones_never():
    runner = JobRunner(keep=2)
    a = runner.submit("version", "a", lambda: {})
    b = runner.submit("version", "b", lambda: {})
    wait_done(runner, a.id)
    wait_done(runner, b.id)
    gate = threading.Event()
    running = runner.submit("version", "c", lambda: gate.wait(5) and {})
    queued = runner.submit("version", "d", lambda: {})  # four jobs, keep=2: both finished ones go
    assert runner.get(a.id) is None and runner.get(b.id) is None
    assert runner.get(running.id) is not None and runner.get(queued.id) is not None
    gate.set()
    wait_done(runner, queued.id)
    runner.close()


def test_close_finishes_queued_jobs_then_refuses_new_ones():
    runner = JobRunner()
    done = []
    for bot in ("a", "b", "c"):
        runner.submit("version", bot, lambda bot=bot: done.append(bot) or {})
    runner.close()
    assert done == ["a", "b", "c"]
    with pytest.raises(RuntimeError, match="closed"):
        runner.submit("version", "a", lambda: {})
    with pytest.raises(ValueError, match="keep"):
        JobRunner(keep=0)


def test_urgent_jobs_go_before_queued_normal_ones_but_never_interrupt():
    runner = JobRunner()
    gate, order = threading.Event(), []
    running = runner.submit("version", "a", lambda: gate.wait(5) and order.append("running") or {})
    wait_running(runner, running.id)
    normal = runner.submit("version", "b", lambda: order.append("normal") or {})
    urgent = runner.submit("test_build", "a", lambda: order.append("urgent") or {}, urgent=True)
    gate.set()
    for job in (running, normal, urgent):
        wait_done(runner, job.id)
    assert order == ["running", "urgent", "normal"]
    runner.close()


def test_the_worker_survives_a_job_that_raises_a_base_exception():
    runner = JobRunner()

    def dies():
        raise SystemExit(3)

    failed = wait_done(runner, runner.submit("version", "a", dies).id)
    assert failed["error"]["kind"] == "internal"
    assert wait_done(runner, runner.submit("version", "b", lambda: {"ok": 1}).id)["result"] == {"ok": 1}
    runner.close()
