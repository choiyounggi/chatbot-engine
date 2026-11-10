"""Background jobs for the builder: training runs here, not inside the HTTP request.

One worker thread runs jobs in order (training is heavy and shares the
embedding model). Jobs live in memory: a server restart forgets them, and the
builder simply starts the job again. A queued job of the same kind for the same
bot is reused instead of adding another, so ten quick saves train once.
Urgent jobs (test builds: someone waits at the test chat) run before queued
normal ones (deployable training), but never interrupt a running job.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class JobFailed(Exception):
    """Raised by a job function for an expected failure; `detail` reaches the builder."""

    def __init__(self, kind: str, detail: dict[str, Any]) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail = kind, detail


@dataclass
class Job:
    id: str
    kind: str
    bot: str
    status: str = "queued"  # queued -> running -> done | failed
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: _now())
    finished_at: str | None = None
    revision: int = 0  # bumps on every change, so a watcher can wait for "newer than what I saw"

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "bot": self.bot, "status": self.status, "result": self.result,
                "error": self.error, "created_at": self.created_at, "finished_at": self.finished_at}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class JobRunner:
    def __init__(self, keep: int = 200) -> None:
        if keep < 1:
            raise ValueError("keep must be >= 1")
        self.keep = keep
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queues: tuple[deque[tuple[Job, Callable[[], dict[str, Any]]]], ...] = (deque(), deque())  # urgent, normal
        self._cond = threading.Condition()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="jobs", daemon=True)
        self._thread.start()

    def submit(self, kind: str, bot: str, fn: Callable[[], dict[str, Any]], urgent: bool = False) -> Job:
        """Queue fn (it reads the bot's current state when it starts). Returns the job."""
        with self._cond:
            if self._closed:
                raise RuntimeError("job runner is closed")
            for queue in self._queues:
                for queued, _ in queue:
                    if queued.kind == kind and queued.bot == bot:
                        return queued
            job = Job(uuid.uuid4().hex, kind, bot)
            self._jobs[job.id] = job
            self._queues[0 if urgent else 1].append((job, fn))
            self._forget_old()
            self._cond.notify_all()
            return job

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._cond:
            job = self._jobs.get(job_id)
            return job.to_dict() if job is not None else None

    def wait(self, job_id: str, seen: int, timeout: float) -> tuple[dict[str, Any], int] | None:
        """Block until the job changes after revision `seen` (or timeout). Returns (job, revision),
        or None for an unknown job."""
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            self._cond.wait_for(lambda: job.revision > seen or job.finished, timeout=timeout)
            return job.to_dict(), job.revision

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join(timeout=60)

    def _forget_old(self) -> None:
        while len(self._jobs) > self.keep:
            oldest = next((j for j in self._jobs.values() if j.finished), None)
            if oldest is None:
                return
            del self._jobs[oldest.id]

    def _set(self, job: Job, **changes: Any) -> None:
        with self._cond:
            for key, value in changes.items():
                setattr(job, key, value)
            job.revision += 1
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: any(self._queues) or self._closed)
                queue = next((q for q in self._queues if q), None)
                if queue is None:
                    return
                job, fn = queue.popleft()
            self._set(job, status="running")
            try:
                result = fn()
            except JobFailed as exc:
                self._set(job, status="failed", error={"kind": exc.kind, **exc.detail}, finished_at=_now())
            except BaseException:  # a bug; the worker must survive it so later jobs still run
                log.exception("job %s (%s for %s) failed", job.id, job.kind, job.bot)
                self._set(job, status="failed", error={"kind": "internal", "detail": "internal error"},
                          finished_at=_now())
            else:
                self._set(job, status="done", result=result, finished_at=_now())
