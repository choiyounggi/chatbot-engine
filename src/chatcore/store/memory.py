"""In-process tracker store (tests, single-process development)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from chatcore.dialogue.events import Event
from chatcore.store.base import SessionBusy, Tracker, check_session_id


class InMemoryStore:
    def __init__(self) -> None:
        self._events: dict[str, list[Event]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._log: list[tuple[str, Event]] = []  # every committed event, in commit order

    @contextmanager
    def session(self, session_id: str) -> Iterator[Tracker]:
        check_session_id(session_id)
        with self._guard:
            lock = self._locks.setdefault(session_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise SessionBusy(session_id)
        try:
            tracker = Tracker(session_id, list(self._events.get(session_id, [])))
            yield tracker
            with self._guard:
                self._events.setdefault(session_id, []).extend(tracker.pending)
                self._log.extend((session_id, e) for e in tracker.pending)
        finally:
            lock.release()

    def events(self, session_id: str) -> list[Event]:
        check_session_id(session_id)
        with self._guard:
            return list(self._events.get(session_id, []))

    def recent(self, prefix: str, event_type: str, limit: int) -> list[tuple[str, Event]]:
        if limit < 1:
            return []
        out: list[tuple[str, Event]] = []
        with self._guard:
            for sid, e in reversed(self._log):
                if sid.startswith(prefix) and e.type == event_type:
                    out.append((sid, e))
                    if len(out) >= limit:
                        break
        return out
