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
            self._events.setdefault(session_id, []).extend(tracker.pending)
        finally:
            lock.release()

    def events(self, session_id: str) -> list[Event]:
        check_session_id(session_id)
        return list(self._events.get(session_id, []))
