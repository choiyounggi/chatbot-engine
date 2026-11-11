"""In-process tracker store (tests, single-process development)."""

from __future__ import annotations

import datetime as dt
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from chatcore.dialogue.events import CommandsReceived, Event, UserUttered
from chatcore.store.base import SessionBusy, Tracker, Utterance, check_session_id


class InMemoryStore:
    def __init__(self, clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC)) -> None:
        self._events: dict[str, list[Event]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._log: list[tuple[str, Event, dt.datetime]] = []  # every committed event, in commit order
        self._clock = clock

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
                now = self._clock()
                self._log.extend((session_id, e, now) for e in tracker.pending)
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
            for sid, e, _ in reversed(self._log):
                if sid.startswith(prefix) and e.type == event_type:
                    out.append((sid, e))
                    if len(out) >= limit:
                        break
        return out

    def utterances(self, prefix: str, since: dt.datetime | None, until: dt.datetime | None,
                   limit: int) -> list[Utterance]:
        if limit < 1:
            return []
        out: list[Utterance] = []
        with self._guard:
            for sid, e, at in reversed(self._log):
                if (isinstance(e, CommandsReceived) and sid.startswith(prefix)
                        and (since is None or at >= since) and (until is None or at < until)):
                    out.append(Utterance(sid, at, dict(e.nlu)))
                    if len(out) >= limit:
                        break
        return out

    def activity(self, prefix: str, types: frozenset[str], since: dt.datetime | None,
                 until: dt.datetime | None) -> list[tuple[str, Event, dt.datetime]]:
        out: list[tuple[str, Event, dt.datetime]] = []
        with self._guard:
            for sid, e, at in self._log:  # oldest first already (commit order)
                if (sid.startswith(prefix) and e.type in types
                        and (since is None or at >= since) and (until is None or at < until)):
                    out.append((sid, e, at))
        return out

    def first_seen(self, prefix: str) -> dict[str, dt.datetime]:
        out: dict[str, dt.datetime] = {}
        with self._guard:
            for sid, e, at in self._log:  # oldest first already (commit order)
                if sid.startswith(prefix) and isinstance(e, UserUttered) and sid not in out:
                    out[sid] = at
        return out
