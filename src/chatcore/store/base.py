"""Tracker stores: per-session event logs with one turn at a time per session."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from chatcore.dialogue.events import Event


class SessionBusy(RuntimeError):
    """Another turn of the same session is being processed (API: HTTP 409)."""


@dataclass
class Tracker:
    """A session's stored events plus the events this turn will add.
    The store persists `pending` atomically when the session block exits
    normally, and discards it when the block raises."""

    session_id: str
    events: list[Event]
    pending: list[Event] = field(default_factory=list)

    def append(self, events: Iterable[Event]) -> None:
        self.pending.extend(events)


@dataclass(frozen=True)
class Utterance:
    """How one user message was understood (a commands_received event) and when it was stored."""

    session_id: str
    at: dt.datetime  # timezone-aware, UTC
    nlu: dict[str, Any]


class TrackerStore(Protocol):
    def session(self, session_id: str) -> AbstractContextManager[Tracker]:
        """Lock the session (SessionBusy if already locked) and yield its tracker."""
        ...

    def events(self, session_id: str) -> list[Event]:
        """Committed events of a session (read-only, no lock)."""
        ...

    def delete_sessions(self, prefix: str) -> int:
        """Delete every session whose id starts with `prefix` (a literal prefix, not a pattern).
        Returns how many sessions were removed."""
        ...

    def recent(self, prefix: str, event_type: str, limit: int) -> list[tuple[str, Event]]:
        """Newest committed events of one type across sessions whose id starts with `prefix`."""
        ...

    def utterances(self, prefix: str, since: dt.datetime | None, until: dt.datetime | None,
                   limit: int) -> list[Utterance]:
        """Understood user messages of sessions whose id starts with `prefix`, stored in
        [since, until) (None: unbounded), newest first, at most `limit`."""
        ...

    def activity(self, prefix: str, types: frozenset[str], since: dt.datetime | None,
                 until: dt.datetime | None) -> list[tuple[str, Event, dt.datetime]]:
        """(session id, event, stored at) of committed events of these types whose session id starts with
        `prefix`, stored in [since, until), oldest first (ties: commit order). `at` is timezone-aware UTC."""
        ...

    def first_seen(self, prefix: str) -> dict[str, dt.datetime]:
        """Session id -> when its first user_uttered event was stored (sessions starting with prefix)."""
        ...


def check_session_id(session_id: str) -> str:
    if (not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128
            or any(ord(c) < 32 or ord(c) == 127 for c in session_id)):
        raise ValueError("session_id must be 1-128 printable characters")
    return session_id
