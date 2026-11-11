"""Postgres tracker store.

One transaction per turn: a transaction-scoped advisory lock on the session
(`pg_try_advisory_xact_lock`, so a second concurrent turn fails fast with
SessionBusy instead of waiting), read the log, insert the new events, commit.
If the turn raises, the transaction rolls back and nothing is stored.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from chatcore.dialogue.events import Event, from_json, to_json
from chatcore.store.base import SessionBusy, Tracker, Utterance, check_session_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_events (
    session_id  text        NOT NULL,
    seq         integer     NOT NULL,
    type        text        NOT NULL,
    payload     jsonb       NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS chat_events_type_time ON chat_events (type, created_at DESC);
"""


class PostgresStore:
    def __init__(self, dsn: str, connect_timeout: int = 5) -> None:
        if not dsn:
            raise ValueError("dsn is required")
        self.dsn = dsn
        self.connect_timeout = connect_timeout

    def _connect(self) -> psycopg.Connection[tuple[object, ...]]:
        return psycopg.connect(self.dsn, connect_timeout=self.connect_timeout)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def session(self, session_id: str) -> Iterator[Tracker]:
        check_session_id(session_id)
        with self._connect() as conn, conn.transaction():
            row = conn.execute("SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0))", (session_id,)).fetchone()
            if row is None or row[0] is not True:
                raise SessionBusy(session_id)
            events = self._load(conn, session_id)
            tracker = Tracker(session_id, events)
            yield tracker
            if tracker.pending:
                start = len(events)
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chat_events (session_id, seq, type, payload) VALUES (%s, %s, %s, %s)",
                        [(session_id, start + i, e.type, Jsonb(to_json(e))) for i, e in enumerate(tracker.pending)],
                    )

    def recent(self, prefix: str, event_type: str, limit: int) -> list[tuple[str, Event]]:
        if limit < 1:
            return []
        pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, payload FROM chat_events WHERE session_id LIKE %s AND type = %s "
                "ORDER BY created_at DESC, seq DESC LIMIT %s",
                (pattern, event_type, limit),
            ).fetchall()
        out: list[tuple[str, Event]] = []
        for sid, payload in rows:
            if not isinstance(payload, dict):
                raise TypeError(f"chat_events.payload must be a JSON object, got {type(payload).__name__}")
            out.append((str(sid), from_json(payload)))
        return out

    def utterances(self, prefix: str, since: dt.datetime | None, until: dt.datetime | None,
                   limit: int) -> list[Utterance]:
        if limit < 1:
            return []
        pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, created_at, payload FROM chat_events WHERE type = 'commands_received' "
                "AND session_id LIKE %s AND (%s::timestamptz IS NULL OR created_at >= %s) "
                "AND (%s::timestamptz IS NULL OR created_at < %s) ORDER BY created_at DESC, seq DESC LIMIT %s",
                (pattern, since, since, until, until, limit),
            ).fetchall()
        out: list[Utterance] = []
        for sid, at, payload in rows:
            if not isinstance(payload, dict) or not isinstance(at, dt.datetime):
                raise TypeError("chat_events rows must hold a JSON object and a timestamp")
            nlu: Any = payload.get("nlu", {})
            out.append(Utterance(str(sid), at.astimezone(dt.UTC), dict(nlu) if isinstance(nlu, dict) else {}))
        return out

    def events(self, session_id: str) -> list[Event]:
        check_session_id(session_id)
        with self._connect() as conn:
            return self._load(conn, session_id)

    def activity(self, prefix: str, types: frozenset[str], since: dt.datetime | None,
                 until: dt.datetime | None) -> list[tuple[str, Event, dt.datetime]]:
        if not types:
            return []
        pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, created_at, payload FROM chat_events WHERE session_id LIKE %s "
                "AND type = ANY(%s) AND (%s::timestamptz IS NULL OR created_at >= %s) "
                "AND (%s::timestamptz IS NULL OR created_at < %s) ORDER BY created_at, session_id, seq",
                (pattern, list(types), since, since, until, until),
            ).fetchall()
        out: list[tuple[str, Event, dt.datetime]] = []
        for sid, at, payload in rows:
            if not isinstance(payload, dict) or not isinstance(at, dt.datetime):
                raise TypeError("chat_events rows must hold a JSON object and a timestamp")
            out.append((str(sid), from_json(payload), at.astimezone(dt.UTC)))
        return out

    def first_seen(self, prefix: str) -> dict[str, dt.datetime]:
        pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, MIN(created_at) FROM chat_events WHERE session_id LIKE %s "
                "AND type = 'user_uttered' GROUP BY session_id",
                (pattern,),
            ).fetchall()
        out: dict[str, dt.datetime] = {}
        for sid, at in rows:
            if not isinstance(at, dt.datetime):
                raise TypeError("chat_events.created_at must be a timestamp")
            out[str(sid)] = at.astimezone(dt.UTC)
        return out

    def delete_sessions(self, prefix: str) -> int:
        pattern = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._connect() as conn:  # one transaction: count, then delete the same rows
            row = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM chat_events WHERE session_id LIKE %s", (pattern,)
            ).fetchone()
            conn.execute("DELETE FROM chat_events WHERE session_id LIKE %s", (pattern,))
        count: Any = row[0] if row is not None else 0
        return int(count)

    @staticmethod
    def _load(conn: psycopg.Connection[tuple[object, ...]], session_id: str) -> list[Event]:
        rows = conn.execute(
            "SELECT payload FROM chat_events WHERE session_id = %s ORDER BY seq", (session_id,)
        ).fetchall()
        out: list[Event] = []
        for (payload,) in rows:
            if not isinstance(payload, dict):
                raise TypeError(f"chat_events.payload must be a JSON object, got {type(payload).__name__}")
            out.append(from_json(payload))
        return out
