"""Bot repository: what the builder edits (drafts), what was trained (versions)
and what is live (deployment history). Trained model files live in artifacts/.

A version is the content hash of its document, so saving the same document
twice gives the same version (idempotent).
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Jsonb


class VersionNotFound(KeyError):
    pass


class RevisionConflict(Exception):
    """The draft changed since the caller read it (another tab or editor saved first)."""

    def __init__(self, current: int | None) -> None:
        super().__init__(f"the draft is at revision {current}")
        self.current = current


@dataclass(frozen=True)
class VersionInfo:
    version: str
    number: int  # 1, 2, 3... per bot, in the order versions were trained (shown as v1, v2...)
    created_at: str  # ISO 8601, UTC
    macro_f1: float | None


@dataclass(frozen=True)
class Deployment:
    version: str
    number: int
    note: str
    deployed_at: str  # ISO 8601, UTC
    kb_version: str | None = None  # None: no knowledge base for this deployment


@dataclass(frozen=True)
class KbVersionInfo:
    version: str  # content hash, computed by the caller
    number: int  # 1, 2, 3... per bot, in the order versions were added
    created_at: str  # ISO 8601, UTC
    faq: int  # number of FAQ entries
    chunks: int  # number of document chunks


@dataclass(frozen=True)
class Draft:
    document: dict[str, Any]
    revision: int  # 1 on first save, +1 on every save


@dataclass(frozen=True)
class Edit:
    """One draft save and what it changed (bot/changes.py summarize), for the builder's 작업이력."""

    revision: int
    saved_at: str  # ISO 8601, UTC
    changes: list[dict[str, Any]]


@dataclass(frozen=True)
class Mark:
    """What the builder did with a user sentence from the training queue (학습대기): added it to an
    intent as an example, ignored it, or put it back (pending)."""

    key: str  # the sentence as grouped (training_queue.key)
    text: str
    status: str  # "added" | "ignored" | "pending"
    intent: str | None  # for "added"
    marked_at: str  # ISO 8601, UTC


MARK_STATUSES = ("added", "ignored", "pending")
SKILL_ERRORS_KEPT = 100  # per bot, like Kakao's 오류 내역


@dataclass(frozen=True)
class SecretInfo:
    name: str
    updated_at: str  # the value itself is never read back through the builder API


class BotRepository(Protocol):
    def bots(self) -> list[str]: ...
    def get_draft(self, bot_id: str) -> dict[str, Any] | None: ...
    def draft(self, bot_id: str) -> Draft | None: ...
    def save_draft(self, bot_id: str, document: dict[str, Any], expected_revision: int | None = None,
                   changes: list[dict[str, Any]] | None = None) -> int:
        """Store the draft and return its new revision. With expected_revision, raise
        RevisionConflict unless the stored draft is at that revision (0: must not exist yet).
        Non-empty `changes` are recorded as that revision's edit, in the same write."""
        ...
    def edits(self, bot_id: str, limit: int = 50) -> list[Edit]: ...  # newest first
    def mark_utterances(self, bot_id: str, items: list[tuple[str, str]], status: str,
                        intent: str | None = None) -> None:
        """Record a status for (key, text) pairs from the training queue."""
        ...
    def utterance_marks(self, bot_id: str, limit: int | None = None) -> list[Mark]: ...  # newest first
    def latest_marks(self, bot_id: str) -> list[Mark]: ...  # the newest mark of each sentence
    def set_secret(self, bot_id: str, name: str, value: str) -> None: ...
    def delete_secret(self, bot_id: str, name: str) -> bool: ...  # False: there was none
    def secret(self, bot_id: str, name: str) -> str | None: ...  # for skill calls only
    def secrets(self, bot_id: str) -> list[SecretInfo]: ...  # names, sorted
    def add_skill_error(self, bot_id: str, error: dict[str, Any]) -> None: ...  # keeps the newest SKILL_ERRORS_KEPT
    def skill_errors(self, bot_id: str, limit: int = SKILL_ERRORS_KEPT) -> list[dict[str, Any]]: ...  # newest first
    def add_version(
        self, bot_id: str, version: str, document: dict[str, Any], macro_f1: float | None
    ) -> VersionInfo: ...
    def get_version(self, bot_id: str, version: str) -> dict[str, Any]: ...
    def versions(self, bot_id: str) -> list[VersionInfo]: ...  # newest first
    def kb_draft(self, bot_id: str) -> Draft | None: ...
    def save_kb_draft(self, bot_id: str, document: dict[str, Any], expected_revision: int | None = None) -> int:
        """Same revision rules as save_draft: returns the new revision (1 on first save, +1 each
        save); with expected_revision, raise RevisionConflict unless the stored KB draft is at that
        revision (0 means: must not exist yet). No edit history for KB drafts."""
        ...
    def add_kb_version(
        self, bot_id: str, version: str, document: dict[str, Any], faq: int, chunks: int
    ) -> KbVersionInfo:
        """Store a built knowledge-base version. Adding a version that already exists returns the
        existing info unchanged (idempotent: same number, same created_at)."""
        ...
    def get_kb_version(self, bot_id: str, version: str) -> dict[str, Any]: ...  # raises VersionNotFound
    def kb_versions(self, bot_id: str) -> list[KbVersionInfo]: ...  # newest first; [] for none
    def deploy(self, bot_id: str, version: str, note: str = "", kb_version: str | None = None) -> None:
        """As today, plus kb_version: None = no knowledge base; an unknown kb_version raises
        VersionNotFound and records nothing."""
        ...
    def deployments(self, bot_id: str) -> list[str]: ...  # newest first; [0] is live
    def history(self, bot_id: str) -> list[Deployment]: ...  # newest first


def _check_mark(status: str, intent: str | None) -> None:
    if status not in MARK_STATUSES:
        raise ValueError(f"status must be one of {MARK_STATUSES}")
    if (status == "added") != (intent is not None):
        raise ValueError("an intent is given exactly when the status is 'added'")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat(timespec="seconds")


class InMemoryRepository:
    def __init__(self) -> None:
        self._drafts: dict[str, Draft] = {}
        self._versions: dict[str, dict[str, tuple[VersionInfo, dict[str, Any]]]] = {}
        self._kb_drafts: dict[str, Draft] = {}
        self._kb_versions: dict[str, dict[str, tuple[KbVersionInfo, dict[str, Any]]]] = {}
        self._deployments: dict[str, list[Deployment]] = {}
        self._edits: dict[str, list[Edit]] = {}  # newest first
        self._marks: dict[str, list[Mark]] = {}  # newest first
        self._secrets: dict[str, dict[str, tuple[str, str]]] = {}  # bot -> name -> (value, updated_at)
        self._skill_errors: dict[str, list[dict[str, Any]]] = {}  # newest first
        self._lock = threading.Lock()

    def bots(self) -> list[str]:
        with self._lock:
            return sorted(set(self._drafts) | set(self._versions))

    def get_draft(self, bot_id: str) -> dict[str, Any] | None:
        draft = self.draft(bot_id)
        return draft.document if draft is not None else None

    def draft(self, bot_id: str) -> Draft | None:
        with self._lock:
            draft = self._drafts.get(bot_id)
            return Draft(dict(draft.document), draft.revision) if draft is not None else None

    def save_draft(self, bot_id: str, document: dict[str, Any], expected_revision: int | None = None,
                   changes: list[dict[str, Any]] | None = None) -> int:
        with self._lock:
            current = self._drafts.get(bot_id)
            at = current.revision if current is not None else 0
            if expected_revision is not None and expected_revision != at:
                raise RevisionConflict(at or None)
            self._drafts[bot_id] = Draft(dict(document), at + 1)
            if changes:
                self._edits.setdefault(bot_id, []).insert(0, Edit(at + 1, _now(), [dict(c) for c in changes]))
            return at + 1

    def edits(self, bot_id: str, limit: int = 50) -> list[Edit]:
        with self._lock:
            return list(self._edits.get(bot_id, []))[:max(limit, 0)]

    def mark_utterances(self, bot_id: str, items: list[tuple[str, str]], status: str,
                        intent: str | None = None) -> None:
        _check_mark(status, intent)
        with self._lock:
            marks = self._marks.setdefault(bot_id, [])
            now = _now()
            for key, text in items:
                marks.insert(0, Mark(key, text, status, intent, now))

    def utterance_marks(self, bot_id: str, limit: int | None = None) -> list[Mark]:
        with self._lock:
            marks = list(self._marks.get(bot_id, []))
        return marks if limit is None else marks[:max(limit, 0)]

    def set_secret(self, bot_id: str, name: str, value: str) -> None:
        with self._lock:
            self._secrets.setdefault(bot_id, {})[name] = (value, _now())

    def delete_secret(self, bot_id: str, name: str) -> bool:
        with self._lock:
            return self._secrets.get(bot_id, {}).pop(name, None) is not None

    def secret(self, bot_id: str, name: str) -> str | None:
        with self._lock:
            entry = self._secrets.get(bot_id, {}).get(name)
        return entry[0] if entry is not None else None

    def secrets(self, bot_id: str) -> list[SecretInfo]:
        with self._lock:
            return [SecretInfo(n, at) for n, (_, at) in sorted(self._secrets.get(bot_id, {}).items())]

    def add_skill_error(self, bot_id: str, error: dict[str, Any]) -> None:
        with self._lock:
            errors = self._skill_errors.setdefault(bot_id, [])
            errors.insert(0, dict(error))
            del errors[SKILL_ERRORS_KEPT:]

    def skill_errors(self, bot_id: str, limit: int = SKILL_ERRORS_KEPT) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._skill_errors.get(bot_id, [])[:max(limit, 0)]]

    def latest_marks(self, bot_id: str) -> list[Mark]:
        latest: dict[str, Mark] = {}
        for m in self.utterance_marks(bot_id):  # newest first
            latest.setdefault(m.key, m)
        return list(latest.values())

    def add_version(self, bot_id: str, version: str, document: dict[str, Any], macro_f1: float | None) -> VersionInfo:
        with self._lock:
            versions = self._versions.setdefault(bot_id, {})
            if version not in versions:
                versions[version] = (VersionInfo(version, len(versions) + 1, _now(), macro_f1), dict(document))
            return versions[version][0]

    def get_version(self, bot_id: str, version: str) -> dict[str, Any]:
        with self._lock:
            try:
                return dict(self._versions[bot_id][version][1])
            except KeyError as exc:
                raise VersionNotFound(f"{bot_id}@{version}") from exc

    def versions(self, bot_id: str) -> list[VersionInfo]:
        with self._lock:
            infos = [info for info, _ in self._versions.get(bot_id, {}).values()]
        return list(reversed(infos))  # insertion order = creation order

    def kb_draft(self, bot_id: str) -> Draft | None:
        with self._lock:
            draft = self._kb_drafts.get(bot_id)
            return Draft(dict(draft.document), draft.revision) if draft is not None else None

    def save_kb_draft(self, bot_id: str, document: dict[str, Any], expected_revision: int | None = None) -> int:
        with self._lock:
            current = self._kb_drafts.get(bot_id)
            at = current.revision if current is not None else 0
            if expected_revision is not None and expected_revision != at:
                raise RevisionConflict(at or None)
            self._kb_drafts[bot_id] = Draft(dict(document), at + 1)
            return at + 1

    def add_kb_version(
        self, bot_id: str, version: str, document: dict[str, Any], faq: int, chunks: int
    ) -> KbVersionInfo:
        with self._lock:
            versions = self._kb_versions.setdefault(bot_id, {})
            if version not in versions:
                versions[version] = (KbVersionInfo(version, len(versions) + 1, _now(), faq, chunks), dict(document))
            return versions[version][0]

    def get_kb_version(self, bot_id: str, version: str) -> dict[str, Any]:
        with self._lock:
            try:
                return dict(self._kb_versions[bot_id][version][1])
            except KeyError as exc:
                raise VersionNotFound(f"{bot_id}@{version}") from exc

    def kb_versions(self, bot_id: str) -> list[KbVersionInfo]:
        with self._lock:
            infos = [info for info, _ in self._kb_versions.get(bot_id, {}).values()]
        return list(reversed(infos))  # insertion order = creation order

    def deploy(self, bot_id: str, version: str, note: str = "", kb_version: str | None = None) -> None:
        with self._lock:
            entry = self._versions.get(bot_id, {}).get(version)
            if entry is None:
                raise VersionNotFound(f"{bot_id}@{version}")
            if kb_version is not None and kb_version not in self._kb_versions.get(bot_id, {}):
                raise VersionNotFound(f"{bot_id}@{kb_version}")
            self._deployments.setdefault(bot_id, []).insert(
                0, Deployment(version, entry[0].number, note, _now(), kb_version)
            )

    def deployments(self, bot_id: str) -> list[str]:
        return [d.version for d in self.history(bot_id)]

    def history(self, bot_id: str) -> list[Deployment]:
        with self._lock:
            return list(self._deployments.get(bot_id, []))


SCHEMA = """
CREATE TABLE IF NOT EXISTS bot_drafts (
    bot_id     text PRIMARY KEY,
    document   jsonb NOT NULL,
    revision   integer NOT NULL DEFAULT 1,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS bot_versions (
    bot_id     text NOT NULL,
    version    text NOT NULL,
    document   jsonb NOT NULL,
    macro_f1   double precision,
    number     integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    seq        bigserial,
    PRIMARY KEY (bot_id, version)
);
CREATE TABLE IF NOT EXISTS bot_deployments (
    id          bigserial PRIMARY KEY,
    bot_id      text NOT NULL,
    version     text NOT NULL,
    note        text NOT NULL DEFAULT '',
    deployed_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (bot_id, version) REFERENCES bot_versions (bot_id, version)
);
-- databases created before revisions, version numbers and deploy notes
ALTER TABLE bot_drafts ADD COLUMN IF NOT EXISTS revision integer NOT NULL DEFAULT 1;
ALTER TABLE bot_versions ADD COLUMN IF NOT EXISTS number integer;
ALTER TABLE bot_deployments ADD COLUMN IF NOT EXISTS note text NOT NULL DEFAULT '';
UPDATE bot_versions v SET number = r.n
  FROM (SELECT b.bot_id, b.version,
               (SELECT COALESCE(MAX(m.number), 0) FROM bot_versions m WHERE m.bot_id = b.bot_id)
               + row_number() OVER (PARTITION BY b.bot_id ORDER BY b.seq) AS n
          FROM bot_versions b WHERE b.number IS NULL) r
 WHERE v.bot_id = r.bot_id AND v.version = r.version;
CREATE UNIQUE INDEX IF NOT EXISTS bot_versions_number ON bot_versions (bot_id, number);
CREATE TABLE IF NOT EXISTS bot_draft_edits (
    bot_id    text NOT NULL,
    revision  integer NOT NULL,
    changes   jsonb NOT NULL,
    saved_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (bot_id, revision)
);
CREATE TABLE IF NOT EXISTS bot_utterance_marks (
    id         bigserial PRIMARY KEY,
    bot_id     text NOT NULL,
    key        text NOT NULL,
    text       text NOT NULL,
    status     text NOT NULL,
    intent     text,
    marked_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS bot_utterance_marks_bot ON bot_utterance_marks (bot_id, id DESC);
CREATE TABLE IF NOT EXISTS bot_secrets (
    bot_id     text NOT NULL,
    name       text NOT NULL,
    value      text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (bot_id, name)
);
CREATE TABLE IF NOT EXISTS bot_skill_errors (
    id      bigserial PRIMARY KEY,
    bot_id  text NOT NULL,
    error   jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS bot_skill_errors_bot ON bot_skill_errors (bot_id, id DESC);
CREATE TABLE IF NOT EXISTS bot_kb_drafts (
    bot_id     text PRIMARY KEY,
    document   jsonb NOT NULL,
    revision   integer NOT NULL DEFAULT 1,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS bot_kb_versions (
    bot_id     text NOT NULL,
    version    text NOT NULL,
    document   jsonb NOT NULL,
    faq        integer NOT NULL,
    chunks     integer NOT NULL,
    number     integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    seq        bigserial,
    PRIMARY KEY (bot_id, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS bot_kb_versions_number ON bot_kb_versions (bot_id, number);
-- a deployment pairs a bot version with a knowledge-base version (or none)
ALTER TABLE bot_deployments ADD COLUMN IF NOT EXISTS kb_version text;
"""


class PostgresRepository:
    def __init__(self, dsn: str, connect_timeout: int = 5) -> None:
        if not dsn:
            raise ValueError("dsn is required")
        self.dsn, self.connect_timeout = dsn, connect_timeout

    def _connect(self) -> psycopg.Connection[tuple[Any, ...]]:
        return psycopg.connect(self.dsn, connect_timeout=self.connect_timeout)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def bots(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT bot_id FROM bot_drafts UNION SELECT bot_id FROM bot_versions ORDER BY 1"
            ).fetchall()
        return [str(r[0]) for r in rows]

    def get_draft(self, bot_id: str) -> dict[str, Any] | None:
        draft = self.draft(bot_id)
        return draft.document if draft is not None else None

    def draft(self, bot_id: str) -> Draft | None:
        with self._connect() as conn:
            row = conn.execute("SELECT document, revision FROM bot_drafts WHERE bot_id = %s", (bot_id,)).fetchone()
        return Draft(dict(row[0]), int(row[1])) if row else None

    def save_draft(self, bot_id: str, document: dict[str, Any], expected_revision: int | None = None,
                   changes: list[dict[str, Any]] | None = None) -> int:
        with self._connect() as conn:  # one transaction: the draft and its edit record together
            if expected_revision is None:
                row = conn.execute(
                    "INSERT INTO bot_drafts (bot_id, document) VALUES (%s, %s) ON CONFLICT (bot_id) DO UPDATE "
                    "SET document = EXCLUDED.document, revision = bot_drafts.revision + 1, updated_at = now() "
                    "RETURNING revision", (bot_id, Jsonb(document))).fetchone()
            elif expected_revision == 0:
                row = conn.execute(
                    "INSERT INTO bot_drafts (bot_id, document) VALUES (%s, %s) ON CONFLICT (bot_id) DO NOTHING "
                    "RETURNING revision", (bot_id, Jsonb(document))).fetchone()
            else:
                row = conn.execute(
                    "UPDATE bot_drafts SET document = %s, revision = revision + 1, updated_at = now() "
                    "WHERE bot_id = %s AND revision = %s RETURNING revision",
                    (Jsonb(document), bot_id, expected_revision)).fetchone()
            if row is None:
                current = conn.execute("SELECT revision FROM bot_drafts WHERE bot_id = %s", (bot_id,)).fetchone()
                raise RevisionConflict(int(current[0]) if current else None)
            if changes:
                conn.execute("INSERT INTO bot_draft_edits (bot_id, revision, changes) VALUES (%s, %s, %s) "
                             "ON CONFLICT (bot_id, revision) DO UPDATE SET changes = EXCLUDED.changes",
                             (bot_id, int(row[0]), Jsonb(changes)))
        return int(row[0])

    def edits(self, bot_id: str, limit: int = 50) -> list[Edit]:
        with self._connect() as conn:
            rows = conn.execute("SELECT revision, saved_at, changes FROM bot_draft_edits WHERE bot_id = %s "
                                "ORDER BY revision DESC LIMIT %s", (bot_id, max(limit, 0))).fetchall()
        return [Edit(int(r), _iso(at), list(c)) for r, at, c in rows]

    def mark_utterances(self, bot_id: str, items: list[tuple[str, str]], status: str,
                        intent: str | None = None) -> None:
        _check_mark(status, intent)
        if not items:
            return
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany("INSERT INTO bot_utterance_marks (bot_id, key, text, status, intent) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            [(bot_id, key, text, status, intent) for key, text in items])

    def utterance_marks(self, bot_id: str, limit: int | None = None) -> list[Mark]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, text, status, intent, marked_at FROM bot_utterance_marks "
                                "WHERE bot_id = %s ORDER BY id DESC LIMIT %s", (bot_id, limit)).fetchall()
        return [Mark(str(k), str(t), str(s), i, _iso(at)) for k, t, s, i, at in rows]

    def latest_marks(self, bot_id: str) -> list[Mark]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT ON (key) key, text, status, intent, marked_at "
                                "FROM bot_utterance_marks WHERE bot_id = %s ORDER BY key, id DESC",
                                (bot_id,)).fetchall()
        return [Mark(str(k), str(t), str(s), i, _iso(at)) for k, t, s, i, at in rows]

    def add_version(self, bot_id: str, version: str, document: dict[str, Any], macro_f1: float | None) -> VersionInfo:
        for attempt in range(3):  # two new versions of one bot at once can pick the same number
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO bot_versions (bot_id, version, document, macro_f1, number) "
                        "SELECT %s, %s, %s, %s, COALESCE(MAX(number), 0) + 1 FROM bot_versions WHERE bot_id = %s "
                        "ON CONFLICT (bot_id, version) DO NOTHING",
                        (bot_id, version, Jsonb(document), macro_f1, bot_id),
                    )
                    row = conn.execute(
                        "SELECT version, COALESCE(number, 0), created_at, macro_f1 FROM bot_versions "
                        "WHERE bot_id = %s AND version = %s", (bot_id, version)).fetchone()
                break
            except psycopg.errors.UniqueViolation:
                if attempt == 2:
                    raise
        assert row is not None
        return VersionInfo(str(row[0]), int(row[1]), _iso(row[2]), row[3])

    def get_version(self, bot_id: str, version: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT document FROM bot_versions WHERE bot_id = %s AND version = %s",
                               (bot_id, version)).fetchone()
        if row is None:
            raise VersionNotFound(f"{bot_id}@{version}")
        return dict(row[0])

    def versions(self, bot_id: str) -> list[VersionInfo]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT version, COALESCE(number, 0), created_at, macro_f1 FROM bot_versions "
                "WHERE bot_id = %s ORDER BY seq DESC",
                (bot_id,),
            ).fetchall()
        return [VersionInfo(str(v), int(n), _iso(c), f) for v, n, c, f in rows]

    def kb_draft(self, bot_id: str) -> Draft | None:
        with self._connect() as conn:
            row = conn.execute("SELECT document, revision FROM bot_kb_drafts WHERE bot_id = %s", (bot_id,)).fetchone()
        return Draft(dict(row[0]), int(row[1])) if row else None

    def save_kb_draft(self, bot_id: str, document: dict[str, Any], expected_revision: int | None = None) -> int:
        with self._connect() as conn:
            if expected_revision is None:
                row = conn.execute(
                    "INSERT INTO bot_kb_drafts (bot_id, document) VALUES (%s, %s) ON CONFLICT (bot_id) DO UPDATE "
                    "SET document = EXCLUDED.document, revision = bot_kb_drafts.revision + 1, updated_at = now() "
                    "RETURNING revision", (bot_id, Jsonb(document))).fetchone()
            elif expected_revision == 0:
                row = conn.execute(
                    "INSERT INTO bot_kb_drafts (bot_id, document) VALUES (%s, %s) ON CONFLICT (bot_id) DO NOTHING "
                    "RETURNING revision", (bot_id, Jsonb(document))).fetchone()
            else:
                row = conn.execute(
                    "UPDATE bot_kb_drafts SET document = %s, revision = revision + 1, updated_at = now() "
                    "WHERE bot_id = %s AND revision = %s RETURNING revision",
                    (Jsonb(document), bot_id, expected_revision)).fetchone()
            if row is None:
                current = conn.execute("SELECT revision FROM bot_kb_drafts WHERE bot_id = %s", (bot_id,)).fetchone()
                raise RevisionConflict(int(current[0]) if current else None)
        return int(row[0])

    def add_kb_version(
        self, bot_id: str, version: str, document: dict[str, Any], faq: int, chunks: int
    ) -> KbVersionInfo:
        for attempt in range(3):  # two new versions of one bot at once can pick the same number
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT INTO bot_kb_versions (bot_id, version, document, faq, chunks, number) "
                        "SELECT %s, %s, %s, %s, %s, COALESCE(MAX(number), 0) + 1 FROM bot_kb_versions "
                        "WHERE bot_id = %s ON CONFLICT (bot_id, version) DO NOTHING",
                        (bot_id, version, Jsonb(document), faq, chunks, bot_id),
                    )
                    row = conn.execute(
                        "SELECT version, COALESCE(number, 0), created_at, faq, chunks FROM bot_kb_versions "
                        "WHERE bot_id = %s AND version = %s", (bot_id, version)).fetchone()
                break
            except psycopg.errors.UniqueViolation:
                if attempt == 2:
                    raise
        assert row is not None
        return KbVersionInfo(str(row[0]), int(row[1]), _iso(row[2]), int(row[3]), int(row[4]))

    def get_kb_version(self, bot_id: str, version: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT document FROM bot_kb_versions WHERE bot_id = %s AND version = %s",
                               (bot_id, version)).fetchone()
        if row is None:
            raise VersionNotFound(f"{bot_id}@{version}")
        return dict(row[0])

    def kb_versions(self, bot_id: str) -> list[KbVersionInfo]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT version, COALESCE(number, 0), created_at, faq, chunks FROM bot_kb_versions "
                "WHERE bot_id = %s ORDER BY seq DESC",
                (bot_id,),
            ).fetchall()
        return [KbVersionInfo(str(v), int(n), _iso(c), int(f), int(ch)) for v, n, c, f, ch in rows]

    def deploy(self, bot_id: str, version: str, note: str = "", kb_version: str | None = None) -> None:
        try:
            with self._connect() as conn:  # one transaction: validate the KB version, then record together
                if kb_version is not None:
                    found = conn.execute("SELECT 1 FROM bot_kb_versions WHERE bot_id = %s AND version = %s",
                                         (bot_id, kb_version)).fetchone()
                    if found is None:
                        raise VersionNotFound(f"{bot_id}@{kb_version}")
                conn.execute("INSERT INTO bot_deployments (bot_id, version, note, kb_version) VALUES (%s, %s, %s, %s)",
                             (bot_id, version, note, kb_version))
        except psycopg.errors.ForeignKeyViolation as exc:
            raise VersionNotFound(f"{bot_id}@{version}") from exc

    def deployments(self, bot_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT version FROM bot_deployments WHERE bot_id = %s ORDER BY id DESC",
                                (bot_id,)).fetchall()
        return [str(r[0]) for r in rows]

    def history(self, bot_id: str) -> list[Deployment]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT d.version, COALESCE(v.number, 0), d.note, d.deployed_at, d.kb_version FROM bot_deployments d "
                "JOIN bot_versions v ON v.bot_id = d.bot_id AND v.version = d.version "
                "WHERE d.bot_id = %s ORDER BY d.id DESC", (bot_id,)).fetchall()
        return [Deployment(str(v), int(n), str(note), _iso(at), str(kb) if kb is not None else None)
                for v, n, note, at, kb in rows]

    def set_secret(self, bot_id: str, name: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO bot_secrets (bot_id, name, value) VALUES (%s, %s, %s) ON CONFLICT (bot_id, name) "
                         "DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (bot_id, name, value))

    def delete_secret(self, bot_id: str, name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("DELETE FROM bot_secrets WHERE bot_id = %s AND name = %s RETURNING name",
                               (bot_id, name)).fetchone()
        return row is not None

    def secret(self, bot_id: str, name: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM bot_secrets WHERE bot_id = %s AND name = %s",
                               (bot_id, name)).fetchone()
        return str(row[0]) if row is not None else None

    def secrets(self, bot_id: str) -> list[SecretInfo]:
        with self._connect() as conn:
            rows = conn.execute("SELECT name, updated_at FROM bot_secrets WHERE bot_id = %s ORDER BY name",
                                (bot_id,)).fetchall()
        return [SecretInfo(str(n), _iso(at)) for n, at in rows]

    def add_skill_error(self, bot_id: str, error: dict[str, Any]) -> None:
        with self._connect() as conn:  # one transaction: add, then keep only the newest ones
            conn.execute("INSERT INTO bot_skill_errors (bot_id, error) VALUES (%s, %s)", (bot_id, Jsonb(error)))
            conn.execute("DELETE FROM bot_skill_errors WHERE bot_id = %s AND id NOT IN "
                         "(SELECT id FROM bot_skill_errors WHERE bot_id = %s ORDER BY id DESC LIMIT %s)",
                         (bot_id, bot_id, SKILL_ERRORS_KEPT))

    def skill_errors(self, bot_id: str, limit: int = SKILL_ERRORS_KEPT) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT error FROM bot_skill_errors WHERE bot_id = %s ORDER BY id DESC LIMIT %s",
                                (bot_id, max(limit, 0))).fetchall()
        return [dict(r[0]) for r in rows]
