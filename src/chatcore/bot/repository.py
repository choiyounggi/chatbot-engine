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


@dataclass(frozen=True)
class VersionInfo:
    version: str
    created_at: str  # ISO 8601, UTC
    macro_f1: float | None


class BotRepository(Protocol):
    def bots(self) -> list[str]: ...
    def get_draft(self, bot_id: str) -> dict[str, Any] | None: ...
    def save_draft(self, bot_id: str, document: dict[str, Any]) -> None: ...
    def add_version(
        self, bot_id: str, version: str, document: dict[str, Any], macro_f1: float | None
    ) -> VersionInfo: ...
    def get_version(self, bot_id: str, version: str) -> dict[str, Any]: ...
    def versions(self, bot_id: str) -> list[VersionInfo]: ...  # newest first
    def deploy(self, bot_id: str, version: str) -> None: ...
    def deployments(self, bot_id: str) -> list[str]: ...  # newest first; [0] is live


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class InMemoryRepository:
    def __init__(self) -> None:
        self._drafts: dict[str, dict[str, Any]] = {}
        self._versions: dict[str, dict[str, tuple[VersionInfo, dict[str, Any]]]] = {}
        self._deployments: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def bots(self) -> list[str]:
        with self._lock:
            return sorted(set(self._drafts) | set(self._versions))

    def get_draft(self, bot_id: str) -> dict[str, Any] | None:
        with self._lock:
            draft = self._drafts.get(bot_id)
            return dict(draft) if draft is not None else None

    def save_draft(self, bot_id: str, document: dict[str, Any]) -> None:
        with self._lock:
            self._drafts[bot_id] = dict(document)

    def add_version(self, bot_id: str, version: str, document: dict[str, Any], macro_f1: float | None) -> VersionInfo:
        with self._lock:
            versions = self._versions.setdefault(bot_id, {})
            if version not in versions:
                versions[version] = (VersionInfo(version, _now(), macro_f1), dict(document))
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

    def deploy(self, bot_id: str, version: str) -> None:
        with self._lock:
            if version not in self._versions.get(bot_id, {}):
                raise VersionNotFound(f"{bot_id}@{version}")
            self._deployments.setdefault(bot_id, []).insert(0, version)

    def deployments(self, bot_id: str) -> list[str]:
        with self._lock:
            return list(self._deployments.get(bot_id, []))


SCHEMA = """
CREATE TABLE IF NOT EXISTS bot_drafts (
    bot_id     text PRIMARY KEY,
    document   jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS bot_versions (
    bot_id     text NOT NULL,
    version    text NOT NULL,
    document   jsonb NOT NULL,
    macro_f1   double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    seq        bigserial,
    PRIMARY KEY (bot_id, version)
);
CREATE TABLE IF NOT EXISTS bot_deployments (
    id          bigserial PRIMARY KEY,
    bot_id      text NOT NULL,
    version     text NOT NULL,
    deployed_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (bot_id, version) REFERENCES bot_versions (bot_id, version)
);
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
        with self._connect() as conn:
            row = conn.execute("SELECT document FROM bot_drafts WHERE bot_id = %s", (bot_id,)).fetchone()
        return dict(row[0]) if row else None

    def save_draft(self, bot_id: str, document: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bot_drafts (bot_id, document) VALUES (%s, %s) "
                "ON CONFLICT (bot_id) DO UPDATE SET document = EXCLUDED.document, updated_at = now()",
                (bot_id, Jsonb(document)),
            )

    def add_version(self, bot_id: str, version: str, document: dict[str, Any], macro_f1: float | None) -> VersionInfo:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bot_versions (bot_id, version, document, macro_f1) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (bot_id, version) DO NOTHING",
                (bot_id, version, Jsonb(document), macro_f1),
            )
            row = conn.execute(
                "SELECT version, created_at, macro_f1 FROM bot_versions WHERE bot_id = %s AND version = %s",
                (bot_id, version),
            ).fetchone()
        assert row is not None
        return VersionInfo(str(row[0]), row[1].astimezone(dt.UTC).isoformat(timespec="seconds"), row[2])

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
                "SELECT version, created_at, macro_f1 FROM bot_versions WHERE bot_id = %s ORDER BY seq DESC",
                (bot_id,),
            ).fetchall()
        return [VersionInfo(str(v), c.astimezone(dt.UTC).isoformat(timespec="seconds"), f) for v, c, f in rows]

    def deploy(self, bot_id: str, version: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute("INSERT INTO bot_deployments (bot_id, version) VALUES (%s, %s)", (bot_id, version))
        except psycopg.errors.ForeignKeyViolation as exc:
            raise VersionNotFound(f"{bot_id}@{version}") from exc

    def deployments(self, bot_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT version FROM bot_deployments WHERE bot_id = %s ORDER BY id DESC",
                                (bot_id,)).fetchall()
        return [str(r[0]) for r in rows]
