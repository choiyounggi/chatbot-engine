"""Web chat settings of a bot (docs/engine-core-plan.md "빌더 단계 E-G").

Which sites may talk to a bot from a browser (CORS), and how the chat widget looks. Not versioned:
a save applies at once, like a skill secret.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

log = logging.getLogger(__name__)

MAX_ORIGINS = 20
DEFAULT_COLOR = "#2563EB"
DEFAULT_PLACEHOLDER = "메시지를 입력하세요"
CACHE_SECONDS = 10.0
_ORIGIN = re.compile(r"^https?://(\[[0-9a-f:.]+\]|[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*)(:[0-9]{1,5})?$")


def normalize_origin(origin: str) -> str:
    """An origin as browsers send it (`scheme://host[:port]`, lower case, no default port).
    ValueError otherwise."""
    value = origin.strip().lower()
    match = _ORIGIN.fullmatch(value)
    if match is None:
        raise ValueError(f"{origin!r} is not an origin like https://example.com (no path, no trailing slash)")
    port = match.group(5)
    if port is not None and not 0 < int(port[1:]) <= 65535:
        raise ValueError(f"{origin!r} has a port outside 1-65535")
    if (value.startswith("https://") and port == ":443") or (value.startswith("http://") and port == ":80"):
        return value.removesuffix(port)  # browsers leave the default port out
    return value


class WebSettings(BaseModel):
    """What the builder edits. `title` None: the bot's label."""

    model_config = ConfigDict(extra="forbid")

    allowed_origins: list[str] = Field(default_factory=list, max_length=MAX_ORIGINS)
    title: str | None = Field(None, min_length=1, max_length=40)
    color: str = Field(DEFAULT_COLOR, pattern=r"^#[0-9A-Fa-f]{6}$")
    placeholder: str = Field(DEFAULT_PLACEHOLDER, min_length=1, max_length=60)

    @field_validator("allowed_origins")
    @classmethod
    def _origins(cls, value: list[str]) -> list[str]:
        out = [normalize_origin(o) for o in value]
        if len(set(out)) != len(out):
            raise ValueError("the same origin is listed twice")
        return out

    @field_validator("title", "placeholder")
    @classmethod
    def _text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value.strip() if value is not None else None

    @field_validator("color")
    @classmethod
    def _color(cls, value: str) -> str:
        return value.upper()


def load(stored: dict[str, Any] | None) -> WebSettings | None:
    """Stored settings as this version reads them: keys it does not know are ignored, and a row it
    cannot read at all counts as defaults (no site allowed) rather than breaking every request."""
    if stored is None:
        return None
    try:
        return WebSettings(**{k: v for k, v in stored.items() if k in WebSettings.model_fields})
    except ValidationError:
        log.warning("unreadable web settings %r: using the defaults (no site allowed)", stored)
        return WebSettings()


def public(settings: WebSettings, bot: str, label: str) -> dict[str, Any]:
    """What a widget may read (never the allowed origins)."""
    return {"bot": bot, "title": settings.title or label or bot, "color": settings.color,
            "placeholder": settings.placeholder}


class OriginCache:
    """Allowed origins per bot, read through `load` at most every CACHE_SECONDS (a save in this
    process forgets its bot at once)."""

    def __init__(self, load: Callable[[str], WebSettings | None], clock: Callable[[], float] = time.monotonic,
                 seconds: float = CACHE_SECONDS) -> None:
        self.load, self.clock, self.seconds = load, clock, seconds
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, frozenset[str]]] = {}

    def allowed(self, bot: str, origin: str) -> bool:
        now = self.clock()
        with self._lock:
            entry = self._entries.get(bot)
        if entry is None or now - entry[0] >= self.seconds:
            settings = self.load(bot)
            origins = frozenset(settings.allowed_origins) if settings is not None else frozenset()
            entry = (now, origins)
            with self._lock:
                self._entries[bot] = entry
        return origin.lower() in entry[1]

    def forget(self, bot: str) -> None:
        with self._lock:
            self._entries.pop(bot, None)
