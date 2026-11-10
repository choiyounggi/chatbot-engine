"""Slot value normalization, response templates and actions."""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from chatcore.bot.josa import attach
from chatcore.bot.schema import (
    PLACEHOLDER,
    BasicCard,
    BotDefinition,
    Carousel,
    DictionaryEntity,
    ListCard,
    QuickReply,
    RegexEntity,
    Response,
    RichBody,
    SimpleImage,
)
from chatcore.dialogue.commands import SlotValue
from chatcore.nlu import sysentities

# ---------------------------------------------------------------- slot values


class SlotValueError(ValueError):
    pass


def normalize_slot(bot: BotDefinition, entity: str, raw: SlotValue) -> SlotValue:
    """The canonical value of `raw` for `entity`, or SlotValueError."""
    if entity == "sys.number":
        number: int | float
        if isinstance(raw, int | float):
            number = raw
        else:
            text = raw.strip().replace(",", "")
            try:
                number = int(text)
            except ValueError:
                try:
                    number = float(text)
                except ValueError as exc:
                    raise SlotValueError(f"{raw!r} is not a number") from exc
        if isinstance(number, float) and not math.isfinite(number):
            raise SlotValueError(f"{raw!r} is not a finite number")
        return number
    if entity in sysentities.NEW_ENTITIES:
        try:
            return sysentities.normalize(entity, raw)
        except ValueError as exc:
            raise SlotValueError(str(exc)) from exc
    text = str(raw).strip()
    if not text:
        raise SlotValueError("empty value")
    if entity == "sys.date":
        try:
            return dt.date.fromisoformat(text).isoformat()
        except ValueError as exc:
            raise SlotValueError(f"{raw!r} is not an ISO date (YYYY-MM-DD)") from exc
    if entity == "sys.time":
        try:
            return dt.time.fromisoformat(text).strftime("%H:%M")
        except ValueError as exc:
            raise SlotValueError(f"{raw!r} is not a time (HH:MM)") from exc
    ent = bot.entities.get(entity)
    if isinstance(ent, DictionaryEntity):
        canon = ent.canonical(text)
        if canon is None:
            raise SlotValueError(f"{raw!r} is not a known {entity}")
        return canon
    if isinstance(ent, RegexEntity):
        if re.fullmatch(ent.pattern, text) is None:
            raise SlotValueError(f"{raw!r} does not match {entity}")
        return text
    raise SlotValueError(f"unknown entity {entity!r}")


# ---------------------------------------------------------------- templates



class MissingVariable(KeyError):
    pass


def lookup(context: Mapping[str, Any], path: str) -> Any:
    """Follow a dotted path through nested mappings; MissingVariable if absent or None."""
    cur: Any = context
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise MissingVariable(path)
        cur = cur[part]
    if cur is None:
        raise MissingVariable(path)
    return cur


def as_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render(template: str, context: Mapping[str, Any]) -> str:
    """Replace `{{ path }}` / `{{ path|은 }}` placeholders. No expressions, no code."""

    def sub(m: re.Match[str]) -> str:
        text = as_text(lookup(context, m.group(1)))
        return attach(text, m.group(2)) if m.group(2) else text

    return PLACEHOLDER.sub(sub, template)


class RenderError(ValueError):
    """A rendered rich response is not valid any more (e.g. a template gave an image address that is not a URL)."""


@dataclass(frozen=True)
class BotMessage:
    response: str  # response ID
    text: str  # the text, or for a rich response a plain-text summary (for text-only channels and logs)
    replies: tuple[QuickReply, ...] = ()
    rich: RichBody | None = None  # the rendered card/image/list/carousel
    kind: str = "text"  # "text" or one of RICH_KINDS

    @property
    def quick_replies(self) -> tuple[str, ...]:
        """The quick reply labels."""
        return tuple(r.label for r in self.replies)


def label_replies(labels: Sequence[str]) -> tuple[QuickReply, ...]:
    """Message quick replies from bare labels (built by the engine, e.g. clarify; not length-checked)."""
    return tuple(QuickReply.model_construct(label=label, action="message", message_text=None, block_id=None)
                 for label in labels)


def render_response(response_id: str, response: Response, context: Mapping[str, Any], rng: random.Random) -> BotMessage:
    """MissingVariable when a template's value is absent, RenderError when a rendered rich body is invalid."""
    replies = tuple(
        QuickReply.model_construct(label=render(q.label, context), action=q.action, block_id=q.block_id,
                                   message_text=render(q.message_text, context) if q.message_text else None)
        for q in response.replies())
    rich = response.rich
    if rich is None:
        return BotMessage(response_id, render(rng.choice(response.bodies()), context), replies)
    try:
        rendered = type(rich).model_validate(_render_all(rich.model_dump(exclude_none=True), context))
    except ValidationError as exc:
        raise RenderError(f"{response_id}: {exc.errors()[0]['msg']}") from exc
    return BotMessage(response_id, summary(rendered), replies, rendered, response.kind)


def _render_all(value: Any, context: Mapping[str, Any], key: str = "") -> Any:
    if isinstance(value, str):
        return value if key in ("action", "type", "block_id") else render(value, context)
    if isinstance(value, dict):
        return {k: _render_all(v, context, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_all(v, context, key) for v in value]
    return value


def summary(body: RichBody) -> str:
    """Plain text for a rich body: what a text-only channel shows."""
    match body:
        case SimpleImage(alt_text=alt):
            return alt
        case BasicCard(title=title, description=description, thumbnail=thumb):
            parts = [p for p in (title, description) if p]
            return "\n".join(parts) if parts else ((thumb.alt_text or "") if thumb else "")
        case ListCard(header=header, items=items):
            return "\n".join([header.title, *(f"- {i.title}" for i in items)])
        case Carousel(items=cards):
            return "\n\n".join(summary(c) for c in cards)


# ---------------------------------------------------------------- actions


class ActionError(RuntimeError):
    pass


ActionFn = Callable[[dict[str, str]], Mapping[str, Any]]


@dataclass(frozen=True)
class _Registered:
    fn: ActionFn
    args_model: type[BaseModel] | None


class ActionRegistry:
    """Named functions a scenario's `action` step may call. Results must be
    JSON-serializable mappings (they are stored in the event log). Each call
    gets `timeout` seconds; a slow action fails the step instead of the turn
    (the worker thread may finish later, so actions must tolerate that)."""

    def __init__(self, timeout: float | None = None) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        self._actions: dict[str, _Registered] = {}
        self.timeout = timeout
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()

    def register(self, name: str, fn: ActionFn, args_model: type[BaseModel] | None = None) -> None:
        if not name or name in self._actions:
            raise ValueError(f"action name {name!r} is empty or already registered")
        self._actions[name] = _Registered(fn, args_model)

    def names(self) -> set[str]:
        return set(self._actions)

    def run(self, name: str, args: dict[str, str]) -> dict[str, Any]:
        entry = self._actions.get(name)
        if entry is None:
            raise ActionError(f"unknown action {name!r}")
        if entry.args_model is not None:
            try:
                entry.args_model.model_validate(args)
            except ValidationError as exc:
                raise ActionError(f"bad arguments for {name}: {exc.errors()}") from exc
        try:
            if self.timeout is None:
                result = entry.fn(args)
            else:
                with self._pool_lock:
                    if self._pool is None:
                        # sized for the API's request threadpool: timed-out calls keep their worker until
                        # their own (network) timeout ends, so a small pool would starve later requests
                        self._pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="action")
                    pool = self._pool
                future = pool.submit(entry.fn, args)
                try:
                    result = future.result(timeout=self.timeout)
                except FutureTimeout as exc:
                    # FutureTimeout is the builtin TimeoutError: it may be the action's own error
                    if future.done():
                        raise
                    raise ActionError(f"{name} timed out after {self.timeout}s") from exc
        except ActionError:
            raise
        except Exception as exc:
            raise ActionError(f"{name} raised {type(exc).__name__}: {exc}") from exc
        if not isinstance(result, Mapping):
            raise ActionError(f"{name} returned {type(result).__name__}, expected a mapping")
        try:  # store exactly what a replay will read back (no NaN, tuples become lists, keys become strings)
            out: dict[str, Any] = json.loads(json.dumps(dict(result), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ActionError(f"{name} returned a value that is not JSON-serializable: {exc}") from exc
        return out


@dataclass
class StubAction:
    """Test double: returns `result` (or raises `error`) and records calls."""

    result: Mapping[str, Any] = field(default_factory=dict)
    error: Exception | None = None
    calls: list[dict[str, str]] = field(default_factory=list)

    def __call__(self, args: dict[str, str]) -> Mapping[str, Any]:
        self.calls.append(dict(args))
        if self.error is not None:
            raise self.error
        return self.result
