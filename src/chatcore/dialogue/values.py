"""Slot value normalization, response templates and actions."""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from chatcore.bot.schema import PLACEHOLDER, BotDefinition, DictionaryEntity, RegexEntity, Response
from chatcore.dialogue.commands import SlotValue

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
    """Replace `{{ path }}` placeholders. No expressions, no code."""
    return PLACEHOLDER.sub(lambda m: as_text(lookup(context, m.group(1))), template)


@dataclass(frozen=True)
class BotMessage:
    response: str  # response ID
    text: str
    quick_replies: tuple[str, ...] = ()


def render_response(response_id: str, response: Response, context: Mapping[str, Any], rng: random.Random) -> BotMessage:
    body = rng.choice(response.bodies())
    return BotMessage(response_id, render(body, context), tuple(render(q, context) for q in response.quick_replies))


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
    JSON-serializable mappings (they are stored in the event log)."""

    def __init__(self) -> None:
        self._actions: dict[str, _Registered] = {}

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
            result = entry.fn(args)
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
