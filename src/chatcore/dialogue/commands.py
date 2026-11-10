"""Commands: the closed set of things understanding (NLU) may ask the engine to do.

Anything outside this set cannot reach the scenario engine. The dict form
(`to_dict` / `from_dict`) is what logs and conversation tests use.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

SlotValue = str | int | float


@dataclass(frozen=True)
class StartScenario:
    scenario: str
    slots: Mapping[str, SlotValue] = field(default_factory=dict)


@dataclass(frozen=True)
class SetSlot:
    name: str
    value: SlotValue


@dataclass(frozen=True)
class CancelScenario:
    pass


@dataclass(frozen=True)
class Restart:
    pass


@dataclass(frozen=True)
class Escape:
    """The bot's escape phrase: end every running scenario, then run the escape block."""


@dataclass(frozen=True)
class Clarify:
    intents: tuple[str, ...]


@dataclass(frozen=True)
class RerunScenario:
    """Run a recently completed scenario again with one slot changed ("부산은?")."""

    scenario: str
    slot: str
    value: SlotValue


@dataclass(frozen=True)
class Fallback:
    pass


Command = StartScenario | SetSlot | CancelScenario | Restart | Escape | Clarify | RerunScenario | Fallback


class CommandFormatError(ValueError):
    pass


def to_dict(cmd: Command) -> dict[str, Any]:
    match cmd:
        case StartScenario(scenario, slots):
            return {"start": scenario, "slots": dict(slots)} if slots else {"start": scenario}
        case SetSlot(name, value):
            return {"set": {name: value}}
        case CancelScenario():
            return {"cancel": True}
        case Restart():
            return {"restart": True}
        case Escape():
            return {"escape": True}
        case Clarify(intents):
            return {"clarify": list(intents)}
        case RerunScenario(scenario, slot, value):
            return {"rerun": scenario, "slot": slot, "value": value}
        case Fallback():
            return {"fallback": True}


def _value(v: Any, where: str) -> SlotValue:
    if isinstance(v, bool) or not isinstance(v, str | int | float):
        raise CommandFormatError(f"{where}: slot value must be a string or number, got {v!r}")
    if isinstance(v, float) and not math.isfinite(v):
        raise CommandFormatError(f"{where}: slot value must be finite, got {v!r}")
    return v


def from_dict(d: Mapping[str, Any]) -> Command:
    if not isinstance(d, Mapping) or not d:
        raise CommandFormatError(f"command must be a non-empty mapping, got {d!r}")
    if "start" in d:
        slots = d.get("slots") or {}
        if not isinstance(slots, Mapping):
            raise CommandFormatError(f"start.slots must be a mapping: {d!r}")
        return StartScenario(str(d["start"]), {str(k): _value(v, "start.slots") for k, v in slots.items()})
    if "set" in d:
        pair = d["set"]
        if not isinstance(pair, Mapping) or len(pair) != 1:
            raise CommandFormatError(f"set needs exactly one name: value pair: {d!r}")
        ((name, value),) = pair.items()
        return SetSlot(str(name), _value(value, "set"))
    if "rerun" in d:
        if "slot" not in d or "value" not in d:
            raise CommandFormatError(f"rerun needs slot and value: {d!r}")
        return RerunScenario(str(d["rerun"]), str(d["slot"]), _value(d["value"], "rerun"))
    if "clarify" in d:
        intents = d["clarify"]
        if not isinstance(intents, list) or len(intents) < 2:
            raise CommandFormatError(f"clarify needs a list of at least two intents: {d!r}")
        return Clarify(tuple(str(i) for i in intents))
    for key, cls in (("cancel", CancelScenario), ("restart", Restart), ("escape", Escape), ("fallback", Fallback)):
        if key in d:
            return cls()
    raise CommandFormatError(f"unknown command: {d!r}")
