"""Session state, computed from events (never stored)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from chatcore.dialogue.commands import SlotValue
from chatcore.dialogue.events import (
    ActionExecuted,
    AttemptFailed,
    Event,
    PromptIssued,
    ScenarioCancelled,
    ScenarioCompleted,
    ScenarioStarted,
    SessionRestarted,
    SessionStarted,
    SlotSet,
    StepEntered,
    UserUttered,
)

RECENT_COMPLETED = 5


class StateError(RuntimeError):
    """The event log is inconsistent (e.g. a step event with no scenario)."""


@dataclass
class Frame:
    scenario: str
    step: int = 0
    slots: dict[str, SlotValue] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    awaiting: str | None = None  # slot whose prompt is pending
    attempts: int = 0  # failed answers to the pending prompt
    called: bool = False  # started by the frame below via call/branch


@dataclass(frozen=True)
class Completed:
    scenario: str
    slots: dict[str, SlotValue]
    turn: int  # user turn number when it completed


@dataclass
class DialogueState:
    bot_id: str | None = None
    bot_version: str | None = None
    kb_version: str | None = None  # the knowledge base the session started with
    turns: int = 0  # user turns so far
    last_active: str | None = None  # ISO time of the last user message or session start, when recorded
    stack: list[Frame] = field(default_factory=list)
    completed: list[Completed] = field(default_factory=list)  # most recent last

    @property
    def started(self) -> bool:
        return self.bot_id is not None

    @property
    def top(self) -> Frame | None:
        return self.stack[-1] if self.stack else None

    def frame(self, scenario: str) -> Frame | None:
        """The topmost frame running `scenario`."""
        for f in reversed(self.stack):
            if f.scenario == scenario:
                return f
        return None


def _top(state: DialogueState, event: Event, scenario: str) -> Frame:
    if not state.stack or state.stack[-1].scenario != scenario:
        raise StateError(f"{event.type} for {scenario!r} but the top frame is "
                         f"{state.stack[-1].scenario if state.stack else None!r}")
    return state.stack[-1]


def apply(state: DialogueState, event: Event) -> DialogueState:
    """Apply one event in place and return the state."""
    match event:
        case SessionStarted(bot_id=bot_id, bot_version=version, at=at, kb_version=kb):
            state.bot_id, state.bot_version, state.kb_version = bot_id, version, kb
            state.last_active = at or state.last_active
        case UserUttered(at=at):
            state.turns += 1
            state.last_active = at or state.last_active
        case ScenarioStarted(scenario=sid, called=called):
            state.stack.append(Frame(sid, called=called))
        case StepEntered(scenario=sid, index=i):
            frame = _top(state, event, sid)
            frame.step, frame.awaiting, frame.attempts = i, None, 0
        case SlotSet(scenario=sid, name=name, value=value):
            target = state.frame(sid)
            if target is None:
                raise StateError(f"slot_set for {sid!r} which is not running")
            target.slots[name] = value
            if target.awaiting == name:
                target.awaiting, target.attempts = None, 0
        case PromptIssued(scenario=sid, slot=slot):
            frame = _top(state, event, sid)
            if frame.awaiting != slot:
                frame.awaiting, frame.attempts = slot, 0
        case AttemptFailed(scenario=sid):
            _top(state, event, sid).attempts += 1
        case ActionExecuted(scenario=sid, action=action, result=result):
            _top(state, event, sid).results[action] = result
        case ScenarioCompleted(scenario=sid):
            frame = _top(state, event, sid)
            state.stack.pop()
            state.completed = (state.completed + [Completed(sid, dict(frame.slots), state.turns)])[-RECENT_COMPLETED:]
        case ScenarioCancelled(scenario=sid):
            _top(state, event, sid)
            state.stack.pop()
        case SessionRestarted():
            state.stack.clear()
            state.completed.clear()
        case _:
            pass  # informational events (commands, bot messages, failures) do not change state
    return state


def fold(events: Iterable[Event]) -> DialogueState:
    state = DialogueState()
    for event in events:
        apply(state, event)
    return state
