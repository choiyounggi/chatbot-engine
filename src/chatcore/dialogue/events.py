"""Tracker events: the append-only record of a session.

Session state is never stored; it is recomputed by applying these events in
order (`chatcore.dialogue.state.fold`). Every change the engine makes —
including which step a scenario is on — is an event, so replaying the log
gives exactly the state the engine had.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from chatcore.dialogue.commands import SlotValue


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionStarted(_Event):
    type: Literal["session_started"] = "session_started"
    bot_id: str
    bot_version: str
    at: str | None = None  # ISO 8601 time (absent in logs written before session timeouts existed)


class UserUttered(_Event):
    type: Literal["user_uttered"] = "user_uttered"
    text: str
    at: str | None = None  # ISO 8601 time


class CommandsReceived(_Event):
    type: Literal["commands_received"] = "commands_received"
    commands: list[dict[str, Any]]
    nlu: dict[str, Any] = Field(default_factory=dict)  # how the commands were derived (for logs and the builder)


class CommandRejected(_Event):
    type: Literal["command_rejected"] = "command_rejected"
    command: dict[str, Any]
    reason: str


class ScenarioStarted(_Event):
    type: Literal["scenario_started"] = "scenario_started"
    scenario: str
    called: bool = False  # started by a parent's call/branch step (cancelled together with it)


class StepEntered(_Event):
    """The top scenario moved to step `index`."""

    type: Literal["step_entered"] = "step_entered"
    scenario: str
    index: int


class SlotSet(_Event):
    type: Literal["slot_set"] = "slot_set"
    scenario: str
    name: str
    value: SlotValue


class SlotRejected(_Event):
    type: Literal["slot_rejected"] = "slot_rejected"
    scenario: str
    name: str
    value: SlotValue
    reason: str


class PromptIssued(_Event):
    """The top scenario asked for `slot` and now waits for it."""

    type: Literal["prompt_issued"] = "prompt_issued"
    scenario: str
    slot: str


class AttemptFailed(_Event):
    """A user turn answered the pending prompt without filling the slot."""

    type: Literal["attempt_failed"] = "attempt_failed"
    scenario: str
    slot: str


class ActionExecuted(_Event):
    type: Literal["action_executed"] = "action_executed"
    scenario: str
    action: str
    result: dict[str, Any]


class ActionFailed(_Event):
    type: Literal["action_failed"] = "action_failed"
    scenario: str
    action: str
    error: str


class ScenarioCompleted(_Event):
    type: Literal["scenario_completed"] = "scenario_completed"
    scenario: str


class ScenarioCancelled(_Event):
    type: Literal["scenario_cancelled"] = "scenario_cancelled"
    scenario: str
    reason: str


class SessionRestarted(_Event):
    type: Literal["session_restarted"] = "session_restarted"


class BotUttered(_Event):
    type: Literal["bot_uttered"] = "bot_uttered"
    response: str
    text: str
    quick_replies: list[str] = Field(default_factory=list)


class TurnFailed(_Event):
    type: Literal["turn_failed"] = "turn_failed"
    reason: str


Event = Annotated[
    SessionStarted | UserUttered | CommandsReceived | CommandRejected | ScenarioStarted | StepEntered | SlotSet
    | SlotRejected | PromptIssued | AttemptFailed | ActionExecuted | ActionFailed | ScenarioCompleted
    | ScenarioCancelled | SessionRestarted | BotUttered | TurnFailed,
    Field(discriminator="type"),
]
EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


def to_json(event: Event) -> dict[str, Any]:
    return event.model_dump(mode="json")


def from_json(data: dict[str, Any]) -> Event:
    return EVENT_ADAPTER.validate_python(data)
