"""Bot definition: the data a builder edits (docs/engine-core-plan.md §3.1).

Keys of `intents`, `entities`, `scenarios` and `responses` are stable IDs;
`label` is the editable display name. Events and session state refer to IDs
only, so renaming a label never breaks a running conversation.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*\}\}")
SYSTEM_ENTITIES = frozenset({"sys.number", "sys.date", "sys.time"})
# Responses the engine itself needs; every bot must define them.
REQUIRED_RESPONSES = frozenset({"fallback", "cancelled", "restarted", "error", "clarify", "input_invalid"})


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Settings(Model):
    intent_threshold: float = Field(0.30, ge=0, le=1)
    similarity_threshold: float = Field(0.39, ge=0, le=1)
    ambiguity_margin: float = Field(0.10, ge=0, le=1)
    # confidence needed to leave a pending question for another intent
    switch_threshold: float = Field(0.80, ge=0, le=1)
    rerun_window: int = Field(3, ge=0, le=20)  # user turns after a scenario ends during which "부산은?" re-runs it
    max_attempts: int = Field(3, ge=1, le=10)  # failed re-asks of one slot before giving up
    max_steps_per_turn: int = Field(20, ge=1, le=200)
    max_stack_depth: int = Field(5, ge=1, le=20)


class Intent(Model):
    label: str = ""
    examples: list[str] = Field(min_length=1)


class DictionaryEntity(Model):
    kind: Literal["dictionary"]
    label: str = ""
    values: dict[str, list[str]] = Field(min_length=1)  # canonical value -> synonyms

    def canonical(self, value: str) -> str | None:
        """The canonical value for a canonical value or any of its synonyms."""
        if value in self.values:
            return value
        for canon, synonyms in self.values.items():
            if value in synonyms:
                return canon
        return None


class RegexEntity(Model):
    kind: Literal["regex"]
    label: str = ""
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return v


Entity = Annotated[DictionaryEntity | RegexEntity, Field(discriminator="kind")]


class Slot(Model):
    entity: str  # an entity ID or sys.number / sys.date / sys.time
    ask: str  # response ID used to ask for the value


class Response(Model):
    text: str | None = None
    variants: list[str] | None = None  # one is picked at random
    quick_replies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _exactly_one_body(self) -> Response:
        if (self.text is None) == (self.variants is None):
            raise ValueError("a response needs exactly one of 'text' or 'variants'")
        if self.variants is not None and (not self.variants or any(not v.strip() for v in self.variants)):
            raise ValueError("'variants' must be non-empty strings")
        if self.text is not None and not self.text.strip():
            raise ValueError("'text' must not be empty")
        if any(not q.strip() or len(q) > 40 for q in self.quick_replies):
            raise ValueError("quick replies must be 1-40 characters")
        return self

    def bodies(self) -> list[str]:
        return [self.text] if self.text is not None else list(self.variants or [])


class CollectStep(Model):
    collect: str


class ActionStep(Model):
    action: str
    args: dict[str, str] = Field(default_factory=dict)  # values may contain {{ templates }}


class RespondStep(Model):
    respond: str


class CallStep(Model):
    call: str  # scenario ID


class Branch(Model):
    on: str  # template path, e.g. "results.get_weather.ok"
    cases: dict[str, str] = Field(min_length=1)  # rendered value -> scenario ID
    default: str | None = None  # scenario ID; None = continue without calling


class BranchStep(Model):
    branch: Branch


Step = CollectStep | ActionStep | RespondStep | CallStep | BranchStep
STEP_KEYS: dict[str, type[Model]] = {
    "collect": CollectStep, "action": ActionStep, "respond": RespondStep, "call": CallStep, "branch": BranchStep,
}


class Trigger(Model):
    intent: str


class Scenario(Model):
    label: str = ""
    trigger: Trigger | None = None  # None: only reachable through call/branch
    slots: dict[str, Slot] = Field(default_factory=dict)
    steps: list[Step] = Field(min_length=1)
    on_error: str | None = None  # response ID when an action fails; default "error"

    @field_validator("steps", mode="before")
    @classmethod
    def _parse_steps(cls, raw: Any) -> Any:
        if not isinstance(raw, list):
            return raw
        parsed: list[Any] = []
        for i, item in enumerate(raw):
            if isinstance(item, BaseModel):
                parsed.append(item)
                continue
            kinds = [k for k in STEP_KEYS if isinstance(item, dict) and k in item]
            if len(kinds) != 1:
                raise ValueError(f"step {i} must have exactly one of {sorted(STEP_KEYS)}: {item!r}")
            parsed.append(STEP_KEYS[kinds[0]].model_validate(item))
        return parsed


class BotDefinition(Model):
    id: ID
    version: str  # content hash of the source files
    label: str = ""
    settings: Settings = Settings()
    intents: dict[ID, Intent] = Field(default_factory=dict)
    entities: dict[ID, Entity] = Field(default_factory=dict)
    scenarios: dict[ID, Scenario] = Field(min_length=1)
    responses: dict[ID, Response] = Field(min_length=1)
