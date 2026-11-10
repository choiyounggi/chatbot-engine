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
# {{ path }} or {{ path|은 }} (a Korean particle chosen by the value's final consonant, see bot/josa.py)
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*(?:\|\s*([가-힣]+)\s*)?\}\}")
SYSTEM_ENTITIES = frozenset({
    "sys.number", "sys.date", "sys.time", "sys.text", "sys.url",
    "sys.number.ordinal", "sys.number.age", "sys.number.percent", "sys.date.period", "sys.time.period",
    "sys.unit.currency", "sys.unit.length", "sys.unit.weight", "sys.unit.volume", "sys.unit.temperature",
    "sys.unit.speed",
})
# A link or image address: http(s), or a template that renders one (checked again after rendering).
URL = re.compile(r"https?://\S+")
MAX_QUICK_REPLY = 40  # characters; longer is an error (Kakao's 14 is only a warning, bot/loader.py)
# Responses the engine itself needs; every bot must define them.
REQUIRED_RESPONSES = frozenset({"fallback", "cancelled", "restarted", "error", "clarify", "input_invalid"})


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# Builder's 5-step similarity dial (engine-core-plan.md "빌더 단계 E-B4"): a target out-of-scope
# rejection rate, not a fixed number. Deploy training searches intent_threshold/similarity_threshold
# for the pair that best meets it (chatcore.nlu.train._search_thresholds).
SimilarityLevel = Literal["low", "normal", "somewhat_high", "high", "very_high"]


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
    # None: intent_threshold/similarity_threshold above are used as given (today's behaviour).
    # Set: deploy training ignores them and searches for thresholds meeting this level's target
    # out-of-scope rejection rate (needs the bot to have out_of_scope examples).
    similarity_level: SimilarityLevel | None = None
    keyword_match: bool = False  # core-morpheme exact match in addition to literal text (nlu/keyword.py)


class Intent(Model):
    label: str = ""
    examples: list[str] = Field(min_length=1)
    # left out of classifier training and cross-validation (still needs >= 2 non-exact_only
    # intents to train); its examples still match exactly, by text and by keyword match
    exact_only: bool = False


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


SlotDefault = str | int | float


class Validation(Model):
    """Check a value with an action before accepting it (Kakao: parameter validation)."""

    action: str  # called with {"value": <the value as text>, "slot": <slot name>}; must return {"valid": bool}
    fail: str  # response ID said when the value is rejected; the slot is then asked again


class Slot(Model):
    entity: str  # an entity ID or one of SYSTEM_ENTITIES
    ask: str | None = None  # response ID used to ask for the value (required slots only)
    required: bool = True  # optional slots take a value from the user's words but are never asked for
    default: SlotDefault | None = None  # optional slots: the value when the user gave none
    validation: Validation | None = None

    @field_validator("default", mode="before")
    @classmethod
    def _no_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("a default must be text or a number, not true/false")
        return v

    @model_validator(mode="after")
    def _ask_or_optional(self) -> Slot:
        if self.required and self.ask is None:
            raise ValueError("a required slot needs 'ask' (the response that asks for it)")
        if self.required and self.default is not None:
            raise ValueError("a required slot cannot have a default (it would never be asked); set required: false")
        return self


# ---------------------------------------------------------------- responses
# Rich bodies follow Kakao's skill response format field for field, in snake_case
# (simpleImage -> simple_image, basicCard.thumbnail.imageUrl -> basic_card.thumbnail.image_url,
# blockId -> block_id), so a Kakao channel adapter is a key rename. A "block" is a scenario ID.


def _text(v: str | None, what: str) -> str | None:
    if v is not None and not v.strip():
        raise ValueError(f"{what} must not be empty")
    return v


def _url(v: str | None, what: str) -> str | None:
    if v is not None and not (v.strip().startswith("{{") or URL.fullmatch(PLACEHOLDER.sub("x", v.strip()))):
        raise ValueError(f"{what} must start with http:// or https:// (or be a {{{{ template }}}})")
    return v


class Link(Model):
    web: str

    @field_validator("web")
    @classmethod
    def _web(cls, v: str) -> str:
        return _url(v, "link.web") or v


class Thumbnail(Model):
    image_url: str
    alt_text: str | None = None
    link: Link | None = None
    fixed_ratio: bool = False

    @field_validator("image_url")
    @classmethod
    def _image(cls, v: str) -> str:
        return _url(v, "image_url") or v


class Button(Model):
    """Kakao card button: webLink / message / block / phone."""

    label: str = Field(min_length=1, max_length=MAX_QUICK_REPLY)
    action: Literal["web_link", "message", "block", "phone"]
    web_link_url: str | None = None
    message_text: str | None = None  # message and block: what the user is shown to say (default: label)
    block_id: str | None = None  # scenario ID
    phone_number: str | None = None

    @model_validator(mode="after")
    def _fields_for_action(self) -> Button:
        _check_action(self.action, {"web_link": self.web_link_url, "block": self.block_id,
                                    "phone": self.phone_number}, "button")
        _url(self.web_link_url, "web_link_url")
        if self.phone_number is not None and not re.fullmatch(r"\+?[0-9][0-9 -]{1,30}", self.phone_number):
            raise ValueError("phone_number must be digits (with optional +, spaces and hyphens)")
        _text(self.message_text, "message_text")
        if self.action not in ("message", "block") and self.message_text is not None:
            raise ValueError(f"a {self.action} button takes no message_text")
        return self


class QuickReply(Model):
    """Kakao quick reply (바로연결): sends a message, or starts a block without understanding."""

    label: str = Field(min_length=1, max_length=MAX_QUICK_REPLY)
    action: Literal["message", "block"] = "message"
    message_text: str | None = None  # default: label
    block_id: str | None = None  # scenario ID, for action "block"

    @model_validator(mode="after")
    def _fields_for_action(self) -> QuickReply:
        if not self.label.strip():
            raise ValueError("a quick reply label must not be empty")
        _check_action(self.action, {"block": self.block_id}, "quick reply")
        _text(self.message_text, "message_text")
        return self


def _check_action(action: str, targets: dict[str, str | None], what: str) -> None:
    """The field an action needs is set, and no other action's field is."""
    for name, value in targets.items():
        if name == action and value is None:
            raise ValueError(f"a {what} with action {action!r} needs its target field")
        if name != action and value is not None:
            raise ValueError(f"a {what} with action {action!r} cannot set the {name!r} target")


class SimpleImage(Model):
    image_url: str
    alt_text: str = Field(min_length=1)  # what text-only channels show

    @field_validator("image_url")
    @classmethod
    def _image(cls, v: str) -> str:
        return _url(v, "image_url") or v


class BasicCard(Model):
    title: str | None = None
    description: str | None = None
    thumbnail: Thumbnail | None = None
    buttons: list[Button] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_content(self) -> BasicCard:
        _text(self.title, "title")
        _text(self.description, "description")
        if self.title is None and self.description is None and self.thumbnail is None:
            raise ValueError("a card needs a title, a description or a thumbnail")
        return self


class ListItem(Model):
    title: str = Field(min_length=1)
    description: str | None = None
    image_url: str | None = None
    link: Link | None = None

    @field_validator("image_url")
    @classmethod
    def _image(cls, v: str | None) -> str | None:
        return _url(v, "image_url")


class ListHeader(Model):
    title: str = Field(min_length=1)


class ListCard(Model):
    header: ListHeader
    items: list[ListItem] = Field(min_length=1)
    buttons: list[Button] = Field(default_factory=list)


class Carousel(Model):
    type: Literal["basic_card"] = "basic_card"
    items: list[BasicCard] = Field(min_length=1)


RichBody = SimpleImage | BasicCard | ListCard | Carousel
RICH_KINDS = ("simple_image", "basic_card", "list_card", "carousel")


class Response(Model):
    """One speech bubble: text (or random variants) or one rich body, plus quick replies."""

    text: str | None = None
    variants: list[str] | None = None  # one is picked at random
    simple_image: SimpleImage | None = None
    basic_card: BasicCard | None = None
    list_card: ListCard | None = None
    carousel: Carousel | None = None
    quick_replies: list[str | QuickReply] = Field(default_factory=list)  # a plain string sends itself

    @model_validator(mode="after")
    def _exactly_one_body(self) -> Response:
        given = [k for k in ("text", "variants", *RICH_KINDS) if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError("a response needs exactly one of 'text', 'variants', "
                             f"{', '.join(repr(k) for k in RICH_KINDS)}; got {given or 'none'}")
        if self.variants is not None and (not self.variants or any(not v.strip() for v in self.variants)):
            raise ValueError("'variants' must be non-empty strings")
        if self.text is not None and not self.text.strip():
            raise ValueError("'text' must not be empty")
        if any(isinstance(q, str) and (not q.strip() or len(q) > MAX_QUICK_REPLY) for q in self.quick_replies):
            raise ValueError(f"quick replies must be 1-{MAX_QUICK_REPLY} characters")
        return self

    @property
    def kind(self) -> str:
        return next((k for k in RICH_KINDS if getattr(self, k) is not None), "text")

    @property
    def rich(self) -> RichBody | None:
        return self.simple_image or self.basic_card or self.list_card or self.carousel

    def bodies(self) -> list[str]:
        """The text bodies (one is said); empty for a rich response."""
        return [self.text] if self.text is not None else list(self.variants or [])

    def replies(self) -> list[QuickReply]:
        return [QuickReply(label=q) if isinstance(q, str) else q for q in self.quick_replies]

    def buttons(self) -> list[Button]:
        cards = ([self.basic_card] if self.basic_card else []) + (self.carousel.items if self.carousel else [])
        return [b for c in cards for b in c.buttons] + (self.list_card.buttons if self.list_card else [])

    def block_targets(self) -> set[str]:
        """Scenario IDs this response offers to start directly (quick replies and buttons)."""
        targets = [q.block_id for q in self.replies()] + [b.block_id for b in self.buttons()]
        return {t for t in targets if t is not None}

    def texts(self, replies: bool = True) -> list[str]:
        """Every string that may hold a {{ template }}: bodies, rich fields and (unless
        replies=False) quick replies."""
        out = list(self.bodies())
        if self.rich is not None:
            out += _strings(self.rich.model_dump(exclude_none=True))
        for q in self.replies() if replies else []:
            out += [q.label] + ([q.message_text] if q.message_text else [])
        return out


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() if k not in ("action", "type") for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


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
    group: str | None = Field(None, min_length=1, max_length=50)  # the builder's folder for this block
    trigger: Trigger | None = None  # None: only reachable through call/branch
    exclusive: bool = False  # starting it ends every running scenario first (e.g. a goodbye)
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


class Escape(Model):
    """Kakao's 탈출 block: these exact phrases end every running scenario."""

    phrases: list[str] = Field(min_length=1)  # compared like other exact phrases (spacing and edge punctuation ignored)
    scenario: str | None = None  # run afterwards; None: say the "cancelled" response

    @field_validator("phrases")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if any(not p.strip() for p in v):
            raise ValueError("escape phrases must not be empty")
        return v


class SessionTimeout(Model):
    """A session idle longer than this starts over at its next message (Kakao: 최대 대기 시간)."""

    minutes: int = Field(ge=1, le=1440)
    notice: str | None = None  # response ID said when a conversation in progress is dropped; default "restarted"


class System(Model):
    """System blocks (all optional; a document without them behaves as before)."""

    welcome: str | None = None  # scenario run when a session is opened (POST .../sessions/{id}/open)
    escape: Escape | None = None
    timeout: SessionTimeout | None = None


class BotDocument(Model):
    """What a builder edits and stores: a bot without its computed version."""

    id: ID
    label: str = ""
    settings: Settings = Settings()
    system: System = System()
    intents: dict[ID, Intent] = Field(default_factory=dict)
    entities: dict[ID, Entity] = Field(default_factory=dict)
    scenarios: dict[ID, Scenario] = Field(min_length=1)
    responses: dict[ID, Response] = Field(min_length=1)


class BotDefinition(BotDocument):
    version: str  # content hash of the document (bot/loader.py document_version)

    def block_targets(self) -> set[str]:
        """Scenarios a client may start directly, because a response offers them as a block button."""
        return {t for r in self.responses.values() for t in r.block_targets()}
