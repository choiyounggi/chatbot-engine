"""Turn processor: one user message in, bot messages out (docs/engine-core-plan.md §3.2)."""

from __future__ import annotations

import datetime as dt
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from chatcore.bot.loader import BotError, validate
from chatcore.bot.schema import BotDefinition
from chatcore.dialogue.commands import Command, Fallback, StartScenario, to_dict
from chatcore.dialogue.engine import ScenarioEngine
from chatcore.dialogue.events import (
    BotUttered,
    CommandsReceived,
    Event,
    SessionRestarted,
    SessionStarted,
    TurnFailed,
    UserUttered,
)
from chatcore.dialogue.state import DialogueState, apply, fold
from chatcore.dialogue.values import CURRENT_TURN, ActionRegistry, BotMessage, TurnContext
from chatcore.store.base import TrackerStore

log = logging.getLogger(__name__)
MAX_INPUT_CHARS = 1000


@dataclass(frozen=True)
class Interpretation:
    commands: list[Command]
    info: dict[str, Any] = field(default_factory=dict)  # logged with the turn (rule, intents, entities ...)


class Interpreter(Protocol):
    """Understanding: user text (+ state) -> commands (chatcore.nlu.interpreter.NluInterpreter)."""

    def interpret(self, text: str, state: DialogueState, bot: BotDefinition) -> Interpretation: ...


@dataclass
class ScriptedInterpreter:
    """Test double: exact text -> commands; anything else -> Fallback."""

    script: Mapping[str, Sequence[Command]] = field(default_factory=dict)

    def interpret(self, text: str, state: DialogueState, bot: BotDefinition) -> Interpretation:
        return Interpretation(list(self.script.get(text, [Fallback()])), {"rule": "scripted", "text": text})


@dataclass(frozen=True)
class TurnResult:
    messages: list[BotMessage]
    events: list[Event]  # events stored by this turn (empty if the input was rejected)


class UnknownBlock(ValueError):
    """A direct block start names a scenario that no response offers as a block button."""


Clock = Callable[[], dt.datetime]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class TurnProcessor:
    def __init__(self, bot: BotDefinition, store: TrackerStore, interpreter: Interpreter,
                 actions: ActionRegistry, rng: random.Random | None = None, clock: Clock = utc_now,
                 kb_version: str | None = None) -> None:
        """kb_version: the knowledge base this runtime answers from (recorded in SessionStarted)."""
        # the registry may already hold this bot's own HTTP skills (BotHost.actions_for): they are not clashes
        report = validate(bot, actions.names() - set(bot.skills))
        if report.errors:
            raise BotError(report.errors)
        self.bot, self.store, self.interpreter, self.clock = bot, store, interpreter, clock
        self.kb_version = kb_version
        self.engine = ScenarioEngine(bot, actions, rng)
        self.blocks = bot.block_targets()

    def handle(self, session_id: str, text: str, block: str | None = None) -> TurnResult:
        """Process one user message. `block`: the user pressed a quick reply or button that starts
        this scenario directly (no understanding; UnknownBlock unless a response offers it).
        Raises SessionBusy if the session is mid-turn; store errors propagate (nothing is stored then)."""
        if block is not None and block not in self.blocks:
            raise UnknownBlock(block)
        clean = " ".join(text.split()) if isinstance(text, str) else ""
        if not clean or len(clean) > MAX_INPUT_CHARS:
            return TurnResult([self.engine.render_system("input_invalid")], [])

        with self.store.session(session_id) as tracker:
            state = fold(tracker.events)
            now = self.clock().isoformat()
            new: list[Event] = []

            def emit(event: Event) -> None:
                new.append(event)
                apply(state, event)

            notices = self._expire(state, emit)
            if not state.started:
                emit(self._started(now))
            elif state.bot_version != self.bot.version or state.kb_version != self.kb_version:
                # The session's version is no longer served. A scenario in progress may have
                # changed in any way (steps, slots, order), so start over and say so; an idle
                # session just continues on this version. A new knowledge base alone changes no
                # scenario: the session just answers from it from now on.
                if state.stack and state.bot_version != self.bot.version:
                    log.info("session %s: bot version changed mid-scenario; restarting", session_id)
                    emit(SessionRestarted())
                    notices.append(self._notice("restarted", emit))
                emit(self._started(now))
            emit(UserUttered(text=clean, at=now))

            def understand() -> Interpretation:
                if block is not None:
                    return Interpretation([StartScenario(block)], {"rule": "block", "text": clean, "block": block})
                return self.interpreter.interpret(clean, state, self.bot)

            new, messages = self._run(session_id, state, understand, new, emit, clean)
            tracker.append(new)
        return TurnResult(notices + messages, new)

    def open(self, session_id: str) -> TurnResult:
        """The user opened the conversation (Kakao: entering the chat room). A new session starts
        and runs the welcome scenario, if the bot has one; an existing session gets nothing."""
        with self.store.session(session_id) as tracker:
            state = fold(tracker.events)
            if state.started:
                return TurnResult([], [])
            new: list[Event] = []

            def emit(event: Event) -> None:
                new.append(event)
                apply(state, event)

            emit(self._started(self.clock().isoformat()))
            welcome = self.bot.system.welcome
            messages: list[BotMessage] = []
            if welcome is not None:
                new, messages = self._run(session_id, state, lambda: Interpretation(
                    [StartScenario(welcome)], {"rule": "welcome"}), new, emit, "")
            tracker.append(new)
        return TurnResult(messages, new)

    def _started(self, at: str) -> SessionStarted:
        return SessionStarted(bot_id=self.bot.id, bot_version=self.bot.version, at=at, kb_version=self.kb_version)

    def _expire(self, state: DialogueState, emit: Callable[[Event], None]) -> list[BotMessage]:
        """Start an idle session over (system.timeout). A dropped conversation is announced;
        an idle one only forgets what it could re-run ("부산은?")."""
        timeout = self.bot.system.timeout
        if timeout is None or state.last_active is None or not (state.stack or state.completed):
            return []
        try:
            idle = self.clock() - dt.datetime.fromisoformat(state.last_active)
        except (ValueError, TypeError):  # unreadable or naive time in the log: do not guess
            log.warning("unreadable session time %r", state.last_active)
            return []
        if idle <= dt.timedelta(minutes=timeout.minutes):
            return []
        dropped = bool(state.stack)
        emit(SessionRestarted())
        return [self._notice(timeout.notice or "restarted", emit)] if dropped else []

    def _notice(self, response_id: str, emit: Callable[[Event], None]) -> BotMessage:
        msg = self.engine.render_system(response_id)
        emit(BotUttered(response=msg.response, text=msg.text, quick_replies=list(msg.quick_replies)))
        return msg

    def _run(self, session_id: str, state: DialogueState, understand: Callable[[], Interpretation],
             new: list[Event], emit: Callable[[Event], None], utterance: str) -> tuple[list[Event], list[BotMessage]]:
        """Understand and run the engine; returns (events to store, messages). Actions see the
        utterance and session (CURRENT_TURN) while it runs."""
        prefix = list(new)
        token = CURRENT_TURN.set(TurnContext(utterance, session_id))
        try:
            interpretation = understand()
            commands = list(interpretation.commands)
            emit(CommandsReceived(commands=[to_dict(c) for c in commands], nlu=interpretation.info))
            outcome = self.engine.run(state, commands)
            return new + outcome.events, outcome.messages
        except Exception as exc:  # a bug must not wedge the session: log, answer, keep only safe events
            log.exception("turn failed for session %s", session_id)
            msg = self.engine.render_system("error")
            return prefix + [TurnFailed(reason=type(exc).__name__),
                             BotUttered(response=msg.response, text=msg.text,
                                        quick_replies=list(msg.quick_replies))], [msg]
        finally:
            CURRENT_TURN.reset(token)
