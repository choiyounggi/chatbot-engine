"""Turn processor: one user message in, bot messages out (docs/engine-core-plan.md §3.2)."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from chatcore.bot.loader import BotError, validate
from chatcore.bot.schema import BotDefinition
from chatcore.dialogue.commands import Command, Fallback, to_dict
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
from chatcore.dialogue.values import ActionRegistry, BotMessage
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


class TurnProcessor:
    def __init__(self, bot: BotDefinition, store: TrackerStore, interpreter: Interpreter,
                 actions: ActionRegistry, rng: random.Random | None = None) -> None:
        report = validate(bot, actions.names())
        if report.errors:
            raise BotError(report.errors)
        self.bot, self.store, self.interpreter = bot, store, interpreter
        self.engine = ScenarioEngine(bot, actions, rng)

    def handle(self, session_id: str, text: str) -> TurnResult:
        """Process one user message. Raises SessionBusy if the session is
        mid-turn; store errors propagate (nothing is stored then)."""
        clean = " ".join(text.split()) if isinstance(text, str) else ""
        if not clean or len(clean) > MAX_INPUT_CHARS:
            return TurnResult([self.engine.render_system("input_invalid")], [])

        with self.store.session(session_id) as tracker:
            state = fold(tracker.events)
            new: list[Event] = []

            def emit(event: Event) -> None:
                new.append(event)
                apply(state, event)

            notices: list[BotMessage] = []
            if not state.started:
                emit(SessionStarted(bot_id=self.bot.id, bot_version=self.bot.version))
            elif state.bot_version != self.bot.version:
                # The session's version is no longer served. A scenario in progress may have
                # changed in any way (steps, slots, order), so start over and say so; an idle
                # session just continues on this version.
                if state.stack:
                    log.info("session %s: bot version changed mid-scenario; restarting", session_id)
                    emit(SessionRestarted())
                    notice = self.engine.render_system("restarted")
                    emit(BotUttered(response=notice.response, text=notice.text,
                                    quick_replies=list(notice.quick_replies)))
                    notices.append(notice)
                emit(SessionStarted(bot_id=self.bot.id, bot_version=self.bot.version))
            emit(UserUttered(text=clean))
            prefix = list(new)
            try:
                interpretation = self.interpreter.interpret(clean, state, self.bot)
                commands = list(interpretation.commands)
                emit(CommandsReceived(commands=[to_dict(c) for c in commands], nlu=interpretation.info))
                outcome = self.engine.run(state, commands)
                new += outcome.events
                messages = notices + outcome.messages
            except Exception as exc:  # a bug must not wedge the session: log, answer, keep only safe events
                log.exception("turn failed for session %s", session_id)
                msg = self.engine.render_system("error")
                new = prefix + [TurnFailed(reason=type(exc).__name__),
                                BotUttered(response=msg.response, text=msg.text, quick_replies=list(msg.quick_replies))]
                messages = notices + [msg]
            tracker.append(new)
        return TurnResult(messages, new)
