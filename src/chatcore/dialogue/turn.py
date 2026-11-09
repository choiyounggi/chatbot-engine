"""Turn processor: one user message in, bot messages out (docs/engine-core-plan.md §3.2)."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

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


class Interpreter(Protocol):
    """Understanding: user text (+ state) -> commands. Phase 3 adds the NLU one."""

    def interpret(self, text: str, state: DialogueState, bot: BotDefinition) -> Sequence[Command]: ...


@dataclass
class ScriptedInterpreter:
    """Test double: exact text -> commands; anything else -> Fallback."""

    script: Mapping[str, Sequence[Command]] = field(default_factory=dict)

    def interpret(self, text: str, state: DialogueState, bot: BotDefinition) -> Sequence[Command]:
        return list(self.script.get(text, [Fallback()]))


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

    def _stack_fits(self, state: DialogueState) -> bool:
        for frame in state.stack:
            scenario = self.bot.scenarios.get(frame.scenario)
            if scenario is None or frame.step > len(scenario.steps) or not set(frame.slots) <= set(scenario.slots):
                return False
        return True

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

            if not state.started:
                emit(SessionStarted(bot_id=self.bot.id, bot_version=self.bot.version))
            elif state.bot_version != self.bot.version:
                # The bot was edited mid-conversation. Keep the session if its running
                # scenarios still exist as they were; otherwise start over cleanly.
                if not self._stack_fits(state):
                    log.info("session %s: bot changed under its scenarios; restarting", session_id)
                    emit(SessionRestarted())
                emit(SessionStarted(bot_id=self.bot.id, bot_version=self.bot.version))
            emit(UserUttered(text=clean))
            prefix = list(new)
            try:
                commands = list(self.interpreter.interpret(clean, state, self.bot))
                emit(CommandsReceived(commands=[to_dict(c) for c in commands]))
                outcome = self.engine.run(state, commands)
                new += outcome.events
                messages = outcome.messages
            except Exception as exc:  # a bug must not wedge the session: log, answer, keep only safe events
                log.exception("turn failed for session %s", session_id)
                msg = self.engine.render_system("error")
                new = prefix + [TurnFailed(reason=type(exc).__name__),
                                BotUttered(response=msg.response, text=msg.text, quick_replies=list(msg.quick_replies))]
                messages = [msg]
            tracker.append(new)
        return TurnResult(messages, new)
