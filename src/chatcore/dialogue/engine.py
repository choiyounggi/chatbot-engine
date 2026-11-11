"""Scenario engine: applies one turn's commands and advances scenarios
deterministically until the bot needs user input (docs/engine-core-plan.md §3.2).

Every state change is emitted as an event and applied to the given state at
once, so `fold(previous_events + outcome.events)` equals the state afterwards.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chatcore.bot.schema import (
    ActionStep,
    BotDefinition,
    BranchStep,
    CallStep,
    CollectStep,
    QuickReply,
    RespondStep,
    Slot,
)
from chatcore.dialogue.commands import (
    AnswerKnowledge,
    CancelScenario,
    Clarify,
    Command,
    Escape,
    Fallback,
    RerunScenario,
    Restart,
    SetSlot,
    SlotValue,
    StartScenario,
    to_dict,
)
from chatcore.dialogue.events import (
    ActionExecuted,
    ActionFailed,
    AttemptFailed,
    BotUttered,
    CommandRejected,
    Event,
    PromptIssued,
    ScenarioCancelled,
    ScenarioCompleted,
    ScenarioStarted,
    SessionRestarted,
    SlotRejected,
    SlotSet,
    StepEntered,
)
from chatcore.dialogue.state import DialogueState, Frame, apply
from chatcore.dialogue.values import (
    ActionError,
    ActionRegistry,
    BotMessage,
    MissingVariable,
    RenderError,
    SlotValueError,
    as_text,
    label_replies,
    lookup,
    normalize_slot,
    render,
    render_response,
)


@dataclass
class TurnOutcome:
    events: list[Event] = field(default_factory=list)
    messages: list[BotMessage] = field(default_factory=list)


class ScenarioEngine:
    def __init__(self, bot: BotDefinition, actions: ActionRegistry, rng: random.Random | None = None) -> None:
        self.bot = bot
        self.actions = actions
        self.rng = rng or random.Random()

    def run(self, state: DialogueState, commands: Sequence[Command]) -> TurnOutcome:
        """Apply `commands`, then advance. Mutates `state` to match the returned events."""
        return _Turn(self, state).run(commands)

    def render_system(self, response_id: str, context: dict[str, Any] | None = None) -> BotMessage:
        """Render a bot-level response outside a turn (e.g. input_invalid, error)."""
        return render_response(response_id, self.bot.responses[response_id], context or {}, self.rng)


class _Turn:
    def __init__(self, engine: ScenarioEngine, state: DialogueState) -> None:
        self.bot, self.actions, self.rng = engine.bot, engine.actions, engine.rng
        self.settings = engine.bot.settings
        self.state = state
        self.out = TurnOutcome()
        self.progressed = False  # set by run(): did this turn answer the pending prompt?

    # ------------------------------------------------------------ plumbing

    def emit(self, event: Event) -> None:
        self.out.events.append(event)
        apply(self.state, event)

    def context(self) -> dict[str, Any]:
        """Slots and action results of every running scenario; inner frames win."""
        slots: dict[str, Any] = {}
        results: dict[str, Any] = {}
        for f in self.state.stack:
            slots.update(f.slots)
            results.update(f.results)
        return {"slots": slots, "results": results}

    def say(self, response_id: str, quick_replies: Sequence[str] | None = None) -> None:
        response = self.bot.responses[response_id]
        try:
            msg = render_response(response_id, response, self.context(), self.rng)
        except (MissingVariable, RenderError):
            if response_id == "fallback":
                raise
            self.say("fallback")
            return
        if quick_replies is not None:
            msg = BotMessage(msg.response, msg.text, label_replies(quick_replies), msg.rich, msg.kind)
        self.out.messages.append(msg)
        self.emit(BotUttered(response=msg.response, text=msg.text, quick_replies=list(msg.quick_replies)))

    def reject(self, cmd: Command, reason: str) -> None:
        self.emit(CommandRejected(command=to_dict(cmd), reason=reason))

    def cancel_chain(self, reason: str) -> None:
        """Cancel the top scenario and every parent that called it (a called
        child is part of its parent's task). A scenario the user started on
        top of another (a digression) is cancelled alone; the one below resumes."""
        while self.state.top is not None:
            frame = self.state.top
            self.emit(ScenarioCancelled(scenario=frame.scenario, reason=reason))
            if not frame.called:
                return

    # ------------------------------------------------------------ commands

    def run(self, commands: Sequence[Command]) -> TurnOutcome:
        waiting = self.state.top if self.state.top is not None and self.state.top.awaiting else None
        waiting_slot = waiting.awaiting if waiting is not None else None
        fallback = False
        hold = False

        for cmd in commands:
            match cmd:
                case StartScenario(scenario=sid, slots=slots):
                    if sid not in self.bot.scenarios:
                        self.reject(cmd, f"unknown scenario {sid!r}")
                        fallback = True
                        continue
                    if self.state.frame(sid) is None:
                        if self.bot.scenarios[sid].exclusive:
                            for f in reversed(list(self.state.stack)):
                                self.emit(ScenarioCancelled(scenario=f.scenario, reason="superseded"))
                        if len(self.state.stack) >= self.settings.max_stack_depth:
                            self.reject(cmd, "scenario stack is full")
                            fallback = True
                            continue
                        self.emit(ScenarioStarted(scenario=sid))
                        self.progressed = True
                    for name, value in slots.items():
                        self.set_slot(cmd, sid, name, value, waiting, waiting_slot)
                case SetSlot(name=name, value=value):
                    target = next((f for f in reversed(self.state.stack)
                                   if name in self.bot.scenarios[f.scenario].slots), None)
                    if target is None:
                        self.reject(cmd, f"no running scenario has slot {name!r}")
                        fallback = True
                        continue
                    self.set_slot(cmd, target.scenario, name, value, waiting, waiting_slot)
                case CancelScenario():
                    if self.state.top is None:
                        self.reject(cmd, "nothing to cancel")
                        fallback = True
                        continue
                    self.cancel_chain("user")
                    self.say("cancelled")
                    self.progressed = True
                case Restart():
                    self.emit(SessionRestarted())
                    self.say("restarted")
                    self.progressed = True
                case Escape():
                    for f in reversed(list(self.state.stack)):
                        self.emit(ScenarioCancelled(scenario=f.scenario, reason="escape"))
                    escape = self.bot.system.escape
                    if escape is not None and escape.scenario is not None:
                        self.emit(ScenarioStarted(scenario=escape.scenario))
                    else:
                        self.say("cancelled")
                    self.progressed = True
                case Clarify(intents=intents):
                    known = [i for i in dict.fromkeys(intents) if i in self.bot.intents]
                    if len(known) < 2:
                        self.reject(cmd, "clarify needs at least two known intents")
                        fallback = True
                        continue
                    self.say("clarify", [self.bot.intents[i].label or i for i in known])
                    self.progressed = True
                    hold = True  # the choice answers this turn; ask nothing else
                case RerunScenario(scenario=sid, slot=slot, value=value):
                    done = next((c for c in reversed(self.state.completed) if c.scenario == sid), None)
                    if done is None or sid not in self.bot.scenarios or self.state.frame(sid) is not None:
                        self.reject(cmd, f"{sid!r} has not completed recently or is still running")
                        fallback = True
                        continue
                    slot_def = self.bot.scenarios[sid].slots.get(slot)
                    if slot_def is None:
                        self.reject(cmd, f"scenario {sid!r} has no slot {slot!r}")
                        fallback = True
                        continue
                    try:
                        new_value = normalize_slot(self.bot, slot_def.entity, value)
                    except SlotValueError as exc:
                        self.reject(cmd, str(exc))
                        fallback = True
                        continue
                    if not self.accepts(sid, slot, slot_def, new_value):
                        continue  # the validation's message is the reply
                    if len(self.state.stack) >= self.settings.max_stack_depth:
                        self.reject(cmd, "scenario stack is full")
                        fallback = True
                        continue
                    self.emit(ScenarioStarted(scenario=sid))
                    for name, old in done.slots.items():
                        if name != slot:
                            self.emit(SlotSet(scenario=sid, name=name, value=old))
                    self.emit(SlotSet(scenario=sid, name=slot, value=new_value))
                    self.progressed = True
                case Fallback():
                    fallback = True
                case AnswerKnowledge(text=text, replies=replies):
                    replies_out = tuple(QuickReply.model_construct(label=label, action="message", message_text=message,
                                                                   block_id=None) for label, message in replies)
                    msg = BotMessage("knowledge", text, replies_out)
                    self.out.messages.append(msg)
                    self.emit(BotUttered(response=msg.response, text=msg.text, quick_replies=list(msg.quick_replies)))
                    self.progressed = True

        if (waiting is not None and waiting_slot is not None and not self.progressed
                and self.state.top is waiting and waiting.awaiting == waiting_slot):
            self.emit(AttemptFailed(scenario=waiting.scenario, slot=waiting_slot))
            if waiting.attempts >= self.settings.max_attempts:
                self.cancel_chain("too_many_attempts")
                self.say("fallback")
            fallback = False  # asking again is the reply
        if fallback:
            self.say("fallback")
        if not hold:
            self.advance()
        return self.out

    def set_slot(self, cmd: Command, sid: str, name: str, value: SlotValue,
                 waiting: Frame | None, waiting_slot: str | None) -> None:
        slot = self.bot.scenarios[sid].slots.get(name)
        if slot is None:
            self.reject(cmd, f"scenario {sid!r} has no slot {name!r}")
            return
        try:
            normalized = normalize_slot(self.bot, slot.entity, value)
        except SlotValueError as exc:
            self.emit(SlotRejected(scenario=sid, name=name, value=value, reason=str(exc)))
            return
        if not self.accepts(sid, name, slot, normalized):
            return
        self.emit(SlotSet(scenario=sid, name=name, value=normalized))
        if waiting is not None and waiting.scenario == sid and name == waiting_slot:
            self.progressed = True

    def accepts(self, sid: str, name: str, slot: Slot, value: SlotValue) -> bool:
        """Run the slot's validation action, if any. A rejected value is answered with the
        validation's `fail` response; a failing validator with the scenario's error response
        (the value is not the user's fault). Either way the slot stays empty, so it is asked again."""
        check = slot.validation
        if check is None:
            return True
        try:
            valid = self.actions.run(check.action, {"value": as_text(value), "slot": name}).get("valid")
            if not isinstance(valid, bool):
                raise ActionError(f"{check.action} must return {{\"valid\": true or false}}, got {valid!r}")
        except ActionError as exc:
            self.emit(ActionFailed(scenario=sid, action=check.action, error=f"{type(exc).__name__}: {exc}"))
            self.emit(SlotRejected(scenario=sid, name=name, value=value, reason=f"{check.action} failed"))
            self.say(self.bot.scenarios[sid].on_error or "error")
            return False
        if not valid:
            self.emit(SlotRejected(scenario=sid, name=name, value=value, reason=f"rejected by {check.action}"))
            self.say(check.fail)
        return valid

    # ------------------------------------------------------------ steps

    def advance(self) -> None:
        steps = 0
        while self.state.stack:
            if steps >= self.settings.max_steps_per_turn:
                for f in reversed(list(self.state.stack)):
                    self.emit(ScenarioCancelled(scenario=f.scenario, reason="step_limit"))
                self.say("fallback")
                return
            steps += 1
            frame = self.state.stack[-1]
            sid = frame.scenario
            scenario = self.bot.scenarios[sid]
            if frame.step >= len(scenario.steps):
                self.emit(ScenarioCompleted(scenario=sid))
                continue
            step = scenario.steps[frame.step]
            match step:
                case CollectStep(collect=slot):
                    slot_def = scenario.slots[slot]
                    if slot in frame.slots or not slot_def.required:
                        if slot not in frame.slots and slot_def.default is not None:  # optional: never asked
                            default = normalize_slot(self.bot, slot_def.entity, slot_def.default)
                            self.emit(SlotSet(scenario=sid, name=slot, value=default))
                        self.emit(StepEntered(scenario=sid, index=frame.step + 1))
                        continue
                    assert slot_def.ask is not None  # required slots have one (schema)
                    self.emit(PromptIssued(scenario=sid, slot=slot))
                    self.say(slot_def.ask)
                    return  # wait for the user
                case RespondStep(respond=response_id):
                    self.say(response_id)
                    self.emit(StepEntered(scenario=sid, index=frame.step + 1))
                case ActionStep(action=action, args=args):
                    try:
                        rendered = {k: render(v, self.context()) for k, v in args.items()}
                        result = self.actions.run(action, rendered)
                    except (ActionError, MissingVariable) as exc:
                        self.emit(ActionFailed(scenario=sid, action=action, error=f"{type(exc).__name__}: {exc}"))
                        self.say(scenario.on_error or "error")  # while the scenario's slots are still in context
                        self.cancel_chain("action_failed")
                        continue
                    self.emit(ActionExecuted(scenario=sid, action=action, result=result))
                    self.emit(StepEntered(scenario=sid, index=frame.step + 1))
                case CallStep(call=child):
                    self.call(frame, child)
                case BranchStep(branch=branch):
                    try:
                        value: str | None = as_text(lookup(self.context(), branch.on))
                    except MissingVariable:
                        value = None
                    target = (branch.cases.get(value) if value is not None else None) or branch.default
                    if target is None:
                        self.emit(StepEntered(scenario=sid, index=frame.step + 1))
                    else:
                        self.call(frame, target)

    def call(self, frame: Frame, child: str) -> None:
        self.emit(StepEntered(scenario=frame.scenario, index=frame.step + 1))
        if len(self.state.stack) >= self.settings.max_stack_depth:
            self.cancel_chain("stack_limit")
            self.say("error")
            return
        self.emit(ScenarioStarted(scenario=child, called=True))
