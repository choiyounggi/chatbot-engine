"""Context-first interpretation: NLU result + dialogue state -> commands
(docs/engine-core-plan.md §3.2, rules 1-6, first match wins).

1. exact system phrase: the bot's escape phrases      -> Escape
   ("취소", "처음으로")                                 -> CancelScenario / Restart
2. a slot is pending and the input carries its entity  -> SetSlot (unless a confident,
   different intent wins by switch_threshold); a pending sys.text slot takes the whole input
   (an input that is only an entity plus particles — "부산은?", "제주도는 어때?" — goes
   to rule 4 before rule 3, and may only start a scenario that takes that entity:
   classifiers give such short inputs confident but meaningless intents)
3. a confident, unambiguous intent with a scenario     -> StartScenario (+ its slots found in the text)
4. no confident intent, but an entity that a scenario
   finished within `rerun_window` turns accepts        -> RerunScenario ("부산은?")
5. the top two intents are too close                   -> Clarify
6. otherwise                                           -> Fallback

Only entity types that matter right now are used (the pending slot, the
predicted scenario, recently finished scenarios), so "세종대왕" does not fill
a location slot unless a location is actually wanted.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from chatcore.bot.schema import BotDefinition, Scenario
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
)
from chatcore.dialogue.state import DialogueState
from chatcore.dialogue.turn import Interpretation
from chatcore.knowledge import Knowledge
from chatcore.nlu.entities import EntityMatch, bare, normalize, pick
from chatcore.nlu.model import OOS, NluResult

log = logging.getLogger(__name__)
TEXT = "sys.text"  # never extracted: it is the whole answer to a pending question
# System entities every bot has always seen; the newer ones count only for bots whose slots use them,
# so an old bot decides exactly as before ("주말은 어때?" must not become an entity-only follow-up).
BASE_SYSTEM = frozenset({"sys.number", "sys.date", "sys.time"})


def relevant(entities: tuple[EntityMatch, ...], bot: BotDefinition) -> tuple[EntityMatch, ...]:
    """The matches of entity types this bot knows: its own entities, the base system entities,
    and the system entities its slots use."""
    used = {slot.entity for sc in bot.scenarios.values() for slot in sc.slots.values()}
    known = set(bot.entities) | BASE_SYSTEM | used
    return tuple(m for m in entities if m.entity in known)

CANCEL_PHRASES = frozenset(
    {"취소", "취소할게", "취소해줘", "그만", "그만할래", "그만할게", "됐어", "안 할래", "안할래", "중단"}
)
RESTART_PHRASES = frozenset({"처음으로", "처음부터", "처음부터 다시", "다시 시작", "리셋"})
# What may surround the entity in an entity-only follow-up ("그럼 부산은요?", "제주도는 어때").
FOLLOWUP_REST = re.compile(
    r"^(?:그럼|그러면|그리고|혹시|그럼혹시)?(?:은|는|이|가|도|의|에|에서|에서는|로|으로)?(?:요)?"
    r"(?:어때|어때요|어떄|어떤가요|어떻게돼|어떻게돼요)?$"
)


def entity_only(text: str, entities: tuple[EntityMatch, ...]) -> list[EntityMatch]:
    """The entities of `text` if, without them, only follow-up filler remains; else []."""
    chosen = pick(list(entities), {e.entity for e in entities})
    if not chosen:
        return []
    norm = normalize(text)
    rest = "".join(ch for i, ch in enumerate(norm) if not any(m.start <= i < m.end for m in chosen))
    rest = re.sub(r"[\s.,!?~…·'\"]", "", rest)
    return chosen if FOLLOWUP_REST.match(rest) else []


class NluInterpreter:
    def __init__(self, parse: Callable[[str], NluResult], knowledge: Knowledge | None = None) -> None:
        """`parse` is NluModel.parse (or a stand-in in tests); `knowledge`: the bot's knowledge base."""
        self.parse = parse
        self.knowledge = knowledge

    def interpret(self, text: str, state: DialogueState, bot: BotDefinition) -> Interpretation:
        result = self.parse(text)
        rule, commands = self._decide(text, result, state, bot)
        info: dict[str, Any] = {"rule": rule, "text": text, **result.to_log()}
        waiting = state.top is not None and state.top.awaiting is not None  # a pending question: ask again
        if self.knowledge is not None and not waiting:
            try:
                found, answer = self.knowledge.consult(text, rule, result.similarity, result.vector)
            except Exception as exc:  # the knowledge base must not break a turn the rules already decided
                log.exception("knowledge-base search failed")
                info["knowledge"] = {"error": type(exc).__name__}
            else:
                if found is not None:
                    info["knowledge"] = found.to_log()
                if answer is not None:
                    info["rule_before"], info["rule"] = rule, "knowledge"
                    commands = [AnswerKnowledge(answer.text, answer.replies, answer.items)]
        return Interpretation(commands, info)

    # ------------------------------------------------------------------

    def _decide(self, text: str, r: NluResult, state: DialogueState, bot: BotDefinition) -> tuple[str, list[Command]]:
        s = bot.settings
        phrase = bare(text)
        escape = bot.system.escape
        if escape is not None and phrase in {bare(p) for p in escape.phrases} - {""}:  # "..." is not an escape
            return "system", [Escape()]
        if phrase in CANCEL_PHRASES:
            return "system", [CancelScenario()]
        if phrase in RESTART_PHRASES:
            return "system", [Restart()]

        scenario_of = {sc.trigger.intent: sid for sid, sc in bot.scenarios.items() if sc.trigger is not None}
        top, second = r.top, r.second
        passes = top.score >= s.intent_threshold and r.similarity >= s.similarity_threshold
        confident = top.intent != OOS and passes
        surely_out = top.intent == OOS and top.score >= s.switch_threshold  # "부산 맛집 추천해줘"
        ambiguous = (r.matched_by not in ("exact", "keyword") and second is not None and second.intent != OOS
                     and top.score - second.score < s.ambiguity_margin)
        target = scenario_of.get(top.intent) if confident else None
        entities = relevant(r.entities, bot)
        bare_entities = entity_only(text, entities)
        if bare_entities and target is not None:
            types = {slot.entity for slot in bot.scenarios[target].slots.values()}
            if not any(m.entity in types for m in bare_entities):
                target = None  # "부산은?" must not start a scenario that has no use for 부산

        # 2. answer to the pending question
        frame = state.top
        if frame is not None and frame.awaiting is not None:
            waiting = bot.scenarios[frame.scenario]
            slot = waiting.slots[frame.awaiting]
            found = pick(list(entities), {slot.entity})
            switching = (target is not None and target != frame.scenario and not ambiguous
                         and top.score >= s.switch_threshold)
            if slot.entity == TEXT and not switching:  # any answer is the value
                return "slot", [SetSlot(frame.awaiting, normalize(text))]
            if found and not switching:
                return "slot", [SetSlot(frame.awaiting, found[0].value)]

        # 4a. entity-only follow-up on a recently finished scenario (before trusting the intent)
        if bare_entities and not surely_out:
            rerun = self._rerun(state, bot, entities)
            if rerun is not None:
                return "rerun", [rerun]

        # 3. a clear intent
        if target is not None and not ambiguous:
            return "intent", [StartScenario(target, self._slots_for(bot.scenarios[target], entities))]

        # 4. follow-up on a recently finished scenario (not when the input is clearly about something else)
        if (not confident or target is None) and not surely_out:
            rerun = self._rerun(state, bot, entities)
            if rerun is not None:
                return "rerun", [rerun]

        # 5. too close to call — only for inputs that pass the same gate as rule 3
        if (ambiguous and confident and second is not None
                and top.intent in scenario_of and second.intent in scenario_of):
            return "clarify", [Clarify((top.intent, second.intent))]

        # 6.
        return "fallback", [Fallback()]

    @staticmethod
    def _rerun(state: DialogueState, bot: BotDefinition, entities: tuple[EntityMatch, ...]) -> RerunScenario | None:
        """Re-run the most recent scenario finished within rerun_window turns
        that has a slot for one of the entities."""
        for done in reversed(state.completed):
            if state.turns - done.turn > bot.settings.rerun_window or done.scenario not in bot.scenarios:
                continue
            for slot_name, slot in bot.scenarios[done.scenario].slots.items():
                found = pick(list(entities), {slot.entity})
                if found:
                    return RerunScenario(done.scenario, slot_name, found[0].value)
        return None

    @staticmethod
    def _slots_for(scenario: Scenario, entities: tuple[EntityMatch, ...]) -> dict[str, SlotValue]:
        """Fill the scenario's slots from entities in the text, one entity per slot, in slot order.
        Limitation: two slots of the same entity type are filled by position only
        ("서울에서 부산으로" cannot tell origin from destination)."""
        chosen = pick(list(entities), {slot.entity for slot in scenario.slots.values()})
        slots: dict[str, SlotValue] = {}
        used: set[int] = set()
        for name, slot in scenario.slots.items():
            for i, m in enumerate(chosen):
                if i not in used and m.entity == slot.entity:
                    slots[name] = m.value
                    used.add(i)
                    break
        return slots
