"""Load a bot directory (bot.yaml + nlu.yaml) and check it before it runs.

`validate` returns every problem at once, so a builder can show them all.
Errors block loading; warnings are advice (e.g. an intent with few examples).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from chatcore.bot.josa import PARTICLES
from chatcore.bot.schema import (
    PLACEHOLDER,
    REQUIRED_RESPONSES,
    SYSTEM_ENTITIES,
    ActionStep,
    BotDefinition,
    BranchStep,
    CallStep,
    CollectStep,
    DictionaryEntity,
    RespondStep,
)
from chatcore.dialogue.values import SlotValueError, normalize_slot

MIN_EXAMPLES = 5
# Kakao channel limits: warnings only, the engine itself serves any channel
KAKAO_BUBBLES, KAKAO_QUICK_REPLIES, KAKAO_LABEL = 3, 10, 14
KAKAO_CARD_BUTTONS, KAKAO_LIST_ITEMS, KAKAO_CAROUSEL = 3, 5, 10
LOOSE_PLACEHOLDER = re.compile(r"\{\{.*?\}\}")


class _BotYamlLoader(yaml.SafeLoader):
    """SafeLoader with two fixes for hand-written bot files:
    - mapping keys stay the exact text written (`on:`, `yes:`, `1:` are keys,
      not booleans/ints that silently collide), and
    - only true/false are booleans (YAML 1.2), not yes/no/on/off, and
    - dates and clock times stay text (`2026-01-01` and `12:30` in examples are sentences, not
      a date and YAML 1.1's base-60 integer 750)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[object, object]:
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            scalar = isinstance(key_node, yaml.ScalarNode)
            key = key_node.value if scalar else self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


_REPLACED = {"tag:yaml.org,2002:" + t for t in ("bool", "timestamp", "int", "float")}
_BotYamlLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag not in _REPLACED]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
# SafeLoader's int/float patterns without their base-60 (`12:30`) branches
_BotYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+)$"),
    list("-+0123456789"),
)
_BotYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9_]+(?:[eE][-+][0-9]+)?"
               r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"),
    list("-+0123456789."),
)
_BotYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
_BotYamlLoader.add_constructor("tag:yaml.org,2002:map", _BotYamlLoader.construct_mapping)


class BotError(ValueError):
    """The bot definition cannot run. `problems` lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("invalid bot definition:\n- " + "\n- ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_BotYamlLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise BotError([f"{path.name}: not valid YAML: {exc}"]) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise BotError([f"{path.name}: top level must be a mapping"])
    return data


def read_document(directory: Path) -> dict[str, Any]:
    """The bot document: `directory/bot.yaml` merged with the optional `nlu.yaml`."""
    bot_file, nlu_file = directory / "bot.yaml", directory / "nlu.yaml"
    if not bot_file.is_file():
        raise BotError([f"{bot_file} not found"])
    raw = _read_yaml(bot_file)
    nlu = _read_yaml(nlu_file) if nlu_file.is_file() else {}
    overlap = sorted(set(raw) & set(nlu))
    if overlap:
        raise BotError([f"keys defined in both bot.yaml and nlu.yaml: {overlap}"])
    return {**raw, **nlu}


def document_version(document: dict[str, Any]) -> str:
    """Content hash of a bot document (key order and formatting do not matter; 1.0 and 1 are the
    same number, as a JSON client may send either)."""

    def plain(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, list):
            return [plain(v) for v in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    try:
        canonical = json.dumps(plain(document), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                               allow_nan=False)
    except (TypeError, ValueError) as exc:  # non-JSON values, mixed key types, NaN
        raise BotError([f"the bot document is not plain JSON data: {exc}"]) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def parse_document(document: dict[str, Any]) -> BotDefinition:
    """BotDefinition from a document, without semantic validation (BotError on schema errors)."""
    if not isinstance(document, dict):
        raise BotError(["a bot document must be a mapping"])
    if "version" in document:
        raise BotError(["'version' is computed from the content and cannot be set"])
    try:
        return BotDefinition.model_validate({**document, "version": document_version(document)})
    except ValidationError as exc:
        raise BotError([f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]) from exc


def bot_from_document(document: dict[str, Any], actions: Iterable[str] | None = None) -> BotDefinition:
    """Parse and validate a bot document. If `actions` is given, every action step must name one of them."""
    bot = parse_document(document)
    report = validate(bot, actions)
    if report.errors:
        raise BotError(report.errors)
    return bot


def load_bot(directory: Path, actions: Iterable[str] | None = None) -> BotDefinition:
    """Load and validate `directory/bot.yaml` (+ `nlu.yaml`)."""
    return bot_from_document(read_document(directory), actions)


def validate(bot: BotDefinition, actions: Iterable[str] | None = None) -> Report:
    errors: list[str] = []
    warnings: list[str] = []
    known_actions = set(actions) if actions is not None else None

    missing = sorted(REQUIRED_RESPONSES - set(bot.responses))
    if missing:
        errors.append(f"missing required responses: {missing}")
    errors += _template_errors(bot)
    system_errors, system_warnings = _system_problems(bot)
    errors += system_errors
    warnings += system_warnings
    response_errors, response_warnings = _response_problems(bot)
    errors += response_errors
    warnings += response_warnings

    triggered: dict[str, str] = {}
    for sid, sc in bot.scenarios.items():
        where = f"scenario {sid}"
        if sc.trigger is not None:
            if sc.trigger.intent == "out_of_scope":
                errors.append(f"{where}: out_of_scope cannot trigger a scenario (it is the fallback)")
            elif sc.trigger.intent not in bot.intents:
                errors.append(f"{where}: trigger intent {sc.trigger.intent!r} is not defined")
            elif sc.trigger.intent in triggered:
                errors.append(f"{where}: intent {sc.trigger.intent!r} already triggers {triggered[sc.trigger.intent]}")
            else:
                triggered[sc.trigger.intent] = sid
        if sc.on_error is not None and sc.on_error not in bot.responses:
            errors.append(f"{where}: on_error response {sc.on_error!r} is not defined")
        for name, slot in sc.slots.items():
            known_entity = slot.entity in bot.entities or slot.entity in SYSTEM_ENTITIES
            if not known_entity:
                errors.append(f"{where}: slot {name!r} uses unknown entity {slot.entity!r}")
            if slot.ask is not None and slot.ask not in bot.responses:
                errors.append(f"{where}: slot {name!r} asks with unknown response {slot.ask!r}")
            if slot.default is not None and known_entity:
                try:
                    normalize_slot(bot, slot.entity, slot.default)
                except SlotValueError as exc:
                    errors.append(f"{where}: slot {name!r} default {slot.default!r} is not a {slot.entity}: {exc}")
            if slot.validation is not None:
                check = slot.validation
                if known_actions is not None and check.action not in known_actions:
                    errors.append(f"{where}: slot {name!r} is validated by unknown action {check.action!r}")
                if check.fail not in bot.responses:
                    errors.append(f"{where}: slot {name!r} fails validation with unknown response {check.fail!r}")
        for name, slot in sc.slots.items():
            if slot.ask in bot.responses and f"slots.{name}" in _paths_of(bot.responses[slot.ask].texts(replies=False)):
                errors.append(f"{where}: response {slot.ask!r} asks for slot {name!r} but also shows it")
        bubbles = sum(isinstance(step, RespondStep) for step in sc.steps)
        if bubbles > KAKAO_BUBBLES:
            warnings.append(f"{where}: {bubbles} responses in one block; a Kakao channel shows at most {KAKAO_BUBBLES}")
        for i, step in enumerate(sc.steps):
            at = f"{where} step {i}"
            if isinstance(step, CollectStep) and step.collect not in sc.slots:
                errors.append(f"{at}: collects undeclared slot {step.collect!r}")
            elif isinstance(step, RespondStep) and step.respond not in bot.responses:
                errors.append(f"{at}: unknown response {step.respond!r}")
            elif isinstance(step, ActionStep) and known_actions is not None and step.action not in known_actions:
                errors.append(f"{at}: unknown action {step.action!r}")
            elif isinstance(step, CallStep) and step.call not in bot.scenarios:
                errors.append(f"{at}: calls unknown scenario {step.call!r}")
            elif isinstance(step, BranchStep):
                targets = list(step.branch.cases.values()) + ([step.branch.default] if step.branch.default else [])
                errors += [f"{at}: branches to unknown scenario {t!r}" for t in targets if t not in bot.scenarios]

    cycle = _find_call_cycle(bot)
    if cycle:
        errors.append(f"scenarios call each other in a cycle: {' -> '.join(cycle)}")
    else:
        depth, path = _deepest_call_chain(bot)
        if depth > bot.settings.max_stack_depth:
            limit = bot.settings.max_stack_depth
            errors.append(f"call chain {' -> '.join(path)} is {depth} deep; max_stack_depth is {limit}")

    owner: dict[str, str] = {}
    for eid, ent in bot.entities.items():
        if isinstance(ent, DictionaryEntity):
            canon_of: dict[str, str] = {}
            for canon, synonyms in ent.values.items():
                for form in [canon, *synonyms]:
                    if form in owner and owner[form] != eid:
                        errors.append(f"entity value {form!r} belongs to both {owner[form]} and {eid}")
                    elif form in canon_of and canon_of[form] != canon:
                        errors.append(f"entity {eid}: {form!r} is a synonym of both {canon_of[form]!r} and {canon!r}")
                    owner[form], canon_of[form] = eid, canon

    seen: dict[str, str] = {}
    for iid, intent in bot.intents.items():
        for ex in intent.examples:
            if ex in seen and seen[ex] != iid:
                errors.append(f"example {ex!r} is in both intents {seen[ex]} and {iid}")
            seen[ex] = iid
        if len(intent.examples) < MIN_EXAMPLES:
            warnings.append(f"intent {iid} has only {len(intent.examples)} examples (recommend >= {MIN_EXAMPLES})")
    return Report(errors, warnings)


def _system_problems(bot: BotDefinition) -> tuple[list[str], list[str]]:
    """welcome / escape / timeout refer to things that exist; escape phrases do not shadow examples."""
    errors: list[str] = []
    warnings: list[str] = []
    system = bot.system
    if system.welcome is not None and system.welcome not in bot.scenarios:
        errors.append(f"system.welcome: unknown scenario {system.welcome!r}")
    if system.escape is not None:
        if system.escape.scenario is not None and system.escape.scenario not in bot.scenarios:
            errors.append(f"system.escape: unknown scenario {system.escape.scenario!r}")
        phrases = {_squash(p) for p in system.escape.phrases}
        empty = [p for p in system.escape.phrases if not _squash(p)]
        if empty:
            errors.append(f"system.escape: phrases with only punctuation never match: {empty}")
        shadowed = sorted({ex for intent in bot.intents.values() for ex in intent.examples if _squash(ex) in phrases})
        if shadowed:
            warnings.append(f"system.escape: these examples are escape phrases, so they always escape: {shadowed}")
    notice = system.timeout.notice if system.timeout is not None else None
    if notice is not None:
        if notice not in bot.responses:
            errors.append(f"system.timeout: unknown notice response {notice!r}")
        elif _paths_of(bot.responses[notice].texts()):
            errors.append(f"system.timeout: notice {notice!r} is said after the conversation is gone; "
                          "remove its variables")
    return errors, warnings


def _squash(text: str) -> str:
    """Roughly what the interpreter compares exact phrases by (bare()), without loading Kiwi."""
    return re.sub(r"\s+", " ", text).strip(" .,!?~…·'\"()[]")


def _response_problems(bot: BotDefinition) -> tuple[list[str], list[str]]:
    """Block buttons point at scenarios; Kakao's size limits as warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    for rid, r in bot.responses.items():
        where = f"response {rid!r}"
        errors += [f"{where}: block button to unknown scenario {t!r}" for t in sorted(r.block_targets())
                   if t not in bot.scenarios]
        replies = r.replies()
        if len(replies) > KAKAO_QUICK_REPLIES:
            warnings.append(f"{where}: {len(replies)} quick replies; Kakao shows at most {KAKAO_QUICK_REPLIES}")
        long = [q.label for q in replies if len(q.label) > KAKAO_LABEL] + [
            b.label for b in r.buttons() if len(b.label) > KAKAO_LABEL]
        if long:
            warnings.append(f"{where}: labels longer than {KAKAO_LABEL} characters are cut on Kakao: {long}")
        cards = ([r.basic_card] if r.basic_card else []) + (r.carousel.items if r.carousel else [])
        if any(len(c.buttons) > KAKAO_CARD_BUTTONS for c in cards) or (
                r.list_card is not None and len(r.list_card.buttons) > KAKAO_CARD_BUTTONS):
            warnings.append(f"{where}: a card has more than {KAKAO_CARD_BUTTONS} buttons (Kakao's limit)")
        if r.list_card is not None and len(r.list_card.items) > KAKAO_LIST_ITEMS:
            warnings.append(f"{where}: {len(r.list_card.items)} list items; Kakao shows at most {KAKAO_LIST_ITEMS}")
        if r.carousel is not None and len(r.carousel.items) > KAKAO_CAROUSEL:
            warnings.append(f"{where}: {len(r.carousel.items)} carousel cards; Kakao shows at most {KAKAO_CAROUSEL}")
    return errors, warnings


def _paths_of(texts: Iterable[str]) -> set[str]:
    return {m.group(1) for t in texts for m in PLACEHOLDER.finditer(t)}


def _template_errors(bot: BotDefinition) -> list[str]:
    """Every {{ path }} must be slots.<declared slot> or results.<used action>.<field>;
    engine-level responses take no variables (they are rendered outside scenarios)."""
    slots = {name for sc in bot.scenarios.values() for name in sc.slots}
    actions = {s.action for sc in bot.scenarios.values() for s in sc.steps if isinstance(s, ActionStep)}

    def bad(path: str) -> str | None:
        head, _, rest = path.partition(".")
        if head == "slots" and rest in slots:
            return None
        if head == "results" and rest.partition(".")[0] in actions and "." in rest:
            return None
        return path

    errors = []
    for where, texts in [(f"response {rid!r}", r.texts()) for rid, r in bot.responses.items()] + [
        (f"scenario {sid} step {i}", list(st.args.values()))
        for sid, sc in bot.scenarios.items() for i, st in enumerate(sc.steps) if isinstance(st, ActionStep)
    ]:
        for t in texts:
            # every "{{ ... }}" must be a valid placeholder, otherwise it would reach the user verbatim
            if len(LOOSE_PLACEHOLDER.findall(t)) != len(PLACEHOLDER.findall(t)) or t.count("{{") != t.count("}}"):
                hint = "write {{ slots.x }} or {{ slots.x|은 }}"
                errors.append(f"{where}: malformed placeholder in {t!r} ({hint})")
        filters = {m.group(2) for t in texts for m in PLACEHOLDER.finditer(t) if m.group(2)}
        errors += [f"{where}: unknown particle filter |{f}" for f in sorted(filters - PARTICLES)]
    for rid, resp in bot.responses.items():
        texts = resp.texts()
        paths = _paths_of(texts)
        if rid in REQUIRED_RESPONSES and paths:
            errors.append(f"response {rid!r} is rendered by the engine itself; remove variables {sorted(paths)}")
            continue
        errors += [f"response {rid!r}: unknown variable {{{{ {p} }}}}" for p in sorted(paths) if bad(p)]
    for sid, sc in bot.scenarios.items():
        for i, step in enumerate(sc.steps):
            if isinstance(step, ActionStep):
                errors += [f"scenario {sid} step {i}: unknown variable {{{{ {p} }}}}"
                           for p in sorted(_paths_of(step.args.values())) if bad(p)]
            elif isinstance(step, BranchStep) and bad(step.branch.on):
                errors.append(f"scenario {sid} step {i}: branch on unknown path {step.branch.on!r}")
    return errors


def _call_edges(bot: BotDefinition) -> dict[str, list[str]]:
    edges: dict[str, list[str]] = {}
    for sid, sc in bot.scenarios.items():
        out: list[str] = []
        for step in sc.steps:
            if isinstance(step, CallStep):
                out.append(step.call)
            elif isinstance(step, BranchStep):
                out += list(step.branch.cases.values()) + ([step.branch.default] if step.branch.default else [])
        edges[sid] = [t for t in out if t in bot.scenarios]
    return edges


def _deepest_call_chain(bot: BotDefinition) -> tuple[int, list[str]]:
    """Longest call/branch chain (in frames) starting from any scenario; assumes no cycles."""
    edges = _call_edges(bot)
    memo: dict[str, list[str]] = {}

    def longest(node: str) -> list[str]:
        if node not in memo:
            memo[node] = [node] + max((longest(n) for n in edges[node]), key=len, default=[])
        return memo[node]

    best = max((longest(s) for s in edges), key=len, default=[])
    return len(best), best


def _find_call_cycle(bot: BotDefinition) -> list[str] | None:
    """A scenario path that calls back into itself, or None."""
    edges = _call_edges(bot)

    state: dict[str, int] = {}  # 1 = on the current path, 2 = done

    def visit(node: str, path: list[str]) -> list[str] | None:
        state[node] = 1
        for nxt in edges.get(node, []):
            if state.get(nxt) == 1:
                return path[path.index(nxt):] + [nxt]
            if state.get(nxt) is None:
                found = visit(nxt, path + [nxt])
                if found:
                    return found
        state[node] = 2
        return None

    for start in sorted(edges):
        if start not in state:
            found = visit(start, [start])
            if found:
                return found
    return None
