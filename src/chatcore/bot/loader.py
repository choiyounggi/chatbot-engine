"""Load a bot directory (bot.yaml + nlu.yaml) and check it before it runs.

`validate` returns every problem at once, so a builder can show them all.
Errors block loading; warnings are advice (e.g. an intent with few examples).
"""

from __future__ import annotations

import hashlib
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

MIN_EXAMPLES = 5
LOOSE_PLACEHOLDER = re.compile(r"\{\{.*?\}\}")


class _BotYamlLoader(yaml.SafeLoader):
    """SafeLoader with two fixes for hand-written bot files:
    - mapping keys stay the exact text written (`on:`, `yes:`, `1:` are keys,
      not booleans/ints that silently collide), and
    - only true/false are booleans (YAML 1.2), not yes/no/on/off."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[object, object]:
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            scalar = isinstance(key_node, yaml.ScalarNode)
            key = key_node.value if scalar else self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


_BotYamlLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
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


def load_bot(directory: Path, actions: Iterable[str] | None = None) -> BotDefinition:
    """Parse `directory/bot.yaml` (+ optional `nlu.yaml`) and validate it.
    If `actions` is given, every action step must name one of them."""
    bot_file, nlu_file = directory / "bot.yaml", directory / "nlu.yaml"
    if not bot_file.is_file():
        raise BotError([f"{bot_file} not found"])
    files = [bot_file] + ([nlu_file] if nlu_file.is_file() else [])
    raw = _read_yaml(bot_file)
    nlu = _read_yaml(nlu_file) if nlu_file.is_file() else {}
    overlap = sorted(set(raw) & set(nlu))
    if overlap:
        raise BotError([f"keys defined in both bot.yaml and nlu.yaml: {overlap}"])
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:12]
    try:
        bot = BotDefinition.model_validate({**raw, **nlu, "version": digest})
    except ValidationError as exc:
        raise BotError([f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]) from exc
    report = validate(bot, actions)
    if report.errors:
        raise BotError(report.errors)
    return bot


def validate(bot: BotDefinition, actions: Iterable[str] | None = None) -> Report:
    errors: list[str] = []
    warnings: list[str] = []
    known_actions = set(actions) if actions is not None else None

    missing = sorted(REQUIRED_RESPONSES - set(bot.responses))
    if missing:
        errors.append(f"missing required responses: {missing}")
    errors += _template_errors(bot)

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
            if slot.entity not in bot.entities and slot.entity not in SYSTEM_ENTITIES:
                errors.append(f"{where}: slot {name!r} uses unknown entity {slot.entity!r}")
            if slot.ask not in bot.responses:
                errors.append(f"{where}: slot {name!r} asks with unknown response {slot.ask!r}")
        for name, slot in sc.slots.items():
            if slot.ask in bot.responses and f"slots.{name}" in _paths_of(bot.responses[slot.ask].bodies()):
                errors.append(f"{where}: response {slot.ask!r} asks for slot {name!r} but also shows it")
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
    for where, texts in [(f"response {rid!r}", r.bodies() + r.quick_replies) for rid, r in bot.responses.items()] + [
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
        texts = resp.bodies() + resp.quick_replies
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
