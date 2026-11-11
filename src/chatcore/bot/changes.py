"""What changed between two saves of a bot draft (the builder's 작업이력, builder plan E4).

Compares two documents section by section and returns one line per added, removed or changed
item. Works on any JSON documents, including invalid drafts (a builder saves work in progress).
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

Area = Literal["bot", "settings", "system", "block", "intent", "entity", "response", "skill"]
Action = Literal["added", "removed", "changed"]

# document section -> area shown to the builder (a scenario is a block there)
SECTIONS: dict[str, Area] = {"scenarios": "block", "intents": "intent", "entities": "entity", "responses": "response",
                             "skills": "skill"}


class Change(TypedDict, total=False):
    area: Area
    id: str
    label: str
    action: Action
    examples_added: int  # intents only
    examples_removed: int


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)


def _mapping(doc: dict[str, Any] | None, key: str) -> dict[str, Any]:
    value = (doc or {}).get(key)
    return value if isinstance(value, dict) else {}


def _label(item: Any) -> str:
    label = item.get("label") if isinstance(item, dict) else None
    return label if isinstance(label, str) else ""


def _examples(item: Any) -> list[str]:
    examples = item.get("examples") if isinstance(item, dict) else None
    return [e for e in examples if isinstance(e, str)] if isinstance(examples, list) else []


def summarize(old: dict[str, Any] | None, new: dict[str, Any]) -> list[Change]:
    """One entry per changed item, in document order: bot label, settings, system, then
    blocks, intents, entities, responses (each by id). `old` None: the first save (everything added
    is not listed one by one; a single "bot added" entry is returned)."""
    if old is None:
        return [Change(area="bot", id=str(new.get("id", "")), label=_label(new), action="added")]
    changes: list[Change] = []
    bot_id = str(new.get("id", ""))
    if not _same(old.get("label"), new.get("label")):
        changes.append(Change(area="bot", id=bot_id, label=_label(new), action="changed"))
    for key in ("settings", "system"):
        if not _same(old.get(key), new.get(key)):
            action: Action = "added" if key not in old else "removed" if key not in new else "changed"
            area: Area = "settings" if key == "settings" else "system"
            changes.append(Change(area=area, id=key, label="", action=action))
    for section, area in SECTIONS.items():
        before, after = _mapping(old, section), _mapping(new, section)
        for item_id in [*after, *(k for k in before if k not in after)]:
            if item_id not in before:
                change = Change(area=area, id=item_id, label=_label(after[item_id]), action="added")
            elif item_id not in after:
                change = Change(area=area, id=item_id, label=_label(before[item_id]), action="removed")
            elif not _same(before[item_id], after[item_id]):
                change = Change(area=area, id=item_id, label=_label(after[item_id]), action="changed")
            else:
                continue
            if area == "intent":
                was, now = _examples(before.get(item_id)), _examples(after.get(item_id))
                change["examples_added"] = len([e for e in now if e not in was])
                change["examples_removed"] = len([e for e in was if e not in now])
            changes.append(change)
    return changes
