"""The builder's training queue (학습대기, builder plan E6): what users said, grouped by sentence.

Each group is one sentence (compared like exact matches: spacing and edge punctuation ignored) with
how often it was said, by how many sessions, when, and how the bot understood it the last time.
What the builder did with a sentence (added it as an example, ignored it) is a Mark in the bot
repository; the latest mark is the sentence's status.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from chatcore.bot.repository import Mark
from chatcore.nlu.entities import bare
from chatcore.store.base import Utterance

MISUNDERSTOOD = frozenset({"fallback", "clarify"})
NOT_SENTENCES = frozenset({"welcome", "block"})  # opening a chat and pressing a block button: nothing was said
RuleFilter = Literal["misunderstood", "all"]
StatusFilter = Literal["pending", "ignored", "added", "all"]


def key(text: str) -> str:
    """The grouping key of a sentence (what an exact match compares)."""
    return bare(text)


@dataclass(frozen=True)
class QueueItem:
    key: str
    text: str  # the latest surface form
    count: int
    sessions: int
    first_seen: str  # ISO 8601, UTC
    last_seen: str
    rule: str  # how the latest one was handled: fallback, clarify, intent, slot, ...
    intent: str | None  # the classifier's top intent the latest time
    score: float | None
    status: str  # pending | ignored | added
    status_intent: str | None  # the intent it was added to


def _iso(at: dt.datetime) -> str:
    return at.astimezone(dt.UTC).isoformat(timespec="seconds")


@dataclass
class _Group:
    text: str  # from the latest message
    rule: str
    intent: str | None
    score: float | None
    last: dt.datetime
    first: dt.datetime
    count: int = 0
    sessions: set[str] = field(default_factory=set)


def _top(nlu: dict[str, object]) -> tuple[str | None, float | None]:
    intents = nlu.get("intents")
    top = intents[0] if isinstance(intents, list) and intents else None
    if not isinstance(top, list | tuple) or len(top) < 2:
        return None, None
    name, score = top[0], top[1]
    return (str(name), float(score) if isinstance(score, int | float) and not isinstance(score, bool) else None)


def group(utterances: Iterable[Utterance], marks: Sequence[Mark], rule: RuleFilter = "misunderstood",
          intent: str | None = None, status: StatusFilter = "pending", limit: int = 100) -> list[QueueItem]:
    """Group `utterances` (newest first) by sentence, most frequent first, then most recent.
    rule "misunderstood": only sentences the bot fell back on or had to ask "A or B?" about the
    latest time; intent: only those whose latest top intent is this one; status: by the latest mark."""
    latest_mark: dict[str, Mark] = {}
    for m in marks:  # newest first: the first one per key is the latest
        latest_mark.setdefault(m.key, m)
    groups: dict[str, _Group] = {}
    for u in utterances:
        text, handled = u.nlu.get("text"), u.nlu.get("rule")
        if not isinstance(text, str) or handled in NOT_SENTENCES or not key(text):
            continue
        k = key(text)
        if k not in groups:  # the first one seen is the latest
            top_intent, top_score = _top(u.nlu)
            groups[k] = _Group(text, str(handled or ""), top_intent, top_score, u.at, u.at)
        g = groups[k]
        g.count += 1
        g.first = u.at  # older ones come later
        g.sessions.add(u.session_id)
    items: list[QueueItem] = []
    for k, g in groups.items():
        mark = latest_mark.get(k)
        state = mark.status if mark is not None else "pending"
        if ((rule == "misunderstood" and g.rule not in MISUNDERSTOOD) or (intent is not None and g.intent != intent)
                or (status != "all" and state != status)):
            continue
        items.append(QueueItem(
            key=k, text=g.text, count=g.count, sessions=len(g.sessions), first_seen=_iso(g.first),
            last_seen=_iso(g.last), rule=g.rule, intent=g.intent, score=g.score, status=state,
            status_intent=mark.intent if mark is not None and state == "added" else None,
        ))
    items.sort(key=lambda i: i.last_seen, reverse=True)  # stable: the count sort below keeps this order on ties
    items.sort(key=lambda i: i.count, reverse=True)
    return items[:max(limit, 0)]
