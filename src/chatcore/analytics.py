"""Conversation analytics (docs/engine-core-plan.md 빌더 단계 E-E, E12): computed on request from the
stored event log, never materialized into a separate table.

All day boundaries are Korean local time (KST has no DST, so this is a fixed +09:00 offset).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass

from chatcore.actions.builtin import KST
from chatcore.dialogue.events import CommandsReceived, Event, ScenarioStarted, UserUttered

SESSION_GAP = dt.timedelta(minutes=30)
MAX_DAYS = 92
RETENTION_DAYS = (1, 7)
ACTIVITY_TYPES = frozenset({"user_uttered", "commands_received", "scenario_started"})


def start_of(day: dt.date) -> dt.datetime:
    """Midnight KST of `day`, as a timezone-aware datetime."""
    return dt.datetime.combine(day, dt.time.min, tzinfo=KST)


def _kst_date(at: dt.datetime) -> dt.date:
    return at.astimezone(KST).date()


@dataclass(frozen=True)
class Totals:
    users: int
    new_users: int
    returning_users: int
    sessions: int
    messages: int
    block_calls: int
    fallbacks: int
    knowledge_answers: int
    knowledge_rate: float | None


@dataclass(frozen=True)
class DayStats:
    date: dt.date
    users: int
    new_users: int
    returning_users: int
    sessions: int
    messages: int
    block_calls: int
    fallbacks: int
    knowledge_answers: int


@dataclass(frozen=True)
class BlockStats:
    block: str
    label: str
    calls: int


@dataclass(frozen=True)
class RetentionRow:
    date: dt.date
    new_users: int
    d1: float | None
    d7: float | None


@dataclass(frozen=True)
class Report:
    from_: dt.date
    to: dt.date
    totals: Totals
    days: list[DayStats]
    blocks: list[BlockStats]
    retention: list[RetentionRow]


def compute_report(rows: list[tuple[str, Event, dt.datetime]], first_seen: Mapping[str, dt.datetime],
                   from_date: dt.date, to_date: dt.date, labels: Mapping[str, str],
                   today: dt.date | None = None) -> Report:
    """`rows`: activity(prefix, {"user_uttered", "commands_received", "scenario_started"},
    since=start_of(from_date) - SESSION_GAP, until=start_of(to_date + 8 days)), oldest first.
    `first_seen`: session id -> when its first user_uttered was ever stored (full history).
    Pure: takes no store or host, so it is deterministic given these inputs."""
    if to_date < from_date:
        raise ValueError(f"from_date {from_date} is after to_date {to_date}")
    if today is None:
        today = dt.datetime.now(KST).date()

    span = (to_date - from_date).days + 1
    days = [from_date + dt.timedelta(days=i) for i in range(span)]
    period_start, period_end = start_of(from_date), start_of(to_date + dt.timedelta(days=1))

    messages = {d: 0 for d in days}
    block_calls = {d: 0 for d in days}
    fallbacks = {d: 0 for d in days}
    knowledge_answers = {d: 0 for d in days}
    session_starts = {d: 0 for d in days}
    block_totals: dict[str, int] = {}
    active_by_day: dict[dt.date, set[str]] = {}  # extends past `to_date` (retention lookahead)

    last_uttered: dict[str, dt.datetime] = {}
    for sid, event, at in rows:
        if isinstance(event, UserUttered):
            day = _kst_date(at)
            prev = last_uttered.get(sid)
            starts_session = prev is None or (at - prev) > SESSION_GAP
            last_uttered[sid] = at
            if at >= period_start:
                active_by_day.setdefault(day, set()).add(sid)
            if period_start <= at < period_end:
                messages[day] += 1
                if starts_session:
                    session_starts[day] += 1
        elif isinstance(event, ScenarioStarted) and period_start <= at < period_end:
            day = _kst_date(at)
            block_calls[day] += 1
            block_totals[event.scenario] = block_totals.get(event.scenario, 0) + 1
        elif isinstance(event, CommandsReceived) and period_start <= at < period_end:
            day = _kst_date(at)
            rule = event.nlu.get("rule")
            if rule == "fallback":
                fallbacks[day] += 1
            elif rule == "knowledge":
                knowledge_answers[day] += 1

    def new_on(day: dt.date) -> set[str]:
        return {sid for sid in active_by_day.get(day, set()) if _kst_date(first_seen[sid]) == day}

    day_rows = []
    for d in days:
        active = active_by_day.get(d, set())
        new = new_on(d)
        day_rows.append(DayStats(
            date=d, users=len(active), new_users=len(new), returning_users=len(active) - len(new),
            sessions=session_starts[d], messages=messages[d], block_calls=block_calls[d],
            fallbacks=fallbacks[d], knowledge_answers=knowledge_answers[d],
        ))

    all_active = {sid for d in days for sid in active_by_day.get(d, set())}
    new_users = {sid for sid in all_active if from_date <= _kst_date(first_seen[sid]) <= to_date}
    total_fallbacks, total_knowledge = sum(fallbacks.values()), sum(knowledge_answers.values())
    knowledge_rate = (total_knowledge / (total_knowledge + total_fallbacks)
                      if total_knowledge + total_fallbacks > 0 else None)
    totals = Totals(
        users=len(all_active), new_users=len(new_users), returning_users=len(all_active) - len(new_users),
        sessions=sum(session_starts.values()), messages=sum(messages.values()),
        block_calls=sum(block_calls.values()), fallbacks=total_fallbacks, knowledge_answers=total_knowledge,
        knowledge_rate=knowledge_rate,
    )

    blocks = [BlockStats(block=bid, label=labels.get(bid, bid), calls=calls)
              for bid, calls in sorted(block_totals.items(), key=lambda kv: (-kv[1], kv[0]))]

    retention = []
    for d in days:
        new = new_on(d)
        if not new:
            continue
        shares: dict[int, float | None] = {}
        for ahead in RETENTION_DAYS:
            target = d + dt.timedelta(days=ahead)
            shares[ahead] = (None if target > today
                            else len(new & active_by_day.get(target, set())) / len(new))
        retention.append(RetentionRow(date=d, new_users=len(new), d1=shares[1], d7=shares[7]))

    return Report(from_=from_date, to=to_date, totals=totals, days=day_rows, blocks=blocks, retention=retention)
