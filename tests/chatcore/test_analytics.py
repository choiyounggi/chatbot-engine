"""Conversation analytics (docs/engine-core-plan.md 빌더 단계 E-E, E12): the done criterion (a scripted
conversation run through the real engine, compared against a literal expected table), unit tests for
every definition edge, the two store methods it reads from, and the HTTP route."""

from __future__ import annotations

import datetime as dt
import random
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from chatcore.actions.builtin import KST
from chatcore.analytics import ACTIVITY_TYPES, MAX_DAYS, SESSION_GAP, compute_report, start_of
from chatcore.api import create_app
from chatcore.bot.loader import read_document
from chatcore.bot.repository import InMemoryRepository
from chatcore.conversations import synthetic_actions
from chatcore.dialogue.commands import AnswerKnowledge, Command, Fallback, StartScenario
from chatcore.dialogue.events import BotUttered, CommandsReceived, ScenarioStarted, SessionStarted, UserUttered
from chatcore.dialogue.turn import Interpretation, TurnProcessor
from chatcore.host import BotHost
from chatcore.store.memory import InMemoryStore

from .conftest import SAMPLE

B = "/v1/builder/bots/sample"


def utc(y: int, m: int, d: int, h: int, mi: int, s: int = 0) -> dt.datetime:
    return dt.datetime(y, m, d, h, mi, s, tzinfo=dt.UTC)


def kst(y: int, m: int, d: int, h: int, mi: int, s: int = 0) -> dt.datetime:
    return dt.datetime(y, m, d, h, mi, s, tzinfo=KST).astimezone(dt.UTC)


# ---------------------------------------------------------------- done criterion: a scripted conversation


class FakeClock:
    """A clock test double: holds a fixed instant until the test moves it forward."""

    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now


@dataclass
class RuleInterpreter:
    """Test double like ScriptedInterpreter, but lets the test pick the logged `nlu.rule` (ScriptedInterpreter
    always says "scripted", so it cannot produce the "fallback"/"knowledge" rules analytics reads)."""

    script: Mapping[str, tuple[Sequence[Command], str]] = field(default_factory=dict)

    def interpret(self, text: str, state: object, bot: object) -> Interpretation:
        commands, rule = self.script.get(text, ([Fallback()], "fallback"))
        return Interpretation(list(commands), {"rule": rule, "text": text})


SCRIPT: dict[str, tuple[Sequence[Command], str]] = {
    "안녕": ([StartScenario("greet")], "scripted"),
    "시간": ([StartScenario("current_time")], "scripted"),
    "환불 되나요": ([AnswerKnowledge("네, 환불됩니다.")], "knowledge"),
}


def test_done_criterion_scripted_conversation_matches_expected_table(sample_bot, registry):
    """3 users, 5 sessions, 2 KST days, known block calls — docs/engine-core-plan.md's done criterion."""
    clock = FakeClock(utc(2026, 11, 1, 0, 0))
    store = InMemoryStore(clock=clock)
    proc = TurnProcessor(sample_bot, store, RuleInterpreter(SCRIPT), registry, rng=random.Random(0), clock=clock)

    def say(session: str, at: dt.datetime, text: str) -> None:
        clock.now = at
        proc.handle(session, text)

    # day 1 (2026-11-01 KST) --------------------------------------------------------------
    say("sample:u1", kst(2026, 11, 1, 9, 0), "안녕")          # u1 session 1 starts (greet)
    say("sample:u1", kst(2026, 11, 1, 9, 10), "모르는말")      # same session (gap 10 min): fallback
    say("sample:u1", kst(2026, 11, 1, 9, 50), "안녕")          # gap 40 min > 30: u1 session 2 (greet)
    say("sample:u2", kst(2026, 11, 1, 10, 0), "환불 되나요")   # u2 session 3 starts: knowledge
    say("sample:u2", kst(2026, 11, 1, 10, 29), "시간")         # same session (gap 29 min): current_time
    say("sample:u3", kst(2026, 11, 1, 23, 50), "안녕")         # u3 session 4 starts (greet)
    # day 2 (2026-11-02 KST), session 4 continues across midnight ------------------------
    say("sample:u3", kst(2026, 11, 2, 0, 10), "모르는말")      # gap 20 min: still session 4, counted on day 2
    say("sample:u1", kst(2026, 11, 2, 9, 5), "시간")           # gap 23h15m > 30: u1 session 5 (current_time)

    # a test session ("<bot>~...", never "<bot>:...") must never be counted
    say("sample~x", kst(2026, 11, 1, 12, 0), "안녕")

    from_date, to_date = dt.date(2026, 11, 1), dt.date(2026, 11, 2)
    prefix = "sample:"
    rows = store.activity(prefix, ACTIVITY_TYPES, start_of(from_date) - SESSION_GAP,
                          start_of(to_date + dt.timedelta(days=8)))
    first_seen = store.first_seen(prefix)
    labels = {sid: s.label for sid, s in sample_bot.scenarios.items()}
    report = compute_report(rows, first_seen, from_date, to_date, labels, today=dt.date(2026, 11, 10))

    t = report.totals
    assert (t.users, t.new_users, t.returning_users, t.sessions, t.messages) == (3, 3, 0, 5, 8)
    assert (t.block_calls, t.fallbacks, t.knowledge_answers) == (5, 2, 1)
    assert t.knowledge_rate == pytest.approx(1 / 3)

    assert len(report.days) == 2
    d1, d2 = report.days
    assert d1.date == dt.date(2026, 11, 1)
    assert (d1.users, d1.new_users, d1.returning_users, d1.sessions) == (3, 3, 0, 4)
    assert (d1.messages, d1.block_calls, d1.fallbacks, d1.knowledge_answers) == (6, 4, 1, 1)
    assert d2.date == dt.date(2026, 11, 2)
    assert (d2.users, d2.new_users, d2.returning_users, d2.sessions) == (2, 0, 2, 1)
    assert (d2.messages, d2.block_calls, d2.fallbacks, d2.knowledge_answers) == (2, 1, 1, 0)

    assert [(b.block, b.label, b.calls) for b in report.blocks] == [
        ("greet", "인사", 3), ("current_time", "현재 시각", 2),
    ]

    assert len(report.retention) == 1
    r = report.retention[0]
    assert (r.date, r.new_users) == (dt.date(2026, 11, 1), 3)
    assert r.d1 == pytest.approx(2 / 3)
    assert r.d7 == pytest.approx(0.0)


# ---------------------------------------------------------------- unit tests: definition edges


def test_session_boundary_exactly_30_minutes_is_the_same_session():
    sid = "sample:u1"
    rows = [(sid, _user_uttered(), kst(2026, 11, 1, 9, 0)), (sid, _user_uttered(), kst(2026, 11, 1, 9, 30))]
    d = dt.date(2026, 11, 1)
    report = compute_report(rows, {sid: kst(2026, 11, 1, 9, 0)}, d, d, {}, today=d)
    assert report.days[0].sessions == 1 and report.totals.messages == 2


def test_session_boundary_30_minutes_and_one_second_is_a_new_session():
    sid = "sample:u1"
    rows = [(sid, _user_uttered(), kst(2026, 11, 1, 9, 0)), (sid, _user_uttered(), kst(2026, 11, 1, 9, 30, 1))]
    d = dt.date(2026, 11, 1)
    report = compute_report(rows, {sid: kst(2026, 11, 1, 9, 0)}, d, d, {}, today=d)
    assert report.days[0].sessions == 2


def test_session_crossing_midnight_kst_belongs_to_its_first_days_count():
    sid = "sample:u1"
    rows = [(sid, _user_uttered(), kst(2026, 11, 1, 23, 50)), (sid, _user_uttered(), kst(2026, 11, 2, 0, 10))]
    first_seen = {sid: kst(2026, 11, 1, 23, 50)}
    report = compute_report(rows, first_seen, dt.date(2026, 11, 1), dt.date(2026, 11, 2), {},
                            today=dt.date(2026, 11, 2))
    d1, d2 = report.days
    assert d1.sessions == 1 and d1.messages == 1  # the session starts on day 1
    assert d2.sessions == 0 and d2.messages == 1  # its second message is on day 2, but not a new session


def test_session_started_before_from_is_not_counted_but_its_message_is():
    sid = "sample:u1"
    d = dt.date(2026, 11, 2)
    # the buffer row (before `from`) only establishes the gap; the in-period row continues it
    rows = [(sid, _user_uttered(), kst(2026, 11, 1, 23, 45)), (sid, _user_uttered(), kst(2026, 11, 2, 0, 5))]
    first_seen = {sid: kst(2026, 11, 1, 23, 45)}
    report = compute_report(rows, first_seen, d, d, {}, today=d)
    assert report.days[0].sessions == 0
    assert report.days[0].messages == 1
    assert report.days[0].users == 1 and report.days[0].new_users == 0 and report.days[0].returning_users == 1


def test_utc_instant_is_bucketed_by_its_kst_day():
    sid = "sample:u1"
    at = utc(2026, 11, 1, 23, 30)  # 2026-11-02 08:30 KST: the next calendar day
    first_seen = {sid: at}
    next_day = dt.date(2026, 11, 2)
    same_day = dt.date(2026, 11, 1)
    report_next = compute_report([(sid, _user_uttered(), at)], first_seen, next_day, next_day, {}, today=next_day)
    assert report_next.totals.messages == 1
    report_same = compute_report([(sid, _user_uttered(), at)], first_seen, same_day, same_day, {}, today=same_day)
    assert report_same.totals.messages == 0


def test_empty_period_is_all_zeros_and_no_knowledge_rate():
    d = dt.date(2026, 11, 1)
    report = compute_report([], {}, d, d, {}, today=d)
    assert len(report.days) == 1
    t = report.totals
    assert (t.users, t.new_users, t.returning_users, t.sessions, t.messages) == (0, 0, 0, 0, 0)
    assert (t.block_calls, t.fallbacks, t.knowledge_answers, t.knowledge_rate) == (0, 0, 0, None)
    assert report.blocks == [] and report.retention == []


def test_retention_is_null_for_days_not_yet_observed():
    sid = "sample:u1"
    d = dt.date(2026, 11, 1)
    rows = [(sid, _user_uttered(), kst(2026, 11, 1, 9, 0))]
    first_seen = {sid: kst(2026, 11, 1, 9, 0)}
    # today == the new-user day itself: tomorrow and +7 days have not happened yet
    report = compute_report(rows, first_seen, d, d, {}, today=d)
    assert len(report.retention) == 1
    assert report.retention[0].d1 is None and report.retention[0].d7 is None
    # today one day later: d1 is now observable (nobody returned), d7 still is not
    later = compute_report(rows, first_seen, d, d, {}, today=d + dt.timedelta(days=1))
    assert later.retention[0].d1 == pytest.approx(0.0) and later.retention[0].d7 is None


def test_block_ranking_ties_break_by_id():
    sid = "sample:u1"
    at = kst(2026, 11, 1, 9, 0)
    rows = [(sid, _scenario_started("b_block"), at), (sid, _scenario_started("a_block"), at)]
    d = dt.date(2026, 11, 1)
    report = compute_report(rows, {}, d, d, {}, today=d)
    assert [b.block for b in report.blocks] == ["a_block", "b_block"]


def test_block_label_falls_back_to_its_id_when_unlabeled():
    sid = "sample:u1"
    at = kst(2026, 11, 1, 9, 0)
    report = compute_report([(sid, _scenario_started("mystery"), at)], {}, dt.date(2026, 11, 1), dt.date(2026, 11, 1),
                            {"mystery": "레이블"}, today=dt.date(2026, 11, 1))
    assert len(report.blocks) == 1 and report.blocks[0].label == "레이블"
    report_unlabeled = compute_report([(sid, _scenario_started("mystery"), at)], {}, dt.date(2026, 11, 1),
                                      dt.date(2026, 11, 1), {}, today=dt.date(2026, 11, 1))
    assert report_unlabeled.blocks[0].label == "mystery"


def test_today_defaults_to_the_current_kst_date():
    sid = "sample:u1"
    at = kst(2026, 11, 1, 9, 0)
    d = dt.date(2026, 11, 1)
    report = compute_report([(sid, _user_uttered(), at)], {sid: at}, d, d, {})  # no `today`: real KST clock
    now_kst = dt.datetime.now(KST).date()
    expected = None if d + dt.timedelta(days=1) > now_kst else pytest.approx(0.0)  # nobody returned in these rows
    assert report.retention[0].d1 == expected


def test_to_before_from_is_rejected():
    d = dt.date(2026, 11, 1)
    with pytest.raises(ValueError, match="after"):
        compute_report([], {}, d, d - dt.timedelta(days=1), {}, today=d)


def _user_uttered() -> UserUttered:
    return UserUttered(text="x")


def _scenario_started(scenario: str) -> ScenarioStarted:
    return ScenarioStarted(scenario=scenario)


# ---------------------------------------------------------------- store methods (memory + postgres)


def postgres_store():
    import os

    db_url = os.environ.get("CHATCORE_TEST_DATABASE_URL")
    if not db_url:
        if os.environ.get("CI"):
            pytest.fail("CHATCORE_TEST_DATABASE_URL must be set in CI")
        pytest.skip("set CHATCORE_TEST_DATABASE_URL (make test-db) to run Postgres tests")
    from chatcore.store.postgres import PostgresStore

    store = PostgresStore(db_url)
    store.ensure_schema()
    return store


STORES = {"memory": InMemoryStore, "postgres": postgres_store}


@pytest.fixture(params=sorted(STORES))
def store(request):
    return STORES[request.param]()


def test_activity_filters_by_prefix_type_and_window(store):
    run = uuid.uuid4().hex[:8]
    a, look_alike = f"act_{run}:s1", f"act_{run}x:s1"  # "_" must not act as a LIKE wildcard
    with store.session(a) as tracker:
        tracker.append([UserUttered(text="하나"), CommandsReceived(commands=[], nlu={"rule": "fallback"}),
                        BotUttered(response="r", text="r")])
    with store.session(look_alike) as tracker:
        tracker.append([UserUttered(text="other bot")])
    with store.session(a) as tracker:
        tracker.append([UserUttered(text="둘")])

    found = store.activity(f"act_{run}:", frozenset({"user_uttered"}), None, None)
    assert [e.text for _, e, _ in found] == ["하나", "둘"]  # oldest first; the look-alike prefix excluded
    assert all(at.tzinfo is not None for _, _, at in found)

    mixed = store.activity(f"act_{run}:", frozenset({"user_uttered", "commands_received"}), None, None)
    assert [type(e).__name__ for _, e, _ in mixed] == ["UserUttered", "CommandsReceived", "UserUttered"]

    first_at = found[0][2]
    assert store.activity(f"act_{run}:", frozenset({"user_uttered"}), first_at, first_at) == []  # [since, until)
    assert len(store.activity(f"act_{run}:", frozenset({"user_uttered"}), first_at, None)) == 2
    assert store.activity(f"none_{run}:", frozenset({"user_uttered"}), None, None) == []


def test_first_seen_is_the_earliest_user_uttered_per_session(store):
    run = uuid.uuid4().hex[:8]
    a, b = f"fs_{run}:s1", f"fs_{run}:s2"
    with store.session(a) as tracker:
        tracker.append([BotUttered(response="r", text="r"), UserUttered(text="first")])
    with store.session(a) as tracker:
        tracker.append([UserUttered(text="second")])
    with store.session(b) as tracker:
        tracker.append([UserUttered(text="only")])

    seen = store.first_seen(f"fs_{run}:")
    assert set(seen) == {a, b}
    # the earlier of session a's two user_uttered events, not the later one
    a_events = store.activity(f"fs_{run}:", frozenset({"user_uttered"}), None, None)
    assert seen[a] == min(at for sid, _, at in a_events if sid == a)
    assert store.first_seen(f"none_{run}:") == {}


# ---------------------------------------------------------------- HTTP route


@pytest.fixture
def host(fake_embedder, tmp_path, trained_sample):
    shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
    h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions, keep=1,
                min_f1=0.0)
    h.seed(SAMPLE)
    return h


@pytest.fixture
def client(host):
    return TestClient(create_app(host))


def seed_scenario_started(host, bot: str, session: str, scenario: str) -> None:
    with host.store.session(f"{bot}:{session}") as tracker:
        tracker.append([SessionStarted(bot_id=bot, bot_version="v"), UserUttered(text="x"),
                        ScenarioStarted(scenario=scenario)])


def test_analytics_default_period_is_the_last_7_days_ending_today_kst(client):
    resp = client.get(f"{B}/analytics")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    today = dt.datetime.now(KST).date()
    assert body["to"] == today.isoformat()
    assert body["from"] == (today - dt.timedelta(days=6)).isoformat()
    assert body["timezone"] == "Asia/Seoul" and body["bot"] == "sample"


def test_analytics_rejects_malformed_date(client):
    assert client.get(f"{B}/analytics", params={"from": "not-a-date", "to": "2026-11-01"}).status_code == 422


def test_analytics_rejects_from_after_to(client):
    r = client.get(f"{B}/analytics", params={"from": "2026-11-05", "to": "2026-11-01"})
    assert r.status_code == 422


def test_analytics_rejects_a_period_over_92_days(client):
    r = client.get(f"{B}/analytics", params={"from": "2026-01-01", "to": "2026-12-31"})
    assert r.status_code == 422


def test_analytics_accepts_the_92_day_maximum(client):
    r = client.get(f"{B}/analytics", params={"from": "2026-01-01", "to": "2026-04-02"})
    assert r.status_code == 200, r.text
    assert (dt.date(2026, 4, 2) - dt.date(2026, 1, 1)).days + 1 == MAX_DAYS


def test_analytics_unknown_bot_is_404(client):
    assert client.get("/v1/builder/bots/nope/analytics").status_code == 404


def test_analytics_labels_come_from_the_live_runtime(client, host):
    seed_scenario_started(host, "sample", "u1", "check_weather")
    today = dt.datetime.now(KST).date()
    r = client.get(f"{B}/analytics", params={"from": today.isoformat(), "to": today.isoformat()})
    blocks = {b["block"]: b for b in r.json()["blocks"]}
    assert blocks["check_weather"] == {"block": "check_weather", "label": "날씨 안내", "calls": 1}


def test_analytics_labels_fall_back_to_the_draft_when_not_deployed(host):
    document = read_document(SAMPLE)
    document["id"] = "draftonly"
    document["scenarios"] = {"only_block": {"label": "초안 라벨", "steps": [{"respond": "greeting"}]}}
    document["intents"], document["entities"] = {}, {}
    required = {"fallback", "cancelled", "restarted", "error", "clarify", "input_invalid"}
    document["responses"] = {k: v for k, v in document["responses"].items() if k in required or k == "greeting"}
    check = host.save_draft("draftonly", document)
    assert not check.errors, check.errors
    assert host.runtime("draftonly") is None

    seed_scenario_started(host, "draftonly", "u1", "only_block")
    client = TestClient(create_app(host))
    today = dt.datetime.now(KST).date()
    r = client.get("/v1/builder/bots/draftonly/analytics", params={"from": today.isoformat(), "to": today.isoformat()})
    assert r.status_code == 200, r.text
    assert r.json()["blocks"] == [{"block": "only_block", "label": "초안 라벨", "calls": 1}]


def test_analytics_requires_the_admin_token(host):
    protected = TestClient(create_app(host, admin_token="s3cr3t"))
    assert protected.get(f"{B}/analytics").status_code == 401
    assert protected.get(f"{B}/analytics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = protected.get(f"{B}/analytics", headers={"Authorization": "Bearer s3cr3t"})
    assert ok.status_code == 200, ok.text

