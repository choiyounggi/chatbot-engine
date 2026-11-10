"""Builder stage B4, engine side: 작업이력 (E4) and 학습대기 (E6).

Draft saves record what changed; user sentences are grouped by sentence over a period, filtered by
how they were handled, which intent they went to and what the builder did with them; adding one to
an intent edits the draft like a save (ETag rules) and the sentence then goes to that intent.
"""

import copy
import datetime as dt
import shutil
import uuid

import pytest
from fastapi.testclient import TestClient

from chatcore.api import create_app
from chatcore.bot.changes import summarize
from chatcore.bot.loader import read_document
from chatcore.bot.repository import InMemoryRepository, Mark
from chatcore.conversations import synthetic_actions
from chatcore.dialogue.events import CommandsReceived
from chatcore.host import BotHost
from chatcore.store.base import Utterance
from chatcore.store.memory import InMemoryStore
from chatcore.training_queue import group, key

from .conftest import SAMPLE
from .test_builder import B, draft, finish, postgres_repository, save, say, ship, trained_sample  # noqa: F401
from .test_turn_and_store import STORES

T0 = dt.datetime(2026, 11, 10, 9, 0, tzinfo=dt.UTC)


# ---------------------------------------------------------------- E4: what a save changed


def test_first_save_is_one_bot_added_line():
    assert summarize(None, {"id": "b", "label": "봇"}) == [{"area": "bot", "id": "b", "label": "봇", "action": "added"}]


def test_summary_lists_each_changed_item_with_example_counts():
    old = read_document(SAMPLE)
    new = copy.deepcopy(old)
    new["label"] = "새 이름"
    new["system"] = {"welcome": "greet"}
    new["settings"] = {**new["settings"], "max_attempts": 5}
    new["scenarios"]["coupon"] = {"label": "쿠폰", "steps": [{"respond": "help"}]}
    del new["scenarios"]["praise"]
    new["scenarios"]["greet"]["label"] = "인사하기"
    new["intents"]["greeting"]["examples"] = new["intents"]["greeting"]["examples"][2:] + ["테스트로 넣은 인사 하나", "테스트로 넣은 인사 둘"]
    new["responses"]["help"]["text"] = "무엇이든 물어보세요"
    changes = summarize(old, new)
    assert {"area": "bot", "id": "sample", "label": "새 이름", "action": "changed"} in changes
    assert {"area": "system", "id": "system", "label": "", "action": "added"} in changes
    assert {"area": "settings", "id": "settings", "label": "", "action": "changed"} in changes
    assert {"area": "block", "id": "coupon", "label": "쿠폰", "action": "added"} in changes
    assert {"area": "block", "id": "praise", "label": "칭찬 답례", "action": "removed"} in changes
    assert {"area": "block", "id": "greet", "label": "인사하기", "action": "changed"} in changes
    assert {"area": "response", "id": "help", "label": "", "action": "changed"} in changes
    greeting = next(c for c in changes if c["area"] == "intent")
    assert greeting == {"area": "intent", "id": "greeting", "label": "인사", "action": "changed",
                        "examples_added": 2, "examples_removed": 2}
    assert len(changes) == 8
    assert summarize(old, copy.deepcopy(old)) == []  # nothing changed
    removed = {k: v for k, v in new.items() if k != "system"}
    assert {"area": "system", "id": "system", "label": "", "action": "removed"} in summarize(new, removed)


def test_summary_tolerates_invalid_drafts():
    # a builder saves work in progress: sections may be missing or the wrong type
    assert summarize({"id": "b", "scenarios": "oops"}, {"id": "b", "scenarios": {"s": 3}, "intents": {"i": {"examples": "x"}}}) == [
        {"area": "block", "id": "s", "label": "", "action": "added"},
        {"area": "intent", "id": "i", "label": "", "action": "added", "examples_added": 0, "examples_removed": 0}]


# ---------------------------------------------------------------- repositories and stores (memory + Postgres)


@pytest.fixture(params=["memory", "postgres"])
def repository(request):
    return InMemoryRepository() if request.param == "memory" else postgres_repository()


def test_repository_records_edits_and_marks(repository):
    bot = f"t_{uuid.uuid4().hex[:8]}"
    doc = {**read_document(SAMPLE), "id": bot}
    assert repository.save_draft(bot, doc, 0, [{"area": "bot", "id": bot, "label": "", "action": "added"}]) == 1
    assert repository.save_draft(bot, doc, 1, []) == 2  # nothing changed: no edit recorded
    assert repository.save_draft(bot, doc, 2, [{"area": "block", "id": "s", "label": "", "action": "added"}]) == 3
    edits = repository.edits(bot)
    assert [(e.revision, e.changes[0]["area"]) for e in edits] == [(3, "block"), (1, "bot")]
    assert repository.edits(bot, limit=1)[0].revision == 3 and repository.edits(bot, limit=0) == []
    assert edits[0].saved_at.endswith("+00:00")
    repository.mark_utterances(bot, [("a", "a!"), ("b", "b")], "ignored")
    repository.mark_utterances(bot, [("a", "a")], "added", "greeting")
    repository.mark_utterances(bot, [], "pending")  # nothing to mark
    marks = repository.utterance_marks(bot)
    assert [(m.key, m.status, m.intent) for m in marks[:1]] == [("a", "added", "greeting")]
    assert sorted((m.key, m.status) for m in marks[1:]) == [("a", "ignored"), ("b", "ignored")]
    assert len(repository.utterance_marks(bot, limit=2)) == 2
    with pytest.raises(ValueError, match="status"):
        repository.mark_utterances(bot, [("a", "a")], "deleted")
    with pytest.raises(ValueError, match="intent"):
        repository.mark_utterances(bot, [("a", "a")], "added")
    with pytest.raises(ValueError, match="intent"):
        repository.mark_utterances(bot, [("a", "a")], "ignored", "greeting")


@pytest.fixture(params=sorted(STORES))
def store(request):
    return STORES[request.param]()


def test_stores_list_understood_messages_in_a_period_newest_first(store):
    bot = f"q{uuid.uuid4().hex[:8]}"
    for i, text in enumerate(["하나", "둘", "셋"]):
        with store.session(f"{bot}:s{i}") as tracker:
            tracker.append([CommandsReceived(commands=[], nlu={"rule": "fallback", "text": text})])
    with store.session(f"{bot}~test") as tracker:  # a test chat: another prefix
        tracker.append([CommandsReceived(commands=[], nlu={"text": "테스트"})])
    found = store.utterances(f"{bot}:", None, None, 10)
    assert [u.nlu["text"] for u in found] == ["셋", "둘", "하나"]
    assert all(u.at.tzinfo is not None for u in found)
    assert [u.nlu["text"] for u in store.utterances(f"{bot}:", None, None, 2)] == ["셋", "둘"]
    assert store.utterances(f"{bot}:", None, None, 0) == []
    middle = found[1].at
    assert all(u.at >= middle for u in store.utterances(f"{bot}:", middle, None, 10))
    assert all(u.at < middle for u in store.utterances(f"{bot}:", None, middle, 10))
    future = dt.datetime.now(dt.UTC) + dt.timedelta(days=1)
    assert store.utterances(f"{bot}:", future, None, 10) == []


def test_memory_store_period_uses_its_clock():
    times = iter([T0, T0 + dt.timedelta(hours=1), T0 + dt.timedelta(hours=2)])
    store = InMemoryStore(clock=lambda: next(times))
    for i in range(3):
        with store.session(f"b:s{i}") as tracker:
            tracker.append([CommandsReceived(commands=[], nlu={"text": str(i)})])
    window = store.utterances("b:", T0 + dt.timedelta(minutes=30), T0 + dt.timedelta(hours=2), 10)
    assert [u.nlu["text"] for u in window] == ["1"]  # since is inclusive, until exclusive


# ---------------------------------------------------------------- grouping


def u(text, rule="fallback", session="b:1", minutes=0, intent=("weather", 0.4), **extra):
    nlu = {"text": text, "rule": rule, "intents": [list(intent)] if intent else [], **extra}
    return Utterance(session, T0 + dt.timedelta(minutes=minutes), nlu)


def test_group_counts_sentences_sessions_and_keeps_the_latest_details():
    newest_first = [u("양자역학 알려줘!", minutes=9, session="b:2"), u("주식 추천", minutes=8),
                    u(" 양자역학  알려줘", minutes=5, rule="clarify", intent=("help", 0.5)), u("양자역학 알려줘", minutes=1)]
    items = group(newest_first, [])
    assert [(i.text, i.count, i.sessions) for i in items] == [("양자역학 알려줘!", 3, 2), ("주식 추천", 1, 1)]
    first = items[0]
    assert first.key == key("양자역학 알려줘") and first.rule == "fallback" and first.intent == "weather"
    assert first.score == pytest.approx(0.4)
    assert first.first_seen == "2026-11-10T09:01:00+00:00" and first.last_seen == "2026-11-10T09:09:00+00:00"
    assert first.status == "pending" and first.status_intent is None


def test_group_orders_by_count_then_most_recent():
    items = group([u("가", minutes=3), u("나", minutes=2), u("다", minutes=1), u("다", minutes=0)], [])
    assert [i.text for i in items] == ["다", "가", "나"]
    assert [i.text for i in group([u("가"), u("나")], [], limit=1)] == ["가"]
    assert group([], []) == []


def test_group_filters_by_rule_intent_and_status():
    said = [u("날씨 어때", rule="intent", intent=("weather", 0.9)), u("모르는 말"), u("또 모르는 말", intent=("help", 0.2)),
            u("열기", rule="block"), u("", rule="fallback"), Utterance("b:1", T0, {"rule": "welcome"}),
            Utterance("b:1", T0, {"text": "점수 없음", "rule": "fallback", "intents": [["x"]]})]
    assert [i.text for i in group(said, [])] == ["모르는 말", "또 모르는 말", "점수 없음"]
    assert [i.text for i in group(said, [], rule="all")] == ["날씨 어때", "모르는 말", "또 모르는 말", "점수 없음"]
    assert [i.text for i in group(said, [], rule="all", intent="weather")] == ["날씨 어때", "모르는 말"]
    assert group(said, [], intent="nope") == []
    assert group(said, [])[2].intent is None  # a malformed intents entry is ignored
    marks = [Mark(key("모르는 말"), "모르는 말", "added", "help", "t2"), Mark(key("모르는 말"), "모르는 말", "ignored", None, "t1"),
             Mark(key("또 모르는 말"), "또 모르는 말", "ignored", None, "t1")]
    assert [i.text for i in group(said, marks)] == ["점수 없음"]  # the latest mark wins
    added = group(said, marks, status="added")
    assert [(i.text, i.status, i.status_intent) for i in added] == [("모르는 말", "added", "help")]
    assert [i.text for i in group(said, marks, status="ignored")] == ["또 모르는 말"]
    assert len(group(said, marks, status="all")) == 3


# ---------------------------------------------------------------- API


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


def queue(client, **params):
    r = client.get(f"{B}/utterances", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_edits_record_what_each_save_changed(client):
    doc = draft(client)
    doc["label"] = "바뀐 봇"
    doc["intents"]["greeting"]["examples"].append("오랜만이야 친구")
    assert save(client, doc).status_code == 200
    assert save(client, doc).status_code == 200  # the same document again: nothing to record
    items = client.get(f"{B}/edits").json()["items"]
    assert items[0]["revision"] == 2
    assert {c["area"] for c in items[0]["changes"]} == {"bot", "intent"}
    intent = next(c for c in items[0]["changes"] if c["area"] == "intent")
    assert intent["examples_added"] == 1 and intent["examples_removed"] == 0
    assert len(items) == 1  # seeding imports the bot file (not an edit); the repeated save changed nothing
    bot_line = next(c for c in items[0]["changes"] if c["area"] == "bot")
    assert bot_line == {"area": "bot", "id": "sample", "label": "바뀐 봇", "action": "changed",
                        "examples_added": None, "examples_removed": None}
    assert client.get(f"{B}/edits", params={"limit": 0}).status_code == 422
    assert client.get("/v1/builder/bots/nobot/edits").status_code == 404


def test_misunderstood_sentence_added_to_a_block_is_understood_after_deploy(client, host):
    sentence = "양자컴퓨터로 날씨 예측해줘"
    assert say(client, sentence, "u1") == ["fallback"]
    assert say(client, sentence + "!", "u2") == ["fallback"]
    build = finish(client, client.post(f"{B}/test-builds"))["result"]["build"]
    client.post(f"{B}/test/{build}/sessions/t/messages", json={"text": sentence})  # test chats are not user data
    listed = queue(client)
    assert [(i["text"], i["count"], i["sessions"]) for i in listed["items"]] == [(sentence + "!", 2, 2)]
    assert listed["scanned"] >= 2 and listed["truncated"] is False

    tag = client.get(f"{B}/draft").headers["ETag"]
    assert client.post(f"{B}/utterances/add", json={"texts": [sentence], "intent": "weather"}).status_code == 428
    assert client.post(f"{B}/utterances/add", json={"texts": [sentence], "intent": "nope"},
                       headers={"If-Match": tag}).status_code == 404
    taken = client.post(f"{B}/utterances/add", json={"texts": ["도움말"], "intent": "weather"}, headers={"If-Match": tag})
    assert taken.status_code == 409 and taken.json()["texts"] == ["도움말"]
    added = client.post(f"{B}/utterances/add", json={"texts": [sentence], "intent": "weather"}, headers={"If-Match": tag})
    assert added.status_code == 200 and added.json()["valid"] and added.headers["ETag"] != tag
    again = client.post(f"{B}/utterances/add", json={"texts": [sentence], "intent": "weather"}, headers={"If-Match": tag})
    assert again.status_code == 412  # the draft moved on
    assert draft(client)["intents"]["weather"]["examples"][-1] == sentence
    assert queue(client)["items"] == []  # added: out of the pending queue
    assert queue(client, status="added")["items"][0]["status_intent"] == "weather"
    assert client.get(f"{B}/utterances/marks").json()["items"][0] == {
        "key": sentence, "text": sentence, "status": "added", "intent": "weather",
        "marked_at": client.get(f"{B}/utterances/marks").json()["items"][0]["marked_at"]}

    ship(client, draft(client))  # train + deploy the draft with the new example
    assert say(client, sentence, "u3") == ["ask_location"]  # the weather block now starts


def test_ignore_and_restore(client):
    say(client, "아무말 대잔치", "u1")
    assert client.post(f"{B}/utterances/ignore", json={"texts": ["아무말 대잔치"]}).json() == {"bot": "sample", "marked": 1}
    assert queue(client)["items"] == []
    assert [i["text"] for i in queue(client, status="ignored")["items"]] == ["아무말 대잔치"]
    client.post(f"{B}/utterances/restore", json={"texts": ["아무말 대잔치"]})
    assert [i["status"] for i in queue(client)["items"]] == ["pending"]
    assert [m["status"] for m in client.get(f"{B}/utterances/marks").json()["items"]] == ["pending", "ignored"]
    assert client.post(f"{B}/utterances/ignore", json={"texts": []}).status_code == 422
    assert client.post(f"{B}/utterances/ignore", json={"texts": [""]}).status_code == 422
    assert client.post("/v1/builder/bots/nobot/utterances/ignore", json={"texts": ["a"]}).status_code == 404


def test_queue_period_and_filters(client):
    say(client, "날씨 알려줘", "u1")
    say(client, "모르는 얘기", "u1")
    later = (dt.datetime.now(dt.UTC) + dt.timedelta(minutes=1)).isoformat()
    assert queue(client, since=later)["items"] == []
    assert [i["text"] for i in queue(client, until=later)["items"]] == ["모르는 얘기"]
    everything = queue(client, rule="all")["items"]
    assert {i["text"] for i in everything} == {"날씨 알려줘", "모르는 얘기"}
    weather = next(i for i in everything if i["text"] == "날씨 알려줘")
    assert [i["text"] for i in queue(client, rule="all", intent=weather["intent"])["items"]][0] == "날씨 알려줘"
    assert client.get(f"{B}/utterances", params={"since": "2026-11-10T00:00:00"}).status_code == 422  # no time zone
    assert client.get(f"{B}/utterances", params={"status": "weird"}).status_code == 422
    assert client.get("/v1/builder/bots/nobot/utterances").status_code == 404
