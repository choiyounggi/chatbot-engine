"""E-E (second half): creating and deleting bots (docs/engine-core-plan.md "빌더 단계 E-E").

Covers the repository's deletion records and purge, the stores' delete_sessions, and the host/API
create/delete/restore/purge_expired behaviour: empty and copy-from creation, every 409/404/422, the
7-day grace period, permanent purge of every table plus conversations and artifacts, and that a
purge or a session-prefix delete never touches a bot whose id is a prefix of another.
"""

import datetime as dt
import os
import shutil
import uuid

import pytest
from fastapi.testclient import TestClient

from chatcore.api import create_app
from chatcore.bot.loader import read_document
from chatcore.bot.repository import InMemoryRepository
from chatcore.conversations import synthetic_actions
from chatcore.dialogue.events import SessionStarted
from chatcore.host import BotHost
from chatcore.store.memory import InMemoryStore

from .conftest import SAMPLE

B = "/v1/builder/bots/sample"


def new_id() -> str:
    return f"t{uuid.uuid4().hex[:10]}"


class Clock:
    """now() pinned to a mutable instant, like test_blocks.Clock (not shared: a local test helper)."""

    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now


# ---------------------------------------------------------------- repository contract


def postgres_repository():
    url = os.environ.get("CHATCORE_TEST_DATABASE_URL")
    if not url:
        if os.environ.get("CI"):
            pytest.fail("CHATCORE_TEST_DATABASE_URL must be set in CI")
        pytest.skip("set CHATCORE_TEST_DATABASE_URL (make test-db) to run Postgres tests")
    from chatcore.bot.repository import PostgresRepository

    repo = PostgresRepository(url)
    repo.ensure_schema()
    return repo


@pytest.fixture(params=["memory", "postgres"])
def repository(request):
    return InMemoryRepository() if request.param == "memory" else postgres_repository()


def test_repository_deletion_lifecycle(repository):
    bot = new_id()
    repository.save_draft(bot, {"id": bot})
    assert bot not in repository.deletions()
    repository.request_deletion(bot, "2026-01-08T00:00:00+00:00")
    assert repository.deletions()[bot] == "2026-01-08T00:00:00+00:00"
    with pytest.raises(KeyError):
        repository.request_deletion(bot, "2026-01-09T00:00:00+00:00")  # already pending
    assert repository.cancel_deletion(bot) is True
    assert bot not in repository.deletions()
    assert repository.cancel_deletion(bot) is False  # nothing pending: boundary


def test_purge_bot_removes_every_table(repository):
    bot = new_id()
    repository.save_draft(bot, {"id": bot}, changes=[{"area": "bot", "id": bot, "label": "x", "action": "added"}])
    repository.add_version(bot, "aaaaaaaaaaaa", {"id": bot}, 0.9)
    repository.deploy(bot, "aaaaaaaaaaaa")
    repository.mark_utterances(bot, [("k", "텍스트")], "ignored")
    repository.set_secret(bot, "KEY", "value")
    repository.add_skill_error(bot, {"kind": "timeout"})
    kb = {"faq": [], "documents": [], "settings": {"mode": "instant"}}
    repository.save_kb_draft(bot, kb)
    repository.add_kb_version(bot, "bbbbbbbbbbbb", kb, 0, 0)
    repository.request_deletion(bot, "2026-01-01T00:00:00+00:00")

    repository.purge_bot(bot)

    assert repository.draft(bot) is None
    assert repository.versions(bot) == []
    assert repository.deployments(bot) == []
    assert repository.edits(bot) == []
    assert repository.utterance_marks(bot) == []
    assert repository.secret(bot, "KEY") is None
    assert repository.skill_errors(bot) == []
    assert repository.kb_draft(bot) is None
    assert repository.kb_versions(bot) == []
    assert bot not in repository.deletions()
    repository.purge_bot(bot)  # idempotent: purging an already-empty bot is a no-op, not an error


def test_purge_bot_does_not_touch_a_bot_whose_id_is_a_prefix_of_another(repository):
    a = new_id()
    b = a + "_two"
    repository.save_draft(a, {"id": a})
    repository.save_draft(b, {"id": b})
    repository.add_version(a, "aaaaaaaaaaaa", {"id": a}, None)
    repository.add_version(b, "bbbbbbbbbbbb", {"id": b}, None)

    repository.purge_bot(a)

    assert repository.draft(a) is None and repository.versions(a) == []
    assert repository.get_draft(b) == {"id": b}
    assert [v.version for v in repository.versions(b)] == ["bbbbbbbbbbbb"]


# ---------------------------------------------------------------- store: delete_sessions


def postgres_store():
    url = os.environ.get("CHATCORE_TEST_DATABASE_URL")
    if not url:
        if os.environ.get("CI"):
            pytest.fail("CHATCORE_TEST_DATABASE_URL must be set in CI")
        pytest.skip("set CHATCORE_TEST_DATABASE_URL (make test-db) to run Postgres tests")
    from chatcore.store.postgres import PostgresStore

    store = PostgresStore(url)
    store.ensure_schema()
    return store


@pytest.fixture(params=["memory", "postgres"])
def store(request):
    return InMemoryStore() if request.param == "memory" else postgres_store()


def test_delete_sessions_matches_only_its_literal_prefix(store):
    bot = new_id()
    other = bot + "_two"  # a prefix-colliding bot: LIKE patterns must not match it
    for sid, who in ((f"{bot}:a", bot), (f"{bot}:b", bot), (f"{bot}~test", bot), (f"{other}:a", other)):
        with store.session(sid) as t:
            t.append([SessionStarted(bot_id=who, bot_version="v1", kb_version=None)])

    assert store.delete_sessions(f"{bot}:") == 2
    assert store.events(f"{bot}:a") == [] and store.events(f"{bot}:b") == []
    assert store.events(f"{bot}~test") != []  # different separator, a different prefix
    assert store.events(f"{other}:a") != []  # prefix-colliding bot untouched

    assert store.delete_sessions(f"{bot}~") == 1
    assert store.events(f"{bot}~test") == []
    assert store.delete_sessions(f"{new_id()}:") == 0  # boundary: nothing matches


# ---------------------------------------------------------------- host: create / delete / restore / purge


@pytest.fixture
def host(fake_embedder, tmp_path, trained_sample):
    shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
    h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions, keep=1,
               min_f1=0.0, clock=Clock(dt.datetime(2026, 1, 1, tzinfo=dt.UTC)))
    h.seed(SAMPLE)
    return h


@pytest.fixture
def client(host):
    return TestClient(create_app(host))


def test_create_empty_bot_is_valid_and_not_served(client, host):
    bot = new_id()
    resp = client.post("/v1/builder/bots", json={"id": bot, "label": "새 봇"})
    assert resp.status_code == 201 and resp.json() == {"bot": bot, "revision": 1}
    check = host.check_draft(host.repository.get_draft(bot))
    assert check.errors == [] and check.warnings == []
    assert host.runtime(bot) is None  # saved as a draft, never deployed
    bots = {b["id"]: b for b in client.get("/v1/builder/bots").json()["bots"]}
    assert bots[bot]["live"] is None and bots[bot]["draft_revision"] == 1 and bots[bot]["deletion"] is None


def test_create_by_copy_copies_draft_and_kb_draft_not_secrets_or_versions(client, host):
    src = new_id()
    client.post("/v1/builder/bots", json={"id": src, "label": "원본"})
    kb = {"faq": [{"id": "f1", "question": "영업시간", "answer": "9-6"}], "documents": [], "settings": {"mode": "instant"}}
    host.repository.save_kb_draft(src, kb)
    host.repository.set_secret(src, "KEY", "value")
    host.repository.add_version(src, "aaaaaaaaaaaa", host.repository.get_draft(src), None)

    dst = new_id()
    resp = client.post("/v1/builder/bots", json={"id": dst, "label": "복사본", "copy_from": src})
    assert resp.status_code == 201

    copied = host.repository.get_draft(dst)
    source = host.repository.get_draft(src)
    assert copied["id"] == dst and copied["label"] == "복사본"
    assert {k: v for k, v in copied.items() if k not in ("id", "label")} == \
           {k: v for k, v in source.items() if k not in ("id", "label")}
    assert host.repository.kb_draft(dst).document == kb
    assert host.repository.secret(dst, "KEY") is None
    assert host.repository.versions(dst) == []


def test_create_409_when_id_already_has_a_draft_or_versions_or_is_waiting_for_deletion(client, host):
    bot = new_id()
    client.post("/v1/builder/bots", json={"id": bot, "label": "a"})
    assert client.post("/v1/builder/bots", json={"id": bot, "label": "b"}).status_code == 409  # has a draft

    versions_only = new_id()
    host.repository.add_version(versions_only, "aaaaaaaaaaaa", {"id": versions_only}, None)
    assert client.post("/v1/builder/bots", json={"id": versions_only, "label": "x"}).status_code == 409

    client.delete(f"{B}")
    assert client.post("/v1/builder/bots", json={"id": "sample", "label": "c"}).status_code == 409


def test_create_404_when_copy_source_unknown_has_no_draft_or_is_deleting(client, host):
    assert client.post(
        "/v1/builder/bots", json={"id": new_id(), "label": "a", "copy_from": "nope"}
    ).status_code == 404

    versions_only = new_id()
    host.repository.add_version(versions_only, "aaaaaaaaaaaa", {"id": versions_only}, None)
    assert client.post(
        "/v1/builder/bots", json={"id": new_id(), "label": "a", "copy_from": versions_only}
    ).status_code == 404

    client.delete(f"{B}")
    assert client.post(
        "/v1/builder/bots", json={"id": new_id(), "label": "a", "copy_from": "sample"}
    ).status_code == 404


def test_create_422_for_a_bad_id_or_label(client):
    assert client.post("/v1/builder/bots", json={"id": "Bad-ID", "label": "x"}).status_code == 422  # pattern
    assert client.post("/v1/builder/bots", json={"id": new_id(), "label": ""}).status_code == 422  # boundary: empty
    assert client.post("/v1/builder/bots", json={"id": new_id(), "label": "x" * 101}).status_code == 422


def test_delete_takes_the_bot_out_of_service_and_drops_test_builds(client, host):
    assert client.post("/v1/bots/sample/sessions/x/messages", json={"text": "안녕"}).status_code == 200
    build = None
    started = client.post(f"{B}/test-builds")
    for _ in range(400):
        job = client.get(f"/v1/builder/jobs/{started.json()['id']}").json()
        if job["status"] == "done":
            build = job["result"]["build"]
            break
    assert build is not None and host.test_runtime("sample", build) is not None

    resp = client.delete(f"{B}")
    assert resp.status_code == 202
    purge_after = resp.json()["purge_after"]
    assert resp.json()["bot"] == "sample" and purge_after

    assert client.post("/v1/bots/sample/sessions/x/messages", json={"text": "안녕"}).status_code == 404
    assert client.post("/v1/bots/sample/sessions/y/open").status_code == 404
    assert host.test_runtime("sample", build) is None  # dropped immediately, not just expired

    bots = {b["id"]: b for b in client.get("/v1/builder/bots").json()["bots"]}
    assert bots["sample"]["deletion"] == {"purge_after": purge_after}

    assert client.delete(f"{B}").status_code == 409  # twice
    assert client.delete("/v1/builder/bots/nope").status_code == 404  # unknown bot


def test_builder_routes_409_while_deleting_except_list_and_restore(client):
    client.delete(f"{B}")
    for method, path, body in (
        ("get", f"{B}/draft", None),
        ("get", f"{B}/edits", None),
        ("get", f"{B}/utterances", None),
        ("get", f"{B}/secrets", None),
        ("get", f"{B}/versions", None),
        ("get", f"{B}/misunderstood", None),
        ("post", f"{B}/test-builds", None),
        ("post", f"{B}/versions", None),
        ("post", f"{B}/deploy", {"version": "0" * 12}),
    ):
        resp = client.request(method, path, json=body)
        assert resp.status_code == 409, (path, resp.status_code, resp.text)
        assert "waiting to be deleted" in resp.json()["detail"]

    assert client.get("/v1/builder/bots").status_code == 200  # the list itself stays open
    assert client.post(f"{B}/restore").status_code == 200  # and restore


def test_restore_serves_the_bot_again(client, host):
    client.delete(f"{B}")
    resp = client.post(f"{B}/restore")
    assert resp.status_code == 200
    assert resp.json()["bot"] == "sample" and resp.json()["live"]
    assert client.post("/v1/bots/sample/sessions/x/messages", json={"text": "안녕"}).status_code == 200
    bots = {b["id"]: b for b in client.get("/v1/builder/bots").json()["bots"]}
    assert bots["sample"]["deletion"] is None

    assert client.post(f"{B}/restore").status_code == 404  # nothing pending now
    assert client.post("/v1/builder/bots/nope/restore").status_code == 404  # never existed


def test_seed_does_not_resurrect_a_bot_waiting_for_deletion(host):
    host.delete_bot("sample")
    assert host.seed(SAMPLE) == "sample"
    assert host.runtime("sample") is None
    assert "sample" in host.repository.deletions()


def test_admin_token_guards_the_new_routes(host):
    secured = TestClient(create_app(host, admin_token="s3cret"))
    bot = new_id()
    assert secured.post("/v1/builder/bots", json={"id": bot, "label": "x"}).status_code == 401
    assert secured.delete(f"{B}").status_code == 401
    assert secured.post(f"{B}/restore").status_code == 401
    headers = {"Authorization": "Bearer s3cret"}
    assert secured.post("/v1/builder/bots", json={"id": bot, "label": "x"}, headers=headers).status_code == 201


# ---------------------------------------------------------------- purge: start(), purge_expired(), prefix safety


def test_purge_at_host_start_does_not_load_a_pending_bot(fake_embedder, tmp_path, trained_sample):
    shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
    repo, store = InMemoryRepository(), InMemoryStore()
    clock = Clock(dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    first = BotHost(repo, store, fake_embedder, tmp_path, actions=synthetic_actions, keep=1, min_f1=0.0, clock=clock)
    first.seed(SAMPLE)
    first.delete_bot("sample")

    clock.now += dt.timedelta(days=8)
    restarted = BotHost(repo, store, fake_embedder, tmp_path, actions=synthetic_actions, keep=1, min_f1=0.0,
                        clock=clock)
    restarted.start()

    assert restarted.bots() == {}
    assert repo.draft("sample") is None  # purged at start, not merely skipped
    assert not (tmp_path / "sample").exists()


def test_purge_expired_removes_every_row_conversations_and_artifacts(fake_embedder, tmp_path):
    repo, store = InMemoryRepository(), InMemoryStore()
    clock = Clock(dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    host = BotHost(repo, store, fake_embedder, tmp_path, actions=synthetic_actions, keep=1, min_f1=0.0, clock=clock)
    bot = new_id()
    doc = {**read_document(SAMPLE), "id": bot}
    host.save_draft(bot, doc)
    version, _ = host.train_draft(bot)
    host.deploy(bot, version.version)
    assert (tmp_path / bot).exists()

    repo.mark_utterances(bot, [("k", "텍스트")], "ignored")
    repo.set_secret(bot, "KEY", "value")
    repo.add_skill_error(bot, {"kind": "timeout"})
    kb = {"faq": [], "documents": [], "settings": {"mode": "instant"}}
    repo.save_kb_draft(bot, kb)
    repo.add_kb_version(bot, "bbbbbbbbbbbb", kb, 0, 0)
    with store.session(f"{bot}:conv") as t:
        t.append([SessionStarted(bot_id=bot, bot_version=version.version, kb_version=None)])
    with store.session(f"{bot}~test") as t:
        t.append([SessionStarted(bot_id=bot, bot_version=version.version, kb_version=None)])
    build = host.build_test(bot)
    assert host.test_runtime(bot, build) is not None

    host.delete_bot(bot)
    assert host.test_runtime(bot, build) is None  # test builds dropped right away

    clock.now += dt.timedelta(days=8)
    assert host.purge_expired() == [bot]

    assert repo.draft(bot) is None
    assert repo.versions(bot) == []
    assert repo.deployments(bot) == []
    assert repo.utterance_marks(bot) == []
    assert repo.secret(bot, "KEY") is None
    assert repo.skill_errors(bot) == []
    assert repo.kb_draft(bot) is None
    assert repo.kb_versions(bot) == []
    assert bot not in repo.deletions()
    assert store.events(f"{bot}:conv") == []
    assert store.events(f"{bot}~test") == []
    assert not (tmp_path / bot).exists()
    assert host.purge_expired() == []  # boundary: nothing left to purge


def test_purge_expired_leaves_a_prefix_colliding_bot_untouched(fake_embedder, tmp_path):
    repo, store = InMemoryRepository(), InMemoryStore()
    clock = Clock(dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
    host = BotHost(repo, store, fake_embedder, tmp_path, actions=synthetic_actions, keep=1, min_f1=0.0, clock=clock)
    a = new_id()
    b = a + "_two"
    versions = {}
    for bot in (a, b):
        doc = {**read_document(SAMPLE), "id": bot}
        host.save_draft(bot, doc)
        version, _ = host.train_draft(bot)
        host.deploy(bot, version.version)
        versions[bot] = version.version
        with store.session(f"{bot}:s") as t:
            t.append([SessionStarted(bot_id=bot, bot_version=version.version, kb_version=None)])

    host.delete_bot(a)
    clock.now += dt.timedelta(days=8)
    assert host.purge_expired() == [a]

    assert repo.draft(a) is None and repo.versions(a) == []
    assert not (tmp_path / a).exists()
    assert store.events(f"{a}:s") == []

    assert repo.draft(b) is not None
    assert [v.version for v in repo.versions(b)] == [versions[b]]
    assert (tmp_path / b).exists()
    assert store.events(f"{b}:s") != []
    assert host.runtime(b) is not None
