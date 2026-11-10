"""Phase 6: builder workflow over HTTP — drafts, training, deploy, rollback, version pinning.

Intent checks use exact training sentences (matched_by == "exact") so they do
not depend on the fake embedder's accuracy.
"""

import copy
import os
import shutil
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from chatcore.api import create_app
from chatcore.bot.loader import document_version, read_document
from chatcore.bot.repository import InMemoryRepository, VersionNotFound
from chatcore.conversations import synthetic_actions
from chatcore.dialogue.state import fold
from chatcore.host import BotHost
from chatcore.store.memory import InMemoryStore

from .conftest import SAMPLE

B = "/v1/builder/bots/sample"


@pytest.fixture(scope="session")
def trained_sample(fake_embedder, tmp_path_factory):
    """The sample bot trained once; each test gets a copy, so seeding reuses it instead of training."""
    root = tmp_path_factory.mktemp("trained")
    BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, root, actions=synthetic_actions, min_f1=0.0).seed(SAMPLE)
    return root


@pytest.fixture
def host(fake_embedder, tmp_path, trained_sample):
    shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
    h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions, keep=1, min_f1=0.0)
    h.seed(SAMPLE)
    return h


@pytest.fixture
def client(host):
    return TestClient(create_app(host))


def say(client, text, session):
    r = client.post(f"/v1/bots/sample/sessions/{session}/messages", json={"text": text})
    assert r.status_code == 200, r.text
    return [m["response"] for m in r.json()["messages"]]


def parse(client, text):
    return client.post("/v1/bots/sample/nlu/parse", json={"text": text}).json()


def ship(client, document):
    """Save a draft, train it, deploy it; returns the new version."""
    saved = client.put(f"{B}/draft", json=document)
    assert saved.status_code == 200 and saved.json()["valid"], saved.json()
    trained = client.post(f"{B}/versions")
    assert trained.status_code == 200, trained.text
    version = trained.json()["version"]
    assert client.post(f"{B}/deploy", json={"version": version}).json()["live"] == version
    return version


def draft(client):
    return copy.deepcopy(client.get(f"{B}/draft").json())


# ---------------------------------------------------------------- plan done criteria


def test_add_example_train_deploy_then_rollback(client):
    sentence = "땡큐 베리 머치 챗봇"
    before = parse(client, sentence)
    assert before["matched_by"] == "classifier"
    doc = draft(client)
    doc["intents"]["thanks"]["examples"].append(sentence)
    first_live = client.get(f"{B}/versions").json()["live"]
    ship(client, doc)
    after = parse(client, sentence)
    assert after["matched_by"] == "exact" and after["intents"][0][0] == "thanks"
    assert client.post(f"{B}/rollback").json()["live"] == first_live
    assert parse(client, sentence)["matched_by"] == "classifier"  # back to the old behaviour


def test_running_session_finishes_on_its_version_after_its_scenario_is_deleted(client):
    assert say(client, "날씨 어때요?", "old") == ["ask_location"]  # exact weather example, on v1
    doc = draft(client)
    del doc["scenarios"]["check_weather"]
    del doc["responses"]["weather_report"]
    ship(client, doc)
    assert say(client, "부산", "old") == ["weather_report"]  # still served by v1
    assert say(client, "날씨 어때요?", "new") == ["fallback"]  # v2 has no weather scenario


def test_session_whose_version_was_dropped_restarts_with_a_notice(client):
    say(client, "날씨 어때요?", "s")  # v1, waiting for a location
    doc = draft(client)
    del doc["scenarios"]["check_weather"]
    del doc["responses"]["weather_report"]
    ship(client, doc)  # v2 (v1 still loaded: keep=1)
    doc["label"] = "세 번째"
    ship(client, doc)  # v3: v1 is dropped
    responses = say(client, "부산", "s")
    assert responses[0] == "restarted"


def test_renaming_labels_keeps_sessions_and_events_working(client, host):
    say(client, "날씨 어때요?", "s")
    doc = draft(client)
    doc["intents"]["weather"]["label"] = "날씨"
    doc["scenarios"]["check_weather"]["label"] = "날씨 알려주기"
    ship(client, doc)
    assert say(client, "부산", "s") == ["weather_report"]  # same session, scenario continues
    state = fold(host.store.events("sample:s"))  # the log still replays
    assert state.stack == [] and state.completed[-1].scenario == "check_weather"
    assert say(client, "날씨 어때요?", "fresh") == ["ask_location"]  # new sessions use the renamed version


# ---------------------------------------------------------------- builder API behaviour


def test_invalid_draft_is_saved_with_its_problems(client):
    doc = draft(client)
    doc["scenarios"]["check_weather"]["steps"].append({"respond": "nope"})
    body = client.put(f"{B}/draft", json=doc).json()
    assert body["valid"] is False and any("nope" in e for e in body["errors"])
    assert client.get(f"{B}/draft").json() == doc  # work in progress is kept
    trained = client.post(f"{B}/versions")
    assert trained.status_code == 422 and trained.json()["errors"]


def test_draft_id_must_match_the_url(client):
    doc = draft(client)
    doc["id"] = "other"
    assert client.put(f"{B}/draft", json=doc).status_code == 422


def test_training_below_the_bar_is_rejected(client, host):
    host.min_f1 = 1.01
    response = client.post(f"{B}/versions")
    assert response.status_code == 422 and "below" in response.json()["detail"]


def test_versions_and_deploy_errors(client, host):
    versions = client.get(f"{B}/versions").json()
    assert versions["live"] == versions["versions"][0]["version"] and versions["versions"][0]["live"]
    assert client.post(f"{B}/deploy", json={"version": "0" * 12}).status_code == 404
    assert client.post(f"{B}/deploy", json={"version": "not-a-hash"}).status_code == 422
    doc = draft(client)
    doc["label"] = "학습 안 함"
    host.repository.add_version("sample", document_version(doc), doc, None)  # stored but never trained
    assert client.post(f"{B}/deploy", json={"version": document_version(doc)}).status_code == 409
    assert client.post(f"{B}/rollback").status_code == 409  # only one deployment so far
    assert client.get("/v1/builder/bots/nope/versions").status_code == 404


def test_misunderstood_lists_fallbacks_with_their_text(client):
    say(client, "양자역학 설명해줘 제발", "m")
    items = client.get(f"{B}/misunderstood").json()["items"]
    assert items and items[0]["text"] == "양자역학 설명해줘 제발" and items[0]["rule"] in ("fallback", "clarify")


def test_admin_token_guards_builder_endpoints(host):
    client = TestClient(create_app(host, admin_token="s3cret"))
    assert client.get(f"{B}/draft").status_code == 401
    assert client.get(f"{B}/draft", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get(f"{B}/misunderstood").status_code == 401
    assert client.post("/v1/bots/sample/sessions/x/messages", json={"text": "안녕"}).status_code == 200  # chat stays open


def test_host_restart_loads_the_deployed_version(host, fake_embedder, tmp_path):
    again = BotHost(host.repository, host.store, fake_embedder, tmp_path, actions=synthetic_actions, keep=1, min_f1=0.0)
    assert again.bots() == {}
    again.start()
    assert again.bots() == host.bots()
    assert again.seed(SAMPLE) == "sample" and again.bots() == host.bots()  # seeding twice changes nothing


def test_training_errors_are_reported_not_crashes(client):
    doc = draft(client)
    doc["intents"]["tiny"] = {"examples": ["가나다라 하나", "가나다라 둘", "가나다라 셋"]}
    assert client.put(f"{B}/draft", json=doc).json()["valid"]  # few examples is only a warning
    response = client.post(f"{B}/versions")
    assert response.status_code == 422 and "at least" in response.json()["detail"]


def test_deploying_a_stored_version_that_no_longer_runs_is_409(client, host):
    doc = draft(client)
    doc["scenarios"]["check_weather"]["steps"].append({"respond": "nope"})
    host.repository.add_version("sample", document_version(doc), doc, None)
    response = client.post(f"{B}/deploy", json={"version": document_version(doc)})
    assert response.status_code == 409 and any("nope" in e for e in response.json()["detail"]["errors"])


def test_rollback_to_a_version_whose_model_files_are_gone_is_409(client, host, tmp_path):
    first = host.bots()["sample"]
    doc = draft(client)
    doc["label"] = "둘째"
    ship(client, doc)
    shutil.rmtree(tmp_path / "sample" / first)
    assert client.post(f"{B}/rollback").status_code == 409
    assert host.bots()["sample"] != first and host.repository.deployments("sample")[0] != first  # nothing changed


def test_concurrent_deploys_leave_memory_and_repository_agreeing(client, host, monkeypatch):
    v1 = host.bots()["sample"]
    doc = draft(client)
    doc["label"] = "둘째"
    client.put(f"{B}/draft", json=doc)
    v2 = client.post(f"{B}/versions").json()["version"]
    inside, release, calls = threading.Event(), threading.Event(), []
    real = host.repository.deploy

    def slow(bot_id, version):
        calls.append(version)
        if version == v2:
            inside.set()
            release.wait(5)
        real(bot_id, version)

    monkeypatch.setattr(host.repository, "deploy", slow)
    first = threading.Thread(target=host.deploy, args=("sample", v2))
    first.start()
    assert inside.wait(5)
    second = threading.Thread(target=host.deploy, args=("sample", v1))
    second.start()
    second.join(2)  # without the lock it would finish (its deploy does not wait) well within this
    assert second.is_alive() and calls == [v2]  # the second deploy waits for the first to finish
    release.set()
    first.join(5)
    second.join(5)
    assert host.bots()["sample"] == host.repository.deployments("sample")[0] == v1


def test_start_retrains_missing_model_files_and_skips_bots_that_cannot_load(host, fake_embedder, tmp_path):
    broken = {**read_document(SAMPLE), "id": "broken"}
    broken["scenarios"]["check_weather"]["steps"].append({"respond": "nope"})
    host.repository.add_version("broken", "bbbbbbbbbbbb", broken, None)
    host.repository.deploy("broken", "bbbbbbbbbbbb")
    elsewhere = tmp_path / "new-machine"  # same database, empty artifacts/
    again = BotHost(host.repository, host.store, fake_embedder, elsewhere, actions=synthetic_actions, min_f1=0.0)
    again.start()
    assert again.bots() == {"sample": host.bots()["sample"]}
    assert (elsewhere / "sample" / host.bots()["sample"] / "meta.json").is_file()


def test_admin_token_rejects_non_ascii_headers(host):
    client = TestClient(create_app(host, admin_token="s3cret"))
    assert client.get(f"{B}/draft", headers={"Authorization": "Bearer sécret".encode("latin-1")}).status_code == 401


def test_errors_while_picking_a_version_are_500_without_details(client, host, monkeypatch):
    def bug(bot_id, key):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(host, "runtime_for_session", bug)
    response = client.post("/v1/bots/sample/sessions/x/messages", json={"text": "안녕"})
    assert response.status_code == 500 and "secret" not in response.text


def test_misunderstood_respects_the_limit_newest_first(client):
    say(client, "양자역학 설명해줘 제발", "m1")
    say(client, "주식 종목 추천해줘 제발", "m2")
    items = client.get(f"{B}/misunderstood", params={"limit": 1}).json()["items"]
    assert [i["session"] for i in items] == ["m2"]


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


def test_repository_contract(repository):
    bot = f"t_{uuid.uuid4().hex[:8]}"
    doc = {**read_document(SAMPLE), "id": bot}
    assert repository.get_draft(bot) is None and repository.deployments(bot) == []
    repository.save_draft(bot, doc)
    assert repository.get_draft(bot) == doc and bot in repository.bots()
    a = repository.add_version(bot, "aaaaaaaaaaaa", doc, 0.9)
    again = repository.add_version(bot, "aaaaaaaaaaaa", {**doc, "label": "ignored"}, 0.1)  # idempotent
    assert again == a and repository.get_version(bot, "aaaaaaaaaaaa") == doc
    repository.add_version(bot, "bbbbbbbbbbbb", doc, None)
    assert [v.version for v in repository.versions(bot)] == ["bbbbbbbbbbbb", "aaaaaaaaaaaa"]
    repository.deploy(bot, "aaaaaaaaaaaa")
    repository.deploy(bot, "bbbbbbbbbbbb")
    assert repository.deployments(bot) == ["bbbbbbbbbbbb", "aaaaaaaaaaaa"]
    with pytest.raises(VersionNotFound):
        repository.deploy(bot, "cccccccccccc")
    with pytest.raises(VersionNotFound):
        repository.get_version(bot, "cccccccccccc")


def test_seeding_reuses_a_model_that_is_already_trained(host, fake_embedder, tmp_path, monkeypatch):
    fresh = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions, min_f1=0.0)

    def no_training(bot_id):
        raise AssertionError("retrained a model that was already in artifacts/")

    monkeypatch.setattr(fresh, "train_draft", no_training)
    loads = []
    real_load = fresh._load
    monkeypatch.setattr(fresh, "_load", lambda bot: loads.append(bot.version) or real_load(bot))
    assert fresh.seed(SAMPLE) == "sample" and fresh.bots() == host.bots()
    assert loads == [host.bots()["sample"]]  # loaded once, by the deploy (a load builds a Kiwi: ~0.3 s)
    (info,) = fresh.repository.versions("sample")
    assert info.macro_f1 == host.repository.versions("sample")[0].macro_f1  # taken from meta.json
