"""Phase 6: builder workflow over HTTP — drafts, training, deploy, rollback, version pinning.

Intent checks use exact training sentences (matched_by == "exact") so they do
not depend on the fake embedder's accuracy.
"""

import copy
import json
import os
import shutil
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from chatcore.api import create_app
from chatcore.bot.loader import document_version, read_document
from chatcore.bot.repository import InMemoryRepository, RevisionConflict, VersionNotFound
from chatcore.conversations import synthetic_actions
from chatcore.dialogue.state import fold
from chatcore.host import BotHost
from chatcore.nlu.train import TrainingError
from chatcore.store.base import SessionBusy
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


def save(client, document):
    """PUT the draft over the current one (with its ETag, as the builder does)."""
    tag = client.get(f"{B}/draft").headers["ETag"]
    return client.put(f"{B}/draft", json=document, headers={"If-Match": tag})


def finish(client, started):
    """Wait for a job started by `started` (a 202 response) and return its final state."""
    assert started.status_code == 202, started.text
    job = started.json()
    for _ in range(800):
        job = client.get(f"/v1/builder/jobs/{job['id']}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.025)
    raise AssertionError(f"job did not finish: {job}")


def train(client):
    return finish(client, client.post(f"{B}/versions"))


def ship(client, document, note=""):
    """Save a draft, train it, deploy it; returns the new version."""
    saved = save(client, document)
    assert saved.status_code == 200 and saved.json()["valid"], saved.json()
    job = train(client)
    assert job["status"] == "done", job
    version = job["result"]["version"]
    assert client.post(f"{B}/deploy", json={"version": version, "note": note}).json()["live"] == version
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
    body = save(client, doc).json()
    assert body["valid"] is False and any("nope" in e for e in body["errors"])
    assert client.get(f"{B}/draft").json() == doc  # work in progress is kept
    job = train(client)
    assert job["status"] == "failed" and job["error"]["kind"] == "invalid_draft" and job["error"]["errors"]


def test_draft_id_must_match_the_url(client):
    doc = draft(client)
    doc["id"] = "other"
    assert save(client, doc).status_code == 422


def test_training_below_the_bar_is_rejected(client, host):
    host.min_f1 = 1.01
    job = train(client)
    assert job["status"] == "failed" and job["error"]["kind"] == "below_bar" and "below" in job["error"]["detail"]


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
    assert save(client, doc).json()["valid"]  # few examples is only a warning
    job = train(client)
    assert job["status"] == "failed" and job["error"]["kind"] == "training" and "at least" in job["error"]["detail"]


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
    save(client, doc)
    v2 = train(client)["result"]["version"]
    inside, release, calls = threading.Event(), threading.Event(), []
    real = host.repository.deploy

    def slow(bot_id, version, note=""):
        calls.append(version)
        if version == v2:
            inside.set()
            release.wait(5)
        real(bot_id, version, note)

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


def test_repository_revisions_numbers_and_history(repository):
    bot = f"t_{uuid.uuid4().hex[:8]}"
    doc = {**read_document(SAMPLE), "id": bot}
    assert repository.draft(bot) is None
    assert repository.save_draft(bot, doc, expected_revision=0) == 1  # create
    with pytest.raises(RevisionConflict) as exists:
        repository.save_draft(bot, doc, expected_revision=0)
    assert exists.value.current == 1
    assert repository.save_draft(bot, {**doc, "label": "둘"}, expected_revision=1) == 2
    with pytest.raises(RevisionConflict) as stale:
        repository.save_draft(bot, {**doc, "label": "셋"}, expected_revision=1)  # the other tab
    assert stale.value.current == 2 and repository.draft(bot).document["label"] == "둘"
    assert repository.save_draft(bot, doc) == 3  # no expectation: plain overwrite (seeding)
    first = repository.add_version(bot, "aaaaaaaaaaaa", doc, 0.9)
    second = repository.add_version(bot, "bbbbbbbbbbbb", doc, None)
    assert (first.number, second.number) == (1, 2)
    assert repository.add_version(bot, "aaaaaaaaaaaa", doc, 0.1).number == 1  # idempotent keeps its number
    repository.deploy(bot, "aaaaaaaaaaaa")
    repository.deploy(bot, "bbbbbbbbbbbb", "예문 추가")
    history = repository.history(bot)
    assert [(d.version, d.number, d.note) for d in history] == [("bbbbbbbbbbbb", 2, "예문 추가"), ("aaaaaaaaaaaa", 1, "")]
    assert all(d.deployed_at for d in history)


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


# ---------------------------------------------------------------- E-A: ETag drafts, jobs, test builds, history


def test_draft_saves_need_the_current_etag(client):
    first = client.get(f"{B}/draft")
    tag, doc = first.headers["ETag"], copy.deepcopy(first.json())
    assert client.put(f"{B}/draft", json=doc).status_code == 428  # no If-Match over an existing draft
    ok = client.put(f"{B}/draft", json=doc, headers={"If-Match": tag})
    assert ok.status_code == 200 and ok.json()["revision"] == int(tag.strip('"')) + 1
    assert ok.headers["ETag"] == f'"{ok.json()["revision"]}"'
    stale = client.put(f"{B}/draft", json=doc, headers={"If-Match": tag})  # the second tab
    assert stale.status_code == 412 and stale.json()["revision"] == ok.json()["revision"]
    assert client.put(f"{B}/draft", json=doc, headers={"If-Match": "W/" + ok.headers["ETag"]}).status_code == 200
    assert client.put(f"{B}/draft", json=doc, headers={"If-Match": "nonsense"}).status_code == 400


def test_a_new_bot_draft_needs_no_etag_and_unknown_drafts_are_404(client):
    created = client.put("/v1/builder/bots/fresh/draft", json={**draft(client), "id": "fresh"})
    assert created.status_code == 200 and created.json()["revision"] == 1
    assert client.get("/v1/builder/bots/nobody/draft").status_code == 404


def coupon_block(doc, examples=("쿠폰 어디서 받아요",)):
    doc["intents"]["coupon"] = {"examples": list(examples)}
    doc["responses"]["coupon_info"] = {"text": "쿠폰함에서 받아요"}
    doc["scenarios"]["coupon"] = {"trigger": {"intent": "coupon"}, "steps": [{"respond": "coupon_info"}]}
    return doc


def test_test_build_answers_a_new_one_example_block_before_any_deploy(client):
    assert save(client, coupon_block(draft(client))).json()["valid"]
    assert train(client)["error"]["kind"] == "training"  # too few examples for a deployable version
    job = finish(client, client.post(f"{B}/test-builds"))
    assert job["status"] == "done", job
    build = job["result"]["build"]
    r = client.post(f"{B}/test/{build}/sessions/t1/messages", json={"text": "쿠폰 어디서 받아요"})
    assert r.status_code == 200 and [m["response"] for m in r.json()["messages"]] == ["coupon_info"]
    assert r.json()["nlu"]["rule"] and r.json()["stack"] == []
    assert say(client, "쿠폰 어디서 받아요", "user") != ["coupon_info"]  # users still talk to the deployed bot
    client.post(f"{B}/test/{build}/sessions/t1/messages", json={"text": "양자역학 설명해줘 제발"})
    items = client.get(f"{B}/misunderstood").json()["items"]
    assert [i["session"] for i in items] == ["user"]  # the user's miss is listed; test chats are not user data


def test_test_chat_shows_the_scenario_stack_and_actions(client):
    build = finish(client, client.post(f"{B}/test-builds"))["result"]["build"]
    url = f"{B}/test/{build}/sessions/t2/messages"
    asked = client.post(url, json={"text": "날씨 어때요?"}).json()
    assert [(f["scenario"], f["awaiting"]) for f in asked["stack"]] == [("check_weather", "location")]
    answered = client.post(url, json={"text": "부산"}).json()
    assert answered["actions"] == [{"scenario": "check_weather", "action": "get_weather", "ok": True}]
    assert answered["stack"] == [] and [m["response"] for m in answered["messages"]] == ["weather_report"]


def test_test_build_failures_are_reported(client, host, monkeypatch):
    doc = draft(client)
    doc["scenarios"]["check_weather"]["steps"].append({"respond": "nope"})
    save(client, doc)
    assert finish(client, client.post(f"{B}/test-builds"))["error"]["kind"] == "invalid_draft"

    def cannot_fit(bot_id):
        raise TrainingError("need at least two intents")

    monkeypatch.setattr(host, "build_test", cannot_fit)
    failed = finish(client, client.post(f"{B}/test-builds"))
    assert failed["error"] == {"kind": "training", "detail": "need at least two intents"}


def test_training_a_bot_without_a_draft_is_a_reported_failure(client, host):
    doc = {**read_document(SAMPLE), "id": "nodraft"}
    host.repository.add_version("nodraft", "cccccccccccc", doc, None)  # versions only (e.g. imported)
    for path in ("versions", "test-builds"):
        job = finish(client, client.post(f"/v1/builder/bots/nodraft/{path}"))
        assert job["error"]["kind"] == "no_draft", job


def test_test_builds_expire_are_capped_and_never_fall_back_to_live(client, host, tmp_path):
    assert client.post(f"{B}/test/{'0' * 12}/sessions/x/messages", json={"text": "안녕"}).status_code == 410
    builds = []
    for label in ("하나", "둘", "셋"):
        doc = draft(client)
        doc["label"] = label
        save(client, doc)
        builds.append(finish(client, client.post(f"{B}/test-builds"))["result"]["build"])
    def build_dirs():
        return sorted(p.name for p in (tmp_path / "_test").iterdir())

    kept = {host._tests[("sample", b)][1].name for b in builds[1:]}
    assert len(set(builds)) == 3 and host.test_runtime("sample", builds[0]) is None  # test_keep=2: oldest went
    assert build_dirs() == sorted(kept)  # its files went with it
    host.test_ttl = 0.01
    time.sleep(0.05)
    assert client.post(f"{B}/test/{builds[2]}/sessions/x/messages", json={"text": "안녕"}).status_code == 410
    assert build_dirs() == []


def test_test_builds_do_not_push_out_versions_that_sessions_are_pinned_to(client, host):
    assert say(client, "날씨 어때요?", "pinned") == ["ask_location"]  # on v1
    doc = draft(client)
    doc["label"] = "둘째"
    ship(client, doc)  # keep=1: v1 and v2 stay loaded
    for label in ("t1", "t2", "t3"):
        doc["label"] = label
        save(client, doc)
        finish(client, client.post(f"{B}/test-builds"))
    v1 = host.repository.versions("sample")[-1].version
    assert host.runtime_for_session("sample", "sample:pinned").bot.version == v1  # still loaded and pinned
    assert say(client, "부산", "pinned") == ["weather_report"]


def test_versions_have_numbers_and_the_history_keeps_notes(client):
    doc = draft(client)
    doc["label"] = "둘째"
    ship(client, doc, note="라벨 변경")
    client.post(f"{B}/rollback")
    body = client.get(f"{B}/versions").json()
    assert [v["number"] for v in body["versions"]] == [2, 1]
    assert [(h["number"], h["note"]) for h in body["history"]] == [(1, "rollback"), (2, "라벨 변경"), (1, "")]
    assert client.post(f"{B}/deploy", json={"version": body["live"], "note": "x" * 101}).status_code == 422


def test_train_version_result_includes_thresholds_and_confusable_pairs(client):
    job = train(client)
    assert job["status"] == "done", job
    result = job["result"]
    assert result["thresholds"]["level"] is None  # the sample bot sets no similarity_level
    assert isinstance(result["confusable"], list)


def test_test_build_result_includes_confusable_pairs(client):
    job = finish(client, client.post(f"{B}/test-builds"))
    assert job["status"] == "done", job
    assert isinstance(job["result"]["confusable"], list)


def test_host_confusable_needs_a_draft(host):
    from chatcore.host import NoDraft

    with pytest.raises(NoDraft):
        host.confusable("nodraft")


def test_job_endpoints_and_the_event_stream(client):
    assert client.get(f"/v1/builder/jobs/{'0' * 32}").status_code == 404
    assert client.get(f"/v1/builder/jobs/{'0' * 32}/events").status_code == 404
    assert client.get("/v1/builder/jobs/not-a-job").status_code == 422
    started = client.post(f"{B}/test-builds").json()
    with client.stream("GET", f"/v1/builder/jobs/{started['id']}/events") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
    assert events[-1]["status"] == "done" and events[-1]["result"]["build"]
    assert all(e["id"] == started["id"] for e in events)


def test_admin_token_guards_jobs_and_test_builds(host):
    client = TestClient(create_app(host, admin_token="s3cret"))
    assert client.post(f"{B}/test-builds").status_code == 401
    assert client.get(f"/v1/builder/jobs/{'0' * 32}").status_code == 401
    assert client.post(f"{B}/test/{'0' * 12}/sessions/x/messages", json={"text": "안녕"}).status_code == 401


def test_the_event_stream_sends_keepalives_while_a_job_waits(client, host, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(host, "build_test", lambda bot_id: "0" * 12 if gate.wait(5) else "")
    started = client.post(f"{B}/test-builds").json()
    lines = []
    with client.stream("GET", f"/v1/builder/jobs/{started['id']}/events", params={"wait": 0.1}) as stream:
        for line in stream.iter_lines():
            lines.append(line)
            if line.startswith(": still waiting"):
                gate.set()
    assert any(line.startswith(": still waiting") for line in lines)
    assert json.loads([line for line in lines if line.startswith("data: ")][-1][6:])["status"] == "done"


def test_a_busy_test_session_is_409(client, host, monkeypatch):
    build = finish(client, client.post(f"{B}/test-builds"))["result"]["build"]
    runtime = host.test_runtime("sample", build)

    def busy(key, *args):
        raise SessionBusy(key)

    monkeypatch.setattr(runtime.processor, "handle", busy)
    monkeypatch.setattr(runtime.processor, "open", busy)
    assert client.post(f"{B}/test/{build}/sessions/t/messages", json={"text": "안녕"}).status_code == 409
    assert client.post(f"{B}/test/{build}/sessions/t/open").status_code == 409


def test_training_and_test_builds_embed_through_the_bulk_lane(fake_embedder, tmp_path):
    class Counting:
        def __init__(self, inner):
            self.inner, self.model_id, self.prompt, self.texts = inner, inner.model_id, inner.prompt, 0

        def encode(self, texts):
            self.texts += len(texts)
            return self.inner.encode(texts)

    chat, bulk = Counting(fake_embedder), Counting(fake_embedder)
    h = BotHost(InMemoryRepository(), InMemoryStore(), chat, tmp_path, actions=synthetic_actions, min_f1=0.0,
                bulk_embedder=bulk)
    h.seed(SAMPLE)  # trains
    trained = bulk.texts
    assert trained > 100 and chat.texts == 0
    h.build_test("sample")
    assert bulk.texts > trained and chat.texts == 0


def test_start_removes_test_builds_of_an_earlier_run(client, host, tmp_path):
    finish(client, client.post(f"{B}/test-builds"))
    assert list((tmp_path / "_test").iterdir())
    host.start()
    assert not (tmp_path / "_test").exists()


def test_unexpected_errors_in_jobs_are_internal_and_logged_not_shown(client, host, monkeypatch):
    def bug(bot_id):
        raise ValueError("secret numpy detail")

    monkeypatch.setattr(host, "build_test", bug)
    failed = finish(client, client.post(f"{B}/test-builds"))
    assert failed["error"] == {"kind": "internal", "detail": "internal error"}

    def missing(bot_id):
        raise KeyError("some_internal_key")  # a bug, not "no draft"

    monkeypatch.setattr(host, "train_draft", missing)
    assert train(client)["error"]["kind"] == "internal"


def test_postgres_versions_tolerate_rows_without_a_number_and_backfill_after_the_max():
    repo = postgres_repository()
    import psycopg

    bot = f"t_{uuid.uuid4().hex[:8]}"
    doc = {**read_document(SAMPLE), "id": bot}
    repo.add_version(bot, "aaaaaaaaaaaa", doc, None)
    with psycopg.connect(repo.dsn) as conn:  # an existing number out of creation order, then a row with none
        conn.execute("UPDATE bot_versions SET number = 7 WHERE bot_id = %s", (bot,))
        conn.execute("INSERT INTO bot_versions (bot_id, version, document) VALUES (%s, %s, %s)",
                     (bot, "bbbbbbbbbbbb", psycopg.types.json.Jsonb(doc)))
    repo.deploy(bot, "bbbbbbbbbbbb")
    assert [(v.version, v.number) for v in repo.versions(bot)] == [("bbbbbbbbbbbb", 0), ("aaaaaaaaaaaa", 7)]
    assert repo.deployments(bot) == ["bbbbbbbbbbbb"] and repo.history(bot)[0].number == 0
    repo.ensure_schema()  # the backfill numbers it after the existing maximum
    assert [(v.version, v.number) for v in repo.versions(bot)] == [("bbbbbbbbbbbb", 8), ("aaaaaaaaaaaa", 7)]


def test_the_bot_list_shows_deployed_and_draft_only_bots(client):
    (sample,) = client.get("/v1/builder/bots").json()["bots"]
    versions = client.get(f"{B}/versions").json()
    assert sample == {"id": "sample", "label": draft(client)["label"], "live": versions["live"], "live_number": 1,
                      "serving": versions["live"], "draft_revision": 1, "undeployed_changes": False}
    doc = draft(client)
    doc["label"] = "바뀐 이름"
    save(client, doc)
    client.put("/v1/builder/bots/fresh/draft", json={**doc, "id": "fresh"})
    bots = {b["id"]: b for b in client.get("/v1/builder/bots").json()["bots"]}
    assert bots["sample"]["label"] == "바뀐 이름" and bots["sample"]["undeployed_changes"] is True
    assert bots["fresh"] == {"id": "fresh", "label": "바뀐 이름", "live": None, "live_number": None, "serving": None,
                             "draft_revision": 1, "undeployed_changes": True}


def test_the_bot_list_needs_the_admin_token_and_handles_an_empty_repository(host, fake_embedder, tmp_path):
    assert TestClient(create_app(host, admin_token="s3cret")).get("/v1/builder/bots").status_code == 401
    empty = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path / "empty", actions=synthetic_actions)
    assert TestClient(create_app(empty)).get("/v1/builder/bots").json() == {"bots": []}


# ---------------------------------------------------------------- E-B: welcome, block buttons, rich responses


def welcome_block(doc):
    """A welcome block that shows a card whose button and quick reply start weather directly."""
    doc["system"] = {"welcome": "welcome"}
    doc["scenarios"]["welcome"] = {"label": "웰컴", "group": "기본", "steps": [{"respond": "welcome_card"}]}
    doc["responses"]["welcome_card"] = {
        "basic_card": {"title": "어서 오세요", "description": "무엇이든 물어보세요",
                       "thumbnail": {"image_url": "https://example.com/hi.png", "alt_text": "인사"},
                       "buttons": [{"label": "날씨 보기", "action": "block", "block_id": "check_weather"}]},
        "quick_replies": [{"label": "날씨", "action": "block", "block_id": "check_weather"}, "도움말"]}
    return doc


def test_test_chat_opens_with_the_welcome_block_and_block_buttons_skip_understanding(client):
    assert save(client, welcome_block(draft(client))).json()["valid"]
    build = finish(client, client.post(f"{B}/test-builds"))["result"]["build"]
    opened = client.post(f"{B}/test/{build}/sessions/w1/open")
    assert opened.status_code == 200, opened.text
    (card,) = opened.json()["messages"]
    assert card["type"] == "basic_card" and card["basic_card"]["title"] == "어서 오세요"
    assert card["text"] == "어서 오세요\n무엇이든 물어보세요" and card["quick_replies"] == ["날씨", "도움말"]
    assert card["quick_reply_items"][0] == {"label": "날씨", "action": "block", "message_text": None,
                                            "block_id": "check_weather"}
    assert client.post(f"{B}/test/{build}/sessions/w1/open").json()["messages"] == []  # already open
    pressed = client.post(f"{B}/test/{build}/sessions/w1/messages", json={"text": "날씨", "block_id": "check_weather"})
    assert pressed.json()["nlu"]["rule"] == "block" and pressed.json()["stack"][0]["awaiting"] == "location"
    bad = client.post(f"{B}/test/{build}/sessions/w1/messages", json={"text": "x", "block_id": "greet"})
    assert bad.status_code == 422  # greet exists, but no button offers it
    assert client.post(f"{B}/test/{'0' * 12}/sessions/w1/open").status_code == 410


def test_users_open_sessions_and_press_blocks_on_the_deployed_bot(client):
    plain = client.post("/v1/bots/sample/sessions/u0/open")
    assert plain.status_code == 200 and plain.json()["messages"] == []  # the sample bot has no welcome block
    ship(client, welcome_block(draft(client)))
    opened = client.post("/v1/bots/sample/sessions/u1/open").json()
    assert [m["response"] for m in opened["messages"]] == ["welcome_card"]
    r = client.post("/v1/bots/sample/sessions/u1/messages", json={"text": "날씨", "block_id": "check_weather"})
    assert [m["response"] for m in r.json()["messages"]] == ["ask_location"]
    assert r.json()["messages"][0]["type"] == "text" and r.json()["messages"][0]["basic_card"] is None
    assert client.post("/v1/bots/sample/sessions/u1/messages",
                       json={"text": "x", "block_id": "nope"}).status_code == 422
    assert client.post("/v1/bots/sample/sessions/u1/messages",
                       json={"text": "x", "block_id": "Bad-Id"}).status_code == 422
    assert client.post("/v1/bots/nobot/sessions/u1/open").status_code == 404
