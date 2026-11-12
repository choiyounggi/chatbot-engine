"""E-D: the knowledge base in conversations and over the builder API — commands, the worker thread,
sessions pinned to (bot, knowledge-base) versions, deploy and rollback by pair, files, test builds."""

import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chatcore.api import create_app
from chatcore.bot.repository import InMemoryRepository
from chatcore.conversations import synthetic_actions
from chatcore.dialogue.commands import AnswerKnowledge, CommandFormatError, from_dict, to_dict
from chatcore.dialogue.events import SessionStarted
from chatcore.dialogue.turn import ScriptedInterpreter, TurnProcessor
from chatcore.host import BotHost, NoKnowledgeBase
from chatcore.knowledge import Candidate, CosineJudge, Knowledge, KnowledgeIndex
from chatcore.nlu.hashing import HashingEmbedder
from chatcore.serving import BatchedEmbedder, Pair, Passage, run_items
from chatcore.store.memory import InMemoryStore

from .conftest import SAMPLE
from .test_knowledge import KB, pdf_with_text

B = "/v1/builder/bots/sample"
REFUND = "결제 후 7일 안에 설정 > 구독에서 환불할 수 있어요."


@pytest.fixture
def host(fake_embedder, tmp_path, trained_sample):
    shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
    h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions, keep=1,
                min_f1=0.0, judge=CosineJudge(0.5))
    h.seed(SAMPLE)
    return h


@pytest.fixture
def client(host):
    return TestClient(create_app(host))


def finish(client, started):
    assert started.status_code == 202, started.text
    for _ in range(500):
        job = client.get(f"/v1/builder/jobs/{started.json()['id']}").json()
        if job["status"] in ("done", "failed"):
            return job
    raise AssertionError("job did not finish")


def put_kb(client, document):
    current = client.get(f"{B}/knowledge/draft")
    headers = {"If-Match": current.headers["ETag"]} if current.status_code == 200 else {}
    return client.put(f"{B}/knowledge/draft", json=document, headers=headers)


def built(client, document=KB):
    assert put_kb(client, document).json()["valid"]
    job = finish(client, client.post(f"{B}/knowledge/versions"))
    assert job["status"] == "done", job
    return job["result"]["version"]


def say(client, text, session):
    r = client.post(f"/v1/bots/sample/sessions/{session}/messages", json={"text": text})
    assert r.status_code == 200, r.text
    return r.json()["messages"]


# ---------------------------------------------------------------- commands, engine, worker


def test_answer_knowledge_round_trips_as_a_dict() -> None:
    cmd = AnswerKnowledge("답", (("짧게", "전체 제목"),), ("refund",))
    assert to_dict(cmd) == {"knowledge": {"text": "답", "replies": [["짧게", "전체 제목"]], "items": ["refund"]}}
    assert from_dict(to_dict(cmd)) == cmd
    assert from_dict({"knowledge": {"text": "답"}}) == AnswerKnowledge("답")


@pytest.mark.parametrize("bad", [{"knowledge": "답"}, {"knowledge": {"text": ""}}, {"knowledge": {"text": "a", "replies": [["x"]]}},
                                 {"knowledge": {"text": "a", "replies": "x"}}, {"knowledge": {"text": "a", "items": "x"}}])
def test_answer_knowledge_refuses_malformed_dicts(bad) -> None:
    with pytest.raises(CommandFormatError):
        from_dict(bad)


def test_the_engine_says_the_answer_with_buttons_that_send_the_whole_title(sample_bot, registry) -> None:
    store = InMemoryStore()
    cmd = AnswerKnowledge("골라 주세요", (("구독을 해지하면…", "구독을 해지하면 환불이 되나요"),), ("a",))
    tp = TurnProcessor(sample_bot, store, ScriptedInterpreter({"질문": [cmd]}), registry, kb_version="0123456789ab")
    out = tp.handle("s1", "질문")
    (msg,) = out.messages
    assert (msg.response, msg.text) == ("knowledge", "골라 주세요")
    assert [(r.label, r.message_text, r.action) for r in msg.replies] == [("구독을 해지하면…", "구독을 해지하면 환불이 되나요", "message")]
    started = next(e for e in store.events("s1") if isinstance(e, SessionStarted))
    assert started.kb_version == "0123456789ab"


def test_a_new_knowledge_base_alone_never_restarts_a_running_scenario(sample_bot, registry) -> None:
    store = InMemoryStore()
    ask = {"날씨": [from_dict({"start": "check_weather"})]}
    TurnProcessor(sample_bot, store, ScriptedInterpreter(ask), registry).handle("s1", "날씨")  # asks for a place
    tp = TurnProcessor(sample_bot, store, ScriptedInterpreter(ask), registry, kb_version="0123456789ab")
    out = tp.handle("s1", "아무 말")
    assert "restarted" not in [m.response for m in out.messages]
    starts = [e for e in store.events("s1") if isinstance(e, SessionStarted)]
    assert [s.kb_version for s in starts] == [None, "0123456789ab"]


class FakeCrossEncoder:
    threshold = 0.2

    def __init__(self):
        self.pairs = []

    def score_pairs(self, pairs):
        self.pairs += list(pairs)
        return np.array([0.9 if "환불" in text else 0.1 for _, text in pairs])


def test_run_items_sends_sentences_passages_and_pairs_to_the_right_model() -> None:
    class Prompted(HashingEmbedder):
        prompt = "query: "

        def encode(self, texts):
            return np.ones((len(texts), 2), dtype=np.float32)

        def encode_passages(self, texts):
            return np.zeros((len(texts), 2), dtype=np.float32)

    ce = FakeCrossEncoder()
    out = run_items(Prompted(), ce, ["문장", Passage("문서"), Pair("질문", "환불 규정"), "또 문장"])
    assert out[0].tolist() == [1, 1] and out[1].tolist() == [0, 0] and out[3].tolist() == [1, 1]
    assert out[2].tolist() == [0.9] and ce.pairs == [("질문", "환불 규정")]
    (missing,) = run_items(Prompted(), None, [Pair("q", "t")])
    assert isinstance(missing, RuntimeError) and "cross-encoder" in str(missing)


def test_the_batched_embedder_runs_passages_and_the_judge_on_its_worker() -> None:
    plain = BatchedEmbedder(HashingEmbedder())
    try:
        assert plain.judge() is None and plain.encode_passages([]).shape == (0, 0)
        assert np.allclose(plain.encode_passages(["문서"]), HashingEmbedder().encode(["문서"]))
        assert np.allclose(plain.bulk().encode_passages(["문서"]), HashingEmbedder().encode(["문서"]))  # type: ignore[attr-defined]
    finally:
        plain.close()
    ce = FakeCrossEncoder()
    judged = BatchedEmbedder(HashingEmbedder(), cross_encoder=ce)  # type: ignore[arg-type]
    try:
        judge = judged.judge()
        index = KnowledgeIndex.build(KB, HashingEmbedder())
        cands = index.search("환불", HashingEmbedder().encode(["환불"])[0], CosineJudge(0)).candidates
        from chatcore.knowledge import Candidate

        scores = judge.score("환불", [Candidate(h.entry, h.cosine) for h in cands])
        assert judge.threshold == 0.2 and judge.name == "cross-encoder"
        assert scores.tolist() == [0.9 if "환불" in h.entry.judge_text else 0.1 for h in cands]
        assert judge.score("환불", []).shape == (0,)
    finally:
        judged.close()


# ---------------------------------------------------------------- the draft over HTTP


def test_the_knowledge_draft_saves_with_etags_even_when_invalid(client) -> None:
    assert client.get(f"{B}/knowledge/draft").status_code == 404
    created = client.put(f"{B}/knowledge/draft", json=KB)
    assert created.status_code == 200 and created.json()["valid"] and created.json()["revision"] == 1
    assert created.headers["ETag"] == '"1"'
    assert client.get(f"{B}/knowledge/draft").json() == KB
    assert client.put(f"{B}/knowledge/draft", json=KB).status_code == 428
    stale = client.put(f"{B}/knowledge/draft", json=KB, headers={"If-Match": '"7"'})
    assert stale.status_code == 412 and stale.json()["revision"] == 1
    bad = client.put(f"{B}/knowledge/draft", json={"faq": [{"id": "x"}]}, headers={"If-Match": '"1"'})
    assert bad.status_code == 200 and not bad.json()["valid"] and bad.json()["version"] is None and bad.json()["errors"]
    assert client.get("/v1/builder/bots/nobody/knowledge/draft").status_code == 404


def test_building_needs_a_valid_draft(client) -> None:
    job = finish(client, client.post(f"{B}/knowledge/versions"))
    assert job["status"] == "failed" and job["error"]["kind"] == "no_draft"
    put_kb(client, {"faq": [{"id": "x"}]})
    job = finish(client, client.post(f"{B}/knowledge/versions"))
    assert job["status"] == "failed" and job["error"]["kind"] == "invalid_draft" and job["error"]["errors"]


def test_a_built_version_lists_previews_and_searches(client) -> None:
    version = built(client)
    listed = client.get(f"{B}/knowledge/versions").json()
    assert listed["live"] is None
    assert [(v["version"], v["number"], v["faq"], v["chunks"], v["live"]) for v in listed["versions"]] == [(version, 1, 2, 1, False)]
    entries = client.get(f"{B}/knowledge/versions/{version}/entries").json()["entries"]
    assert [(e["id"], e["kind"]) for e in entries] == [("refund", "faq"), ("widget", "faq"), ("policy#0", "doc")]
    assert entries[2]["title"] == "이용 안내 - 개인정보" and entries[2]["source"] == "이용 안내"
    assert client.get(f"{B}/knowledge/versions/{'0' * 12}/entries").status_code == 404
    found = client.post(f"{B}/knowledge/search", json={"query": "구독 환불 어떻게 해요"}).json()
    assert found["version"] == version and found["judge"] == "cosine" and found["threshold"] == 0.5
    assert found["answer"]["text"] == REFUND and not found["exact"]
    assert [(c["id"], c["passed"]) for c in found["candidates"]] == [("refund", True), ("widget", False), ("policy#0", False)]
    miss = client.post(f"{B}/knowledge/search", json={"query": "주식 추천해줘", "version": version}).json()
    assert miss["answer"] is None
    assert client.post(f"{B}/knowledge/search", json={"query": "x", "version": "0" * 12}).status_code == 404
    assert client.post(f"{B}/knowledge/search", json={"query": ""}).status_code == 422


def test_searching_before_anything_is_built_is_a_conflict(client, host) -> None:
    assert client.post(f"{B}/knowledge/search", json={"query": "환불"}).status_code == 409
    with pytest.raises(NoKnowledgeBase):
        host.build_kb("sample")


# ---------------------------------------------------------------- conversations, deploy, rollback


def test_deployed_with_a_knowledge_base_the_bot_answers_new_sessions_from_it(client, host) -> None:
    live = host.bots()["sample"]
    assert say(client, "구독 환불 어떻게 해요", "before")[0]["response"] == "fallback"
    version = built(client)
    deployed = client.post(f"{B}/deploy", json={"version": live, "knowledge_version": version})
    assert deployed.json() == {"bot": "sample", "live": live, "knowledge_version": version}
    (msg,) = say(client, "구독 환불 어떻게 해요", "after")
    assert (msg["response"], msg["text"]) == ("knowledge", REFUND)
    assert say(client, "구독 환불 어떻게 해요", "before")[0]["response"] == "fallback"  # pinned to its start
    nlu = [e for e in host.store.events("sample:after") if e.type == "commands_received"][-1].nlu
    assert nlu["rule"] == "knowledge" and nlu["rule_before"] == "fallback" and nlu["knowledge"]["answered"] == ["refund"]
    assert say(client, "주식 추천해줘", "after")[0]["response"] == "fallback"
    queue = [i["text"] for i in client.get(f"{B}/misunderstood").json()["items"]]
    assert "주식 추천해줘" in queue and "구독 환불 어떻게 해요" in queue  # only the "before" session's turns
    history = client.get(f"{B}/versions").json()["history"]
    assert history[0]["knowledge_version"] == version and history[1]["knowledge_version"] is None
    assert client.get(f"{B}/knowledge/versions").json()["live"] == version


def test_rollback_restores_the_previous_pair_and_unknown_knowledge_versions_are_refused(client, host) -> None:
    live = host.bots()["sample"]
    version = built(client)
    assert client.post(f"{B}/deploy", json={"version": live, "knowledge_version": "0" * 12}).status_code == 404
    client.post(f"{B}/deploy", json={"version": live, "knowledge_version": version})
    back = client.post(f"{B}/rollback").json()
    assert back == {"bot": "sample", "live": live, "knowledge_version": None}
    assert host.live("sample") == (live, None)
    assert say(client, "구독 환불 어떻게 해요", "new")[0]["response"] == "fallback"
    again = client.post(f"{B}/rollback").json()  # same bot version, the knowledge base differs: a real step back
    assert again["knowledge_version"] == version


def test_a_restarted_server_rebuilds_missing_index_files(host, fake_embedder, tmp_path, client) -> None:
    live = host.bots()["sample"]
    version = built(client)
    host.deploy("sample", live, "", version)
    shutil.rmtree(tmp_path / "sample" / "kb")
    fresh = BotHost(host.repository, InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions,
                    min_f1=0.0, judge=CosineJudge(0.5))
    fresh.start()
    assert fresh.live("sample") == (live, version) and (tmp_path / "sample" / "kb" / version / "index.json").is_file()
    (msg,) = TestClient(create_app(fresh)).post("/v1/bots/sample/sessions/s/messages",
                                                json={"text": "구독 환불 어떻게 해요"}).json()["messages"]
    assert msg["text"] == REFUND


def test_the_knowledge_action_uses_the_bots_knowledge_base(host, client) -> None:
    runtime = host.runtime("sample")
    assert runtime.processor.engine.actions.run("knowledge", {"query": "위치 정보는 언제 지우나요"})["found"] == "no"
    host.deploy("sample", host.bots()["sample"], "", built(client))
    actions = host.runtime("sample").processor.engine.actions
    assert actions.run("knowledge", {"query": "위치 정보는 언제 지우나요"})["source"] == "이용 안내"


def test_a_test_build_answers_from_the_knowledge_draft_and_refuses_an_invalid_one(client) -> None:
    put_kb(client, KB)
    job = finish(client, client.post(f"{B}/test-builds"))
    assert job["status"] == "done", job
    build = job["result"]["build"]
    r = client.post(f"{B}/test/{build}/sessions/t1/messages", json={"text": "구독 환불 어떻게 해요"}).json()
    assert r["messages"][0]["text"] == REFUND
    put_kb(client, {"faq": [{"id": "x"}]})
    job = finish(client, client.post(f"{B}/test-builds"))
    assert job["status"] == "failed" and job["error"]["kind"] == "invalid_knowledge"


def test_a_failing_judge_leaves_the_turn_as_the_rules_decided(host, client) -> None:
    class Broken(CosineJudge):
        def score(self, query, candidates):
            raise RuntimeError("model gone")

    host.judge = Broken()
    host.deploy("sample", host.bots()["sample"], "", built(client))
    assert say(client, "구독 환불 어떻게 해요", "s")[0]["response"] == "fallback"
    nlu = [e for e in host.store.events("sample:s") if e.type == "commands_received"][-1].nlu
    assert nlu["knowledge"] == {"error": "RuntimeError"} and nlu["rule"] == "fallback"


def test_a_knowledge_base_is_not_consulted_while_a_question_is_pending(host, client) -> None:
    host.deploy("sample", host.bots()["sample"], "", built(client))
    assert say(client, "오늘 날씨 어때", "s")[0]["response"] == "ask_location"
    replies = say(client, "구독 환불 어떻게 해요", "s")
    assert "knowledge" not in [m["response"] for m in replies]


# ---------------------------------------------------------------- files


def test_extract_returns_the_text_of_an_uploaded_file(client) -> None:
    def up(data, kind):
        return client.post(f"{B}/knowledge/extract", content=data, headers={"Content-Type": kind})

    assert up(pdf_with_text("Refunds within 7 days"), "application/pdf").json() == {"text": "Refunds within 7 days"}
    assert up("# 안내\n\n본문".encode(), "text/markdown; charset=utf-8").json() == {"text": "# 안내\n\n본문"}
    assert up(b"", "text/plain").status_code == 422
    assert up(b"<p>", "text/html").status_code == 422
    assert up(b"x" * (10 * 1024 * 1024 + 1), "text/plain").status_code == 413
    assert client.post("/v1/builder/bots/nobody/knowledge/extract", content=b"x").status_code == 404


def test_knowledge_endpoints_need_the_admin_token(host) -> None:
    locked = TestClient(create_app(host, "secret-token-for-tests"))
    for method, path in (("get", "knowledge/draft"), ("put", "knowledge/draft"), ("post", "knowledge/extract"),
                         ("post", "knowledge/versions"), ("get", "knowledge/versions"), ("post", "knowledge/search"),
                         ("get", f"knowledge/versions/{'0' * 12}/entries")):
        assert getattr(locked, method)(f"{B}/{path}").status_code == 401, path


def test_knowledge_in_a_conversation_reuses_the_nlu_vector(fake_embedder, host, client) -> None:
    host.deploy("sample", host.bots()["sample"], "", built(client))
    knowledge = host.runtime("sample").processor.interpreter.knowledge
    assert isinstance(knowledge, Knowledge)
    before = fake_embedder.calls
    say(client, "구독 환불 어떻게 해요", "s")
    assert fake_embedder.calls - before == 1  # the NLU's encoding only; the knowledge base reused it


def test_the_index_cache_keeps_only_the_most_recent(host, client, monkeypatch) -> None:
    import chatcore.host as host_module

    monkeypatch.setattr(host_module, "KB_CACHE", 1)
    first = built(client)
    second = built(client, {**KB, "faq": KB["faq"][:1]})
    assert list(host._indexes) == [("sample", second)]
    assert host.kb_index("sample", first).version == first  # loaded again from its files
    assert list(host._indexes) == [("sample", first)]


def test_a_stored_knowledge_base_that_no_longer_hashes_to_its_version_is_refused(host) -> None:
    from chatcore.knowledge import KnowledgeError

    host.repository.add_kb_version("sample", "0123456789ab", KB, 2, 1)  # not KB's hash; no files either
    with pytest.raises(KnowledgeError, match="hashes to"):
        host.kb_index("sample", "0123456789ab")


# ---------------------------------------------------------------- review fixes


def test_a_cross_encoder_failure_fails_only_its_own_items() -> None:
    class Failing(FakeCrossEncoder):
        def score_pairs(self, pairs):
            raise RuntimeError("out of memory")

    out = run_items(HashingEmbedder(), Failing(), ["문장", Pair("q", "t"), Passage("문서")])
    assert out[0].shape == (HashingEmbedder.dims,) and out[2].shape == (HashingEmbedder.dims,)
    assert isinstance(out[1], RuntimeError)
    batched = BatchedEmbedder(HashingEmbedder(), cross_encoder=Failing())  # type: ignore[arg-type]
    try:
        with pytest.raises(RuntimeError, match="out of memory"):
            hits = KnowledgeIndex.build(KB, HashingEmbedder()).search(
                "환불", HashingEmbedder().encode(["환불"])[0], CosineJudge(0)).candidates
            batched.judge().score("q", [Candidate(h.entry, h.cosine) for h in hits])
        assert batched.encode(["문장"]).shape == (1, HashingEmbedder.dims)  # the worker survives
    finally:
        batched.close()


def test_a_skill_named_knowledge_still_works_and_replaces_the_built_in(host, client) -> None:
    doc = client.get(f"{B}/draft").json()
    doc["skills"] = {"knowledge": {"url": "https://example.com/kb", "timeout_ms": 1000}}
    tag = client.get(f"{B}/draft").headers["ETag"]
    saved = client.put(f"{B}/draft", json=doc, headers={"If-Match": tag}).json()
    assert saved["valid"], saved["errors"]
    from chatcore.bot.loader import bot_from_document

    bot = bot_from_document(doc, host.actions.names())
    kb = KnowledgeIndex.build(KB, HashingEmbedder())
    actions = host.actions_for(bot, Knowledge(kb, CosineJudge(), HashingEmbedder()))
    assert actions._actions["knowledge"].timeout == 1.5  # the skill's own, not the built-in


def test_kb_read_endpoints_report_an_unusable_stored_version_as_a_conflict(host, client) -> None:
    host.repository.add_kb_version("sample", "0123456789ab", KB, 2, 1)
    assert client.get(f"{B}/knowledge/versions/0123456789ab/entries").status_code == 409
    assert client.post(f"{B}/knowledge/search", json={"query": "환불", "version": "0123456789ab"}).status_code == 409
    r = client.post(f"{B}/deploy", json={"version": host.bots()["sample"], "knowledge_version": "fedcba987654"})
    assert r.status_code == 404 and "fedcba987654" in r.json()["detail"]


def test_a_deploy_without_knowledge_version_keeps_the_live_one_and_null_removes_it(host, client) -> None:
    live = host.bots()["sample"]
    version = built(client)
    client.post(f"{B}/deploy", json={"version": live, "knowledge_version": version})
    kept = client.post(f"{B}/deploy", json={"version": live, "note": "다시"}).json()
    assert kept["knowledge_version"] == version and host.live("sample") == (live, version)
    gone = client.post(f"{B}/deploy", json={"version": live, "knowledge_version": None}).json()
    assert gone["knowledge_version"] is None and host.live("sample") == (live, None)


def test_a_knowledge_base_that_cannot_load_at_start_leaves_the_bot_served_without_it(host, client, fake_embedder,
                                                                                     tmp_path) -> None:
    live = host.bots()["sample"]
    host.repository.add_kb_version("sample", "0123456789ab", KB, 2, 1)  # hashes to something else; no files
    host.repository.deploy("sample", live, "", "0123456789ab")
    fresh = BotHost(host.repository, InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions,
                    min_f1=0.0, judge=CosineJudge(0.5))
    fresh.start()
    assert fresh.live("sample") == (live, None)


def test_concurrent_cold_loads_of_one_index_build_it_once(host, client, monkeypatch, tmp_path) -> None:
    import threading

    version = built(client)
    host._indexes.clear()
    shutil.rmtree(tmp_path / "sample" / "kb")
    builds = []
    real = KnowledgeIndex.build

    def slow(document, embedder):
        builds.append(1)
        threading.Event().wait(0.05)
        return real(document, embedder)

    monkeypatch.setattr(KnowledgeIndex, "build", staticmethod(slow))
    got = []
    threads = [threading.Thread(target=lambda: got.append(host.kb_index("sample", version))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(builds) == 1 and len(got) == 4 and all(g is got[0] for g in got)  # one index, shared
