"""HTTP API, runtime assembly, CLI loop and the batched embedder (fake embedder, no model download)."""

import io
import json
import logging
import random
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from chatcore.api import create_app
from chatcore.cli import chat
from chatcore.conversations import synthetic_actions
from chatcore.dialogue.turn import TurnProcessor
from chatcore.host import StaticHost
from chatcore.nlu.entities import EntityExtractor
from chatcore.nlu.interpreter import NluInterpreter
from chatcore.nlu.model import CachedEmbedder, ModelMismatch, NluModel
from chatcore.nlu.train import train
from chatcore.runtime import BotRuntime, build_runtime, default_store
from chatcore.serving import BatchedEmbedder
from chatcore.store.memory import InMemoryStore

from .conftest import ROOT, SAMPLE


def static(rt):
    return StaticHost({"sample": rt}, rt.processor.store)


@pytest.fixture(scope="module")
def artifacts(sample_bot, fake_embedder, tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    train(sample_bot, CachedEmbedder(fake_embedder), root)
    return root


@pytest.fixture
def runtime(sample_bot, fake_embedder, artifacts):
    return build_runtime(SAMPLE, artifacts, fake_embedder, InMemoryStore(), actions=synthetic_actions())


@pytest.fixture
def client(runtime):
    return TestClient(create_app(static(runtime)))


def say(client, text, session="s1", **params):
    return client.post(f"/v1/bots/sample/sessions/{session}/messages", json={"text": text}, params=params)


def test_health(client, sample_bot):
    assert client.get("/healthz").json() == {"ok": True, "bots": {"sample": sample_bot.version}}


def test_conversation_over_http(client):
    """Exact training sentences keep the intent independent of the fake embedder."""
    first = say(client, "날씨 어때요?")
    assert first.status_code == 200
    assert [m["response"] for m in first.json()["messages"]] == ["ask_location"]
    assert first.json()["messages"][0]["quick_replies"] == ["서울", "부산", "제주"]
    second = say(client, "부산").json()
    assert second["messages"] == [{"response": "weather_report", "text": "부산은 지금 맑음이에요.", "quick_replies": [],
                                   "type": "text", "simple_image": None, "basic_card": None, "list_card": None,
                                   "carousel": None, "quick_reply_items": []}]
    assert second["nlu"] is None


def test_debug_returns_the_nlu_result(client):
    body = say(client, "안녕하세요", debug="true").json()
    assert body["nlu"]["rule"] == "intent" and body["nlu"]["intents"][0][0] == "greeting"


def test_sessions_are_separate(client):
    say(client, "날씨 어때요?", session="a")
    assert [m["response"] for m in say(client, "부산", session="b").json()["messages"]] != ["weather_report"]


@pytest.mark.parametrize("text", ["", "   ", "가" * 1001])
def test_invalid_text_is_answered_by_the_bot(client, text):
    body = say(client, text).json()
    assert [m["response"] for m in body["messages"]] == ["input_invalid"]


@pytest.mark.parametrize(
    "path, payload, status",
    [("/v1/bots/nope/sessions/s/messages", {"text": "hi"}, 404), ("/v1/bots/sample/sessions/bad id!/messages", {"text": "hi"}, 422),
     ("/v1/bots/Sample/sessions/s/messages", {"text": "hi"}, 422), ("/v1/bots/sample/sessions/s/messages", {}, 422),
     ("/v1/bots/sample/sessions/s/messages", {"text": "x" * 10_001}, 422), ("/v1/bots/sample/sessions/" + "s" * 65 + "/messages", {"text": "hi"}, 422),
     ("/v1/bots/nope/nlu/parse", {"text": "hi"}, 404), ("/v1/bots/sample/nlu/parse", {"text": " "}, 422)],
)
def test_bad_requests(client, path, payload, status):
    assert client.post(path, json=payload).status_code == status


def test_concurrent_turn_of_the_same_session_gets_409(sample_bot, fake_embedder, artifacts):
    entered, release = threading.Event(), threading.Event()
    store = InMemoryStore()
    model = NluModel.load(sample_bot, artifacts / "sample" / sample_bot.version, fake_embedder, EntityExtractor(sample_bot))

    def slow_parse(text):
        if text == "안녕하세요":  # only the first session's message is slow
            entered.set()
            release.wait(5)
        return model.parse(text)

    proc = TurnProcessor(sample_bot, store, NluInterpreter(slow_parse), synthetic_actions(), rng=random.Random(0))
    client = TestClient(create_app(static(BotRuntime(sample_bot, proc, model))))
    first: dict = {}
    t = threading.Thread(target=lambda: first.update(r=say(client, "안녕하세요")))
    t.start()
    assert entered.wait(5)
    try:
        busy = say(client, "안녕")
        started = time.perf_counter()
        other = say(client, "안녕", session="other")  # another session is not blocked
        other_seconds = time.perf_counter() - started
    finally:
        release.set()
        t.join(5)
    assert busy.status_code == 409 and "previous message" in busy.json()["detail"]
    assert first["r"].status_code == 200 and other.status_code == 200 and other_seconds < 2


class BrokenStore:
    def session(self, session_id):
        raise ConnectionError("could not connect to db.internal:5432 password=hunter2")

    def events(self, session_id):
        raise ConnectionError("down")


def test_storage_failure_is_503_without_details(sample_bot, fake_embedder, artifacts):
    rt = build_runtime(SAMPLE, artifacts, fake_embedder, BrokenStore(), actions=synthetic_actions())
    response = say(TestClient(create_app(static(rt))), "안녕하세요")
    assert response.status_code == 503
    assert "hunter2" not in response.text and "db.internal" not in response.text


def test_nlu_parse(client):
    body = client.post("/v1/bots/sample/nlu/parse", json={"text": "부산광역시 날씨 어때요?"}).json()
    assert body["entities"][0] == {"entity": "location", "value": "부산", "text": "부산광역시"}
    assert {"intents", "similarity", "matched_by"} <= set(body)


def test_each_turn_is_logged_as_json(client, caplog):
    with caplog.at_level(logging.INFO, logger="chatcore.turn"):
        say(client, "안녕하세요")
    line = json.loads(caplog.records[-1].getMessage())
    assert line["intent"] == "greeting" and line["responses"] == ["greeting"] and line["ms"] >= 0


# ---------------------------------------------------------------- runtime, CLI, batching


def test_build_runtime_needs_a_trained_model(fake_embedder, tmp_path):
    with pytest.raises(ModelMismatch, match="make train"):
        build_runtime(SAMPLE, tmp_path, fake_embedder, InMemoryStore())


def test_default_store_is_memory_without_database_url(monkeypatch):
    monkeypatch.delenv("CHATCORE_DATABASE_URL", raising=False)
    assert isinstance(default_store(), InMemoryStore)


def test_cli_chat_loop(runtime):
    inputs = iter(["안녕하세요", "/debug", "날씨 어때요?", "/quit"])
    out = io.StringIO()
    assert chat(runtime, lambda: next(inputs), out) == 0
    text = out.getvalue()
    assert "봇> " in text and "[서울 | 부산 | 제주]" in text and "(debug on)" in text and "intent" in text


def test_cli_chat_ends_on_eof(runtime):
    def eof():
        raise EOFError

    out = io.StringIO()
    assert chat(runtime, eof, out) == 0


def test_batched_embedder_matches_the_inner_one(fake_embedder):
    batched = BatchedEmbedder(fake_embedder)
    try:
        np.testing.assert_allclose(batched.encode(["가나", "다라"]), fake_embedder.encode(["가나", "다라"]))
        assert batched.encode([]).shape == (0, 0)
        assert (batched.model_id, batched.prompt) == (fake_embedder.model_id, fake_embedder.prompt)
    finally:
        batched.close()


def test_longest_ids_fit_the_store(client):
    assert say(client, "안녕하세요", session="s" * 63).status_code == 200
    assert say(client, "안녕하세요", session="s" * 64).status_code == 422


@pytest.mark.parametrize("text, status", [("가" * 1001, 422), ("  부산광역시   날씨  ", 200)])
def test_nlu_parse_applies_turn_input_rules(client, text, status):
    r = client.post("/v1/bots/sample/nlu/parse", json={"text": text})
    assert r.status_code == status
    if status == 200:
        assert r.json()["text"] == "부산광역시 날씨"


class CorruptStore(BrokenStore):
    def session(self, session_id):
        raise KeyError("inconsistent log")


def test_non_storage_failure_is_500_not_retryable_503(sample_bot, fake_embedder, artifacts):
    rt = build_runtime(SAMPLE, artifacts, fake_embedder, CorruptStore(), actions=synthetic_actions())
    response = say(TestClient(create_app(static(rt))), "안녕하세요")
    assert response.status_code == 500 and "inconsistent" not in response.text


def test_nlu_parse_failure_is_500_without_details(runtime):
    def boom(text):
        raise RuntimeError("secret detail")

    runtime.model.parse = boom
    r = TestClient(create_app(static(runtime))).post("/v1/bots/sample/nlu/parse", json={"text": "안녕"})
    assert r.status_code == 500 and "secret" not in r.text


def test_bulk_lane_of_batched_embedder_gives_the_same_vectors(fake_embedder):
    batched = BatchedEmbedder(fake_embedder)
    lane = batched.bulk()
    try:
        assert (lane.model_id, lane.prompt) == (batched.model_id, batched.prompt)
        assert np.allclose(lane.encode(["안녕", "날씨"]), batched.encode(["안녕", "날씨"]))
        assert lane.encode([]).shape == (0, 0)
    finally:
        batched.close()



def test_docs_openapi_json_is_up_to_date():
    from chatcore.openapi import render

    stored = (ROOT / "docs/openapi.json").read_text(encoding="utf-8")
    assert stored == render(), "docs/openapi.json is stale: run `make openapi`"


def test_the_schema_describes_drafts_with_the_bot_document_model():
    from chatcore.openapi import schema

    spec = schema()
    draft = spec["paths"]["/v1/builder/bots/{bot}/draft"]
    assert draft["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/BotDocument"
    assert draft["put"]["requestBody"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/BotDocument"
    assert "version" not in spec["components"]["schemas"]["BotDocument"]["properties"]  # computed, never sent


def test_hashing_embedder_is_deterministic_normalized_and_handles_empty_input():
    import numpy as np

    from chatcore.nlu.hashing import HashingEmbedder

    emb = HashingEmbedder()
    a, b = emb.encode(["날씨 어때", "날씨 어때"]), emb.encode(["날씨 어때"])
    assert np.array_equal(a[0], b[0]) and np.isclose(np.linalg.norm(a[0]), 1.0)
    assert emb.encode([""]).shape == (1, 256) and not emb.encode([""]).any()  # no n-grams: zero vector, no NaN
    assert emb.encode([]).shape == (0, 256)
