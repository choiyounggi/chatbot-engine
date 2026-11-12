"""E-F: generated knowledge-base answers in conversations, the knowledge action, search and the API."""

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from chatcore.api import create_app
from chatcore.bot.repository import InMemoryRepository
from chatcore.conversations import synthetic_actions
from chatcore.dialogue.values import CURRENT_TURN, TurnContext
from chatcore.generation import Generator, LlmClient, LlmConfig
from chatcore.host import BotHost
from chatcore.knowledge import CosineJudge, KbResult, Knowledge, KnowledgeIndex
from chatcore.nlu.entities import normalize
from chatcore.nlu.hashing import HashingEmbedder
from chatcore.store.memory import InMemoryStore

from .conftest import SAMPLE
from .test_knowledge import KB

EMB = HashingEmbedder()
B = "/v1/builder/bots/sample"
GENERATED = "결제 후 7일 안에 환불을 신청하실 수 있어요."  # its only number, 7, is in the refund passage


class FakeLlm:
    """An OpenAI-compatible server that answers from a list (or raises)."""

    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, url, headers, body, timeout):
        self.requests.append(json.loads(body))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return json.dumps({"choices": [{"message": {"content": answer}}]}).encode()


def generator(*answers, depth=3):
    llm = FakeLlm(*answers)
    return Generator(LlmClient(LlmConfig("http://llm.local", "test-model", 5), llm), depth=depth), llm


def knowledge(settings, gen=None, on_reject="extract"):
    index = KnowledgeIndex.build({**KB, "settings": settings}, EMB)
    return Knowledge(index, CosineJudge(0.5), EMB, gen, on_reject)


def found(k, text):
    return k.index.search(text, EMB.encode([normalize(text)])[0], k.judge)


# ---------------------------------------------------------------- Knowledge.answer


def test_generation_is_off_unless_the_settings_ask_for_it():
    gen, llm = generator(GENERATED)
    result, answer = knowledge({"mode": "instant"}, gen).answer("구독 환불 어떻게 해요", found(knowledge({}), "구독 환불 어떻게 해요"))
    assert answer.text == "결제 후 7일 안에 설정 > 구독에서 환불할 수 있어요." and llm.requests == []
    assert "generated" not in result.to_log()


def test_a_generated_answer_replaces_the_passage_and_is_logged():
    gen, llm = generator(GENERATED)
    k = knowledge({"mode": "instant", "generate": True}, gen)
    result, answer = k.answer("구독 환불 어떻게 해요", found(k, "구독 환불 어떻게 해요"))
    assert answer.text == GENERATED and answer.items == ("refund",) and answer.replies == ()  # an FAQ: no source line
    assert result.to_log()["generated"] == {"model": "test-model", "ms": result.generated["ms"], "used": ["refund"],
                                           "outcome": "generated"}
    assert "[질문] 구독 환불 어떻게 해요" in llm.requests[0]["messages"][1]["content"]


def test_a_generated_answer_from_a_document_names_its_source():
    gen, _ = generator("위치 정보는 30일 뒤에 지워요.")
    k = knowledge({"mode": "instant", "generate": True}, gen)
    _, answer = k.answer("위치 정보는 언제 지우나요", found(k, "위치 정보는 언제 지우나요"))
    assert answer.text == "위치 정보는 30일 뒤에 지워요.\n\n출처: 이용 안내"


@pytest.mark.parametrize("on_reject, expected", [("extract", "결제 후 7일 안에 설정 > 구독에서 환불할 수 있어요."), ("fallback", None)])
@pytest.mark.parametrize("reply", ["NO_ANSWER", "14일 안에 환불돼요.", TimeoutError()])  # 14 is in no passage
def test_a_rejected_generation_shows_the_passage_or_nothing(on_reject, expected, reply):
    gen, _ = generator(reply)
    k = knowledge({"mode": "instant", "generate": True}, gen, on_reject)
    result, answer = k.answer("구독 환불 어떻게 해요", found(k, "구독 환불 어떻게 해요"))
    assert (answer.text if answer else None) == expected
    assert result.generated["outcome"] in ("no_answer", "unsupported_number", "timeout")


def test_suggestions_and_chosen_titles_are_never_generated():
    gen, llm = generator(GENERATED)
    k = knowledge({"mode": "suggest", "generate": True}, gen)
    k.answer("구독 환불 어떻게 해요", found(k, "구독 환불 어떻게 해요"))
    k2 = knowledge({"mode": "instant", "generate": True}, gen)
    _, answer = k2.answer("이용 안내 - 개인정보", found(k2, "이용 안내 - 개인정보"))
    assert answer.text.startswith("위치 정보는 날씨 조회에만") and llm.requests == []


def test_nothing_passed_means_no_generation_at_all():
    gen, llm = generator(GENERATED)
    k = knowledge({"mode": "instant", "generate": True}, gen)
    result, answer = k.answer("주식 추천해줘", found(k, "주식 추천해줘"))
    assert answer is None and llm.requests == [] and result.generated is None


def test_consult_and_the_knowledge_action_answer_through_generation():
    gen, _ = generator(GENERATED)
    k = knowledge({"mode": "instant", "generate": True}, gen)
    _, answer = k.consult("구독 환불 어떻게 해요", "fallback", 0.2, None)
    assert answer.text == GENERATED
    token = CURRENT_TURN.set(TurnContext("구독 환불 어떻게 해요", "s"))
    try:
        assert k.action()({})["answer"] == GENERATED
    finally:
        CURRENT_TURN.reset(token)


def test_the_action_gives_the_best_entry_text_in_suggest_mode():
    k = knowledge({"mode": "suggest"})
    assert k.action()({"query": "구독 환불 어떻게 해요"})["answer"] == "결제 후 7일 안에 설정 > 구독에서 환불할 수 있어요."


def test_a_kb_result_without_generation_logs_as_before():
    assert "generated" not in KbResult((), ()).to_log()


# ---------------------------------------------------------------- host and API


@pytest.fixture
def make_host(fake_embedder, tmp_path, trained_sample):
    def make(gen):
        shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
        h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions,
                    keep=1, min_f1=0.0, judge=CosineJudge(0.5), generator=gen)
        h.seed(SAMPLE)
        return h
    return make


def deploy_kb(host, settings):
    host.save_kb_draft("sample", {**KB, "settings": settings})
    info = host.build_kb("sample")
    host.deploy("sample", host.bots()["sample"], "", info.version)
    return info.version


def test_a_conversation_and_the_search_test_answer_with_the_generated_text(make_host):
    gen, _ = generator(GENERATED)
    host = make_host(gen)
    version = deploy_kb(host, {"mode": "instant", "generate": True})
    client = TestClient(create_app(host))
    (msg,) = client.post("/v1/bots/sample/sessions/s/messages", json={"text": "구독 환불 어떻게 해요"}).json()["messages"]
    assert (msg["response"], msg["text"]) == ("knowledge", GENERATED)
    nlu = [e for e in host.store.events("sample:s") if e.type == "commands_received"][-1].nlu
    assert nlu["knowledge"]["generated"]["outcome"] == "generated"
    body = client.post(f"{B}/knowledge/search", json={"query": "구독 환불 어떻게 해요", "version": version}).json()
    assert body["answer"]["text"] == GENERATED and body["generated"]["outcome"] == "generated"


def test_without_an_llm_the_setting_changes_nothing(make_host):
    host = make_host(None)
    deploy_kb(host, {"mode": "instant", "generate": True})
    client = TestClient(create_app(host))
    (msg,) = client.post("/v1/bots/sample/sessions/s/messages", json={"text": "구독 환불 어떻게 해요"}).json()["messages"]
    assert msg["text"] == "결제 후 7일 안에 설정 > 구독에서 환불할 수 있어요."
    body = client.post(f"{B}/knowledge/search", json={"query": "구독 환불 어떻게 해요"}).json()
    assert body["generated"] is None


def test_llm_status_reports_configuration_and_reachability(make_host):
    assert TestClient(create_app(make_host(None))).get("/v1/builder/llm").json() == {
        "configured": False, "model": None, "ok": None, "ms": None, "error": None}
    ok, llm = generator("네")
    body = TestClient(create_app(make_host(ok))).get("/v1/builder/llm").json()
    assert body["configured"] and body["ok"] and body["model"] == "test-model" and llm.requests[0]["max_tokens"] == 8
    down, _ = generator(ConnectionRefusedError())
    body = TestClient(create_app(make_host(down))).get("/v1/builder/llm").json()
    assert body["configured"] and body["ok"] is False and body["error"]


def test_llm_status_needs_the_admin_token(make_host):
    assert TestClient(create_app(make_host(None), "secret-token-for-tests")).get("/v1/builder/llm").status_code == 401
