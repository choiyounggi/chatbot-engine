"""E-H: LLM-proposed block examples and FAQ similar questions — parsing, screening, search, API."""

import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chatcore import augment
from chatcore.api import create_app
from chatcore.augment import Proposal, faq_messages, intent_messages, parse_items, propose, screen
from chatcore.bot.repository import InMemoryRepository
from chatcore.conversations import synthetic_actions
from chatcore.generation import GenerationError, Generator, LlmClient, LlmConfig
from chatcore.host import BotHost
from chatcore.knowledge import MAX_SIMILAR, CosineJudge, FaqEntry, Knowledge, KnowledgeIndex
from chatcore.nlu.entities import normalize
from chatcore.nlu.hashing import HashingEmbedder
from chatcore.store.memory import InMemoryStore

from .conftest import SAMPLE
from .test_knowledge import KB

EMB = HashingEmbedder()
B = "/v1/builder/bots/sample"


def reply(items):
    return json.dumps({"choices": [{"message": {"content": json.dumps({"items": items}, ensure_ascii=False)},
                                    "finish_reason": "stop"}]}).encode()


class FakeLlm:
    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, url, headers, body, timeout):
        self.requests.append((json.loads(body), timeout))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


# ---------------------------------------------------------------- parsing and screening


def test_parse_reads_the_json_asked_for_a_bare_list_or_lines() -> None:
    assert parse_items('설명\n{"items": [{"text": "날시 알려줘", "type": "typo"}, {"text": "날씨는?", "type": "weird"}]}') == [
        ("날시 알려줘", "typo"), ("날씨는?", "paraphrase")]
    assert parse_items('["오늘 날씨", {"text": "날씨 어때", "type": "casual"}, 3]') == [
        ("오늘 날씨", "paraphrase"), ("날씨 어때", "casual")]
    assert parse_items("```\n1. 날씨 알려줘\n- 오늘 날씨는?\n• \"날씨 어때\"\n```") == [
        ("날씨 알려줘", "paraphrase"), ("오늘 날씨는?", "paraphrase"), ("날씨 어때", "paraphrase")]
    assert parse_items('{"items": "not a list"}\n') == []
    assert parse_items("") == []


def test_screen_marks_duplicates_invalid_conflicts_and_keeps_the_rest() -> None:
    items = [("오늘 날시 알려줘", "typo"), ("오늘  날씨 알려줘", "paraphrase"), ("오늘 날시 알려줘", "typo"),
             ("{location} 날씨 어떰", "casual"), ("{city} 날씨", "synonym"), ("날씨 {알려", "typo"),
             ("가" * 301, "paraphrase"), ("   ", "typo"), ("안녕하세요", "paraphrase")]
    out = screen(items, ["오늘 날씨 알려줘"], {"location"}, lambda t: "‘인사’ 블록으로 판단돼요 (0.98)" if t == "안녕하세요" else None)
    assert [(p.status, p.text[:12]) for p in out] == [
        ("ok", "오늘 날시 알려줘"), ("duplicate", "오늘 날씨 알려줘"), ("duplicate", "오늘 날시 알려줘"),
        ("ok", "{location} 날"), ("invalid", "{city} 날씨"), ("invalid", "날씨 {알려"),
        ("invalid", "가" * 12), ("invalid", ""), ("conflict", "안녕하세요")]
    assert out[-1].detail == "‘인사’ 블록으로 판단돼요 (0.98)"
    skipped = screen([("날씨 어떰", "casual")], [], set(), None, "검사 건너뜀")
    assert skipped == [Proposal("날씨 어떰", "casual", "ok", "검사 건너뜀")]
    assert skipped[0].to_dict() == {"text": "날씨 어떰", "type": "casual", "status": "ok", "detail": "검사 건너뜀"}


def test_messages_show_at_most_twenty_examples_and_the_count() -> None:
    system, user = intent_messages("날씨 문의", [f"예문 {i}" for i in range(30)], 15)
    assert "JSON" in system["content"] and "예문 19" in user["content"] and "예문 20" not in user["content"]
    assert "새 문장 15개" in user["content"]
    _, faq = faq_messages("환불 되나요", "7일 안에 돼요", ["환불돼요?"], 10)
    assert "[질문] 환불 되나요\n- 환불돼요?" in faq["content"] and "[답] 7일 안에 돼요" in faq["content"]


def test_propose_asks_for_varied_sentences_and_refuses_an_empty_answer() -> None:
    llm = FakeLlm(reply([{"text": "날시", "type": "typo"}]))
    client = LlmClient(LlmConfig("http://llm.local", "m"), llm)
    assert propose(client, intent_messages("날씨", ["날씨"], 5)) == [("날시", "typo")]
    body, _ = llm.requests[0]
    assert (body["temperature"], body["max_tokens"]) == (augment.TEMPERATURE, augment.MAX_TOKENS)
    empty = LlmClient(LlmConfig("http://llm.local", "m"), FakeLlm(reply([])))
    with pytest.raises(GenerationError):
        propose(empty, intent_messages("날씨", ["날씨"], 5))


# ---------------------------------------------------------------- FAQ similar questions


@pytest.mark.parametrize("similar", [["   "], ["가" * 301], ["구독 환불은  어떻게 하나요"], ["환불?", "환불?"],
                                     [f"q{i}" for i in range(MAX_SIMILAR + 1)]])
def test_similar_questions_are_checked(similar) -> None:
    with pytest.raises(ValidationError):
        FaqEntry(id="refund", question="구독 환불은 어떻게 하나요", answer="7일 안에", similar=similar)


def kb_with_similar():
    kb = json.loads(json.dumps(KB))
    kb["faq"][0]["similar"] = ["구독 환블 어케 해요", "결제 취소하고 돈 돌려받기"]
    return kb


def test_a_similar_question_finds_its_faq_and_the_judge_reads_it_with_the_answer() -> None:
    index = KnowledgeIndex.build(kb_with_similar(), EMB)
    assert index.faq_vectors.shape[0] == 4  # 2 questions + 2 similar
    assert index.faq[0].keys == ("구독 환불은 어떻게 하나요", "구독 환블 어케 해요", "결제 취소하고 돈 돌려받기")
    seen = []

    class Recording(CosineJudge):
        def score(self, query, candidates):
            seen.extend(c.judge_input for c in candidates)
            return super().score(query, candidates)

    q = "결제 취소하고 돈 돌려받기"
    result = index.search(q, EMB.encode([normalize(q)])[0], Recording(0.5))
    assert result.answers[0].entry.id == "refund"
    assert seen[0] == "결제 취소하고 돈 돌려받기\n결제 후 7일 안에 설정 > 구독에서 환불할 수 있어요."
    assert index.nearest_faq(EMB.encode([normalize(q)])[0])[0].id == "refund"
    scores = index.faq_scores(EMB.encode([normalize(q)])[0])
    assert set(scores) == {"refund", "widget"} and scores["refund"] == pytest.approx(1.0, abs=1e-5)
    assert index.exact("결제 취소하고 돈 돌려받기") is None  # buttons send the question itself only


def test_without_similar_questions_search_is_as_before() -> None:
    plain = KnowledgeIndex.build(KB, EMB)
    q = "구독 환불 어떻게 해요"
    hit = plain.search(q, EMB.encode([normalize(q)])[0], CosineJudge(0.5)).candidates[0]
    assert plain.faq_vectors.shape[0] == 2 and hit.entry.id == "refund"
    assert plain._faq_candidates(EMB.encode([normalize(q)])[0])[0].judge_input == plain.faq[0].judge_text


def test_an_index_with_similar_questions_round_trips_and_an_older_format_is_built_again(tmp_path) -> None:
    index = KnowledgeIndex.build(kb_with_similar(), EMB)
    index.save(tmp_path)
    loaded = KnowledgeIndex.load(tmp_path, index.version, EMB)
    assert loaded.faq == index.faq and np.array_equal(loaded.faq_vectors, index.faq_vectors)
    meta = json.loads((tmp_path / "index.json").read_text())
    del meta["format"]  # an index from before E-H
    (tmp_path / "index.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format"):
        KnowledgeIndex.load(tmp_path, index.version, EMB)


def test_empty_index_has_no_faq_scores() -> None:
    assert KnowledgeIndex.build({}, EMB).faq_scores(EMB.encode(["x"])[0]) == {}


# ---------------------------------------------------------------- API


@pytest.fixture
def make_host(fake_embedder, tmp_path, trained_sample):
    def make(llm=None):
        shutil.copytree(trained_sample, tmp_path, dirs_exist_ok=True)
        gen = Generator(LlmClient(LlmConfig("http://llm.local", "test-model", 3.5), llm)) if llm else None
        h = BotHost(InMemoryRepository(), InMemoryStore(), fake_embedder, tmp_path, actions=synthetic_actions,
                    keep=1, min_f1=0.0, judge=CosineJudge(0.5), generator=gen)
        h.seed(SAMPLE)
        return h
    return make


def test_block_proposals_are_screened_against_the_draft_and_the_deployed_model(make_host) -> None:
    llm = FakeLlm(reply([{"text": "{location} 날시 어때요?", "type": "typo"}, {"text": "{location} 날씨 어때요?", "type": "paraphrase"},
                         {"text": "안녕하세요", "type": "casual"}, {"text": "{도시} 날씨", "type": "synonym"}]))
    client = TestClient(create_app(make_host(llm)))
    r = client.post(f"{B}/augment", json={"kind": "intent", "id": "weather", "count": 4})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["kind"], body["id"], body["model"]) == ("intent", "weather", "test-model") and body["ms"] >= 0
    assert [(i["status"], i["type"]) for i in body["items"]] == [
        ("ok", "typo"), ("duplicate", "paraphrase"), ("conflict", "casual"), ("invalid", "synonym")]
    assert "‘인사’ 블록" in body["items"][2]["detail"]
    sent, timeout = llm.requests[0]
    assert timeout == augment.TIMEOUT and "[블록] 날씨 문의" in sent["messages"][1]["content"]


def test_faq_proposals_are_checked_against_the_newest_index(make_host) -> None:
    llm = FakeLlm(reply([{"text": "구독 환블 어케 해요", "type": "typo"}, {"text": "위젯은 어떻게 추가하나요", "type": "paraphrase"},
                         {"text": "구독 환불은 어떻게 하나요", "type": "paraphrase"}]))
    host = make_host(llm)
    client = TestClient(create_app(host))
    host.save_kb_draft("sample", KB)
    first = client.post(f"{B}/augment", json={"kind": "faq", "id": "refund"}).json()
    assert [i["status"] for i in first["items"]] == ["ok", "ok", "duplicate"]  # no index yet: not checked
    assert "보지 않았어요" in first["items"][0]["detail"]
    host.build_kb("sample")
    checked = client.post(f"{B}/augment", json={"kind": "faq", "id": "refund"}).json()
    assert [i["status"] for i in checked["items"]] == ["ok", "conflict", "duplicate"]
    assert checked["items"][1]["detail"] == "‘위젯은 어떻게 추가하나요’ FAQ에 더 가까워요"


def test_augment_refuses_without_an_llm_an_unknown_target_bad_input_and_a_failed_llm(make_host) -> None:
    assert TestClient(create_app(make_host())).post(f"{B}/augment", json={"kind": "intent", "id": "weather"}).status_code == 409
    client = TestClient(create_app(make_host(FakeLlm(b"not json at all {"))))
    assert client.post(f"{B}/augment", json={"kind": "intent", "id": "nobody"}).status_code == 404
    assert client.post(f"{B}/augment", json={"kind": "faq", "id": "refund"}).status_code == 404  # no knowledge base
    for bad in ({"kind": "entity", "id": "x"}, {"kind": "intent", "id": "weather", "count": 0},
                {"kind": "intent", "id": "weather", "count": 41}, {"kind": "intent", "id": "a b"}):
        assert client.post(f"{B}/augment", json=bad).status_code == 422
    failed = client.post(f"{B}/augment", json={"kind": "intent", "id": "weather"})
    assert failed.status_code == 502 and "did not answer" in failed.json()["detail"]
    down = TestClient(create_app(make_host(FakeLlm(TimeoutError()))))
    assert "timeout" in down.post(f"{B}/augment", json={"kind": "intent", "id": "weather"}).json()["detail"]


def test_augment_needs_the_admin_token_and_a_bot_not_being_deleted(make_host) -> None:
    host = make_host(FakeLlm(reply([{"text": "x", "type": "typo"}])))
    assert TestClient(create_app(host, "secret-token-for-tests")).post(
        f"{B}/augment", json={"kind": "intent", "id": "weather"}).status_code == 401
    host.delete_bot("sample")
    assert TestClient(create_app(host)).post(f"{B}/augment", json={"kind": "intent", "id": "weather"}).status_code == 409


def test_a_conversation_finds_an_faq_by_its_similar_question(make_host) -> None:
    host = make_host()
    host.save_kb_draft("sample", kb_with_similar())
    info = host.build_kb("sample")
    host.deploy("sample", host.bots()["sample"], "", info.version)
    knowledge = Knowledge(host.kb_index("sample", info.version), CosineJudge(0.5), host.embedder)
    _, answer = knowledge.consult("결제 취소하고 돈 돌려받기", "fallback", 0.1, None)
    assert answer is not None and answer.items == ("refund",)
