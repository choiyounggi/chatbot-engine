"""E-F: generative answers — prompt, OpenAI-compatible client, grounding check, generator."""

import http.client
import json
import socket
import threading
import time
import urllib.error

import pytest

from chatcore.generation import (
    MAX_ANSWER_CHARS,
    NO_ANSWER,
    SYSTEM_PROMPT,
    GenerationError,
    Generator,
    LlmClient,
    LlmConfig,
    build_messages,
    check,
    numbers,
    post,
    strip_thinking,
)
from chatcore.knowledge import Entry

FAQ = Entry("refund", "faq", "연간 구독 환불 되나요?", "결제일로부터 14일 이내에는 전액 환불돼요. 금액은 39,000원이에요.")
DOC = Entry("policy#0", "doc", "정책 - 환불 (1/2)", "월 구독은 해지해도 다음 결제일까지 쓸 수 있습니다.", "정책.md", "정책 - 환불")


def reply(content):
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()


class Transport:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, json.loads(body), timeout))
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a


def client(*answers, key=None):
    t = Transport(*answers)
    return LlmClient(LlmConfig("http://llm.local/", "m1", timeout=5, api_key=key), t), t


# ---------------------------------------------------------------- prompt


def test_messages_number_the_passages_and_put_the_question_last() -> None:
    system, user = build_messages("환불 돼요?", [FAQ, DOC])
    assert system == {"role": "system", "content": SYSTEM_PROMPT} and NO_ANSWER in SYSTEM_PROMPT
    assert user["role"] == "user"
    assert user["content"] == ("[자료 1] 연간 구독 환불 되나요?\n결제일로부터 14일 이내에는 전액 환불돼요. 금액은 39,000원이에요.\n\n"
                               "[자료 2] 정책 - 환불\n월 구독은 해지해도 다음 결제일까지 쓸 수 있습니다.\n\n[질문] 환불 돼요?")


# ---------------------------------------------------------------- check


def test_numbers_drop_thousands_separators_but_keep_decimals() -> None:
    assert numbers("39,000원, 1.5배, 14일, 2026년") == {"39000", "1.5", "14", "2026"}
    assert numbers("숫자 없음") == set()


@pytest.mark.parametrize("answer, outcome", [
    ("결제일로부터 14일 이내라면 39000원 전액 환불돼요.", "generated"),  # 39000 = 39,000 in the passage
    ("14일이 지나면 7일 치를 빼고 돌려드려요.", "unsupported_number"),  # 7 is nowhere in the passages
    ("NO_ANSWER", "no_answer"),
    ("죄송해요. NO_ANSWER", "no_answer"),
    ("   ", "empty"),
    ("가" * (MAX_ANSWER_CHARS + 1), "too_long"),
    ("가" * MAX_ANSWER_CHARS, "generated"),  # exactly at the limit
])
def test_check(answer, outcome) -> None:
    assert check(answer, [FAQ, DOC]) == outcome


def test_check_without_passages_allows_no_numbers() -> None:
    assert check("1번", []) == "unsupported_number" and check("숫자 없는 답", []) == "generated"


# ---------------------------------------------------------------- client


def test_client_posts_an_openai_style_request_and_strips_thinking() -> None:
    c, t = client(reply("<think>음…</think>\n 답이에요. "), key="sk-test-not-real")
    assert c.chat([{"role": "user", "content": "q"}]) == "답이에요."
    url, headers, body, timeout = t.calls[0]
    assert url == "http://llm.local/v1/chat/completions" and timeout == 5
    assert headers == {"Content-Type": "application/json", "Authorization": "Bearer sk-test-not-real"}
    assert body == {"model": "m1", "messages": [{"role": "user", "content": "q"}], "temperature": 0, "max_tokens": 320,
                    "stream": False}


def test_client_sends_no_authorization_without_a_key() -> None:
    c, t = client(reply("ok"))
    c.chat([])
    assert "Authorization" not in t.calls[0][1]


@pytest.mark.parametrize("failure, outcome", [
    (TimeoutError(), "timeout"),
    (urllib.error.URLError(TimeoutError()), "timeout"),
    (urllib.error.URLError(ConnectionRefusedError()), "error"),
    (OSError("reset"), "error"),
    (b"not json", "error"),
    (json.dumps({"choices": []}).encode(), "error"),
    (json.dumps({"choices": [{"message": {"content": None}}]}).encode(), "error"),
    (b"\xff\xfe", "error"),
    (http.client.BadStatusLine("garbage"), "error"),  # not an HTTP server
    (http.client.IncompleteRead(b"ab"), "error"),  # a chunked reply cut short
    (ValueError("the reply is too large"), "error"),
])
def test_client_turns_every_failure_into_a_generation_error(failure, outcome) -> None:
    c, _ = client(failure)
    with pytest.raises(GenerationError) as exc:
        c.chat([])
    assert exc.value.outcome == outcome


# ---------------------------------------------------------------- generator


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.25
        return self.t


def test_generator_keeps_a_checked_answer_and_uses_at_most_depth_passages() -> None:
    c, t = client(reply("14일 이내면 전액 환불돼요."))
    g = Generator(c, depth=1, clock=Clock()).generate("환불 돼요?", [FAQ, DOC])
    assert (g.text, g.outcome, g.used, g.ms) == ("14일 이내면 전액 환불돼요.", "generated", ("refund",), 250)
    assert "[자료 2]" not in t.calls[0][2]["messages"][1]["content"]
    assert g.to_log("m1") == {"model": "m1", "ms": 250, "used": ["refund"], "outcome": "generated"}


def test_generator_rejects_but_keeps_the_raw_answer_for_logs() -> None:
    c, _ = client(reply("30일 안에 환불돼요."))
    g = Generator(c).generate("환불 돼요?", [FAQ, DOC])
    assert g.text is None and g.outcome == "unsupported_number" and g.raw == "30일 안에 환불돼요."


def test_generator_reports_a_failed_call_and_never_raises() -> None:
    c, _ = client(TimeoutError())
    g = Generator(c).generate("환불 돼요?", [FAQ])
    assert (g.text, g.outcome, g.used) == (None, "timeout", ("refund",))


def test_generator_without_passages_does_not_call_the_model_and_depth_must_be_positive() -> None:
    c, t = client()
    assert Generator(c).generate("q", []).outcome == "no_answer" and t.calls == []
    with pytest.raises(ValueError):
        Generator(c, depth=0)


# ---------------------------------------------------------------- thinking, deadline, the key


@pytest.mark.parametrize("content, answer", [
    ("<think>음…</think>답", "답"),
    ("자료에 7일이라고 있으니…</think>\n7일 안에 환불돼요.", "7일 안에 환불돼요."),  # the template opened the block
    ("<think>a</think><think>b</think> 답 ", "답"),
])
def test_thinking_is_removed(content, answer) -> None:
    assert strip_thinking(content) == answer


@pytest.mark.parametrize("content", ["<think>자료에 7일이라고", "답이에요 <think>그런데"])
def test_an_answer_still_thinking_is_an_error_never_text(content) -> None:
    c, _ = client(reply(content))
    with pytest.raises(GenerationError) as exc:
        c.chat([])
    assert exc.value.outcome == "error" and "thinking" in exc.value.detail


def test_the_timeout_bounds_the_whole_call_whatever_the_transport_does() -> None:
    def stuck(url, headers, body, timeout):
        time.sleep(1.5)
        return reply("늦은 답")

    c = LlmClient(LlmConfig("http://llm.local", "m1", timeout=0.2), stuck)
    start = time.monotonic()
    with pytest.raises(GenerationError) as exc:
        c.chat([])
    assert exc.value.outcome == "timeout" and time.monotonic() - start < 1.0


def test_the_api_key_is_not_in_the_config_repr() -> None:
    config = LlmConfig("http://llm.local", "m1", api_key="sk-test-not-real")
    assert "sk-test-not-real" not in repr(config) and config.api_key == "sk-test-not-real"


# ---------------------------------------------------------------- post (a real socket on 127.0.0.1)


@pytest.fixture
def server():
    """A one-connection TCP server: `server(handler)` runs handler(conn) and returns the URL."""
    socks = []

    def start(handler):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        socks.append(s)

        def serve():
            conn, _ = s.accept()
            with conn:
                conn.recv(65536)
                try:
                    handler(conn)
                except OSError:  # the client gave up
                    pass

        threading.Thread(target=serve, daemon=True).start()
        return f"http://127.0.0.1:{s.getsockname()[1]}/v1/chat/completions"

    yield start
    for s in socks:
        s.close()


def http_reply(body: bytes) -> bytes:
    return b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%b" % (len(body), body)


def test_post_returns_the_reply_body(server) -> None:
    url = server(lambda conn: conn.sendall(http_reply(reply("답"))))
    assert json.loads(post(url, {}, b"{}", 2))["choices"][0]["message"]["content"] == "답"


def test_post_gives_up_on_a_slow_reply_after_the_timeout(server) -> None:
    def drip(conn):
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n")
        for _ in range(20):
            time.sleep(0.15)
            conn.sendall(b"x")

    url = server(drip)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        post(url, {}, b"{}", 0.5)
    assert time.monotonic() - start < 1.5  # each byte comes within the socket timeout; the deadline still holds


def test_post_refuses_a_reply_that_is_too_large(server) -> None:
    url = server(lambda conn: conn.sendall(http_reply(b"x" * 1_000_001)))
    with pytest.raises(ValueError):
        post(url, {}, b"{}", 2)


def test_a_server_that_is_not_http_is_a_generation_error(server) -> None:
    url = server(lambda conn: conn.sendall(b"garbage\r\n\r\n"))
    c = LlmClient(LlmConfig(url.removesuffix("/v1/chat/completions"), "m1", timeout=2))
    with pytest.raises(GenerationError) as exc:
        c.chat([])
    assert exc.value.outcome == "error"
