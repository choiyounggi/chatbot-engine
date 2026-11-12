"""Generative answers over the knowledge base (docs/engine-core-plan.md "빌더 단계 E-F").

The knowledge base decides WHETHER to answer (retrieval + judge, E-D); an LLM only decides HOW to say
it, from the passages that passed. Any server speaking OpenAI-compatible Chat Completions works
(`POST {url}/v1/chat/completions`: mlx_lm.server, llama.cpp, Ollama, vLLM, LM Studio, gateways).
A generated answer is kept only when a deterministic check finds nothing it cannot back: no
"NO_ANSWER", not empty or too long, and every number in it appears in the passages. The check is about
numbers, not meaning: a flipped negation ("환불돼요" for "환불되지 않아요"), a number word ("일주일"),
or a number taken from the FAQ question in the wrong sense passes it; a number the user said and the
model repeats ("3일 전에 샀는데" -> "3일이 지나서") fails it.

The server must answer without a thinking block (Qwen3: `enable_thinking: false` in the server's chat
template arguments); an answer still inside one is an error, never shown.
"""

from __future__ import annotations

import http.client
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from chatcore.knowledge import Entry

NO_ANSWER = "NO_ANSWER"
MAX_ANSWER_CHARS = 600
MAX_TOKENS = 320
OUTCOMES = ("generated", "no_answer", "unsupported_number", "too_long", "empty", "timeout", "error")
GENERATION_DEPTH = 1  # passages given to the model (docs/benchmarks/rag.md: 1 is as good as 3 and faster)
LLM_TIMEOUT = 3.5  # seconds for the whole call: Kakao waits 5 s for a skill, and the turn also needs NLU and the judge
MAX_REPLY_BYTES = 1_000_000

SYSTEM_PROMPT = """당신은 고객 상담 챗봇입니다. 아래 [자료]에 있는 내용만으로 사용자의 [질문]에 답하세요.
규칙:
1. 자료에 없는 내용은 쓰지 마세요. 추측하지 마세요.
2. 숫자, 기간, 금액, 조건은 자료에 쓰인 그대로 옮기세요.
3. 한국어 존댓말로 2~4문장, 질문에 대한 답부터 말하세요.
4. 자료로 답할 수 없으면 다른 말 없이 NO_ANSWER 라고만 쓰세요.
5. "자료에 따르면", 자료 번호 같은 말은 쓰지 마세요."""

_THINK = re.compile(r"<think>.*?</think>", re.S)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


class GenerationError(Exception):
    def __init__(self, outcome: str, detail: str) -> None:  # outcome: "timeout" | "error" | "too_long" (cut off)
        super().__init__(f"{outcome}: {detail}")
        self.outcome, self.detail = outcome, detail


def passage_text(entry: Entry) -> str:
    return f"{entry.title}\n{entry.body}" if entry.kind == "faq" else f"{entry.heading or entry.title}\n{entry.body}"


def build_messages(question: str, passages: Sequence[Entry]) -> list[dict[str, str]]:
    sources = "\n\n".join(f"[자료 {i}] {passage_text(e)}" for i, e in enumerate(passages, 1))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{sources}\n\n[질문] {question}"}]


def numbers(text: str) -> set[str]:
    """Digit runs in a text, thousands separators removed ("3,900" -> "3900"; "1.5" stays)."""
    out = set()
    for n in _NUMBER.findall(text):
        out.add(n.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", n) else n)
    return out


def check(answer: str, passages: Sequence[Entry]) -> str:
    """The outcome of a generated answer: "generated" when it may be shown, else why not."""
    text = answer.strip()
    if not text:
        return "empty"
    if NO_ANSWER in text:
        return "no_answer"
    if len(text) > MAX_ANSWER_CHARS:
        return "too_long"
    allowed = set().union(*(numbers(passage_text(e)) for e in passages)) if passages else set()
    if numbers(text) - allowed:
        return "unsupported_number"
    return "generated"


Transport = Callable[[str, dict[str, str], bytes, float], bytes]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is an error: following it would send the API key to wherever it points."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    """One HTTP POST that gives up after `timeout` seconds in all (a server sending its reply slowly
    included) and follows no redirect. TimeoutError, OSError (urllib.error.HTTPError for a redirect or
    an error status), http.client.HTTPException or ValueError (reply too large)."""
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with _OPENER.open(request, timeout=timeout) as response:
        reply = bytearray()
        while chunk := response.read1(65536):
            reply += chunk
            if len(reply) > MAX_REPLY_BYTES:
                raise ValueError("the reply is too large")
            if time.monotonic() > deadline:
                raise TimeoutError
        return bytes(reply)


def strip_thinking(content: str) -> str:
    """The answer without a model's thinking: whole <think>…</think> blocks, or everything up to a
    </think> whose opening tag the chat template put in the prompt. GenerationError when a block is
    left open (the model ran out of tokens while thinking)."""
    text = _THINK.sub("", content)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        raise GenerationError("error", "the answer is still thinking: turn thinking off on the LLM server")
    return text.strip()


@dataclass(frozen=True)
class LlmConfig:
    url: str  # the server's base URL, e.g. http://127.0.0.1:8800
    model: str
    timeout: float = LLM_TIMEOUT  # seconds for the whole call
    api_key: str | None = field(default=None, repr=False)  # sent as a Bearer token (taken from an environment variable)


# a call that overruns its deadline is abandoned, not waited for (post() ends it soon after)
_CALLS = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm")


class LlmClient:
    """OpenAI-compatible Chat Completions, non-streaming, temperature 0, at most `timeout` seconds."""

    def __init__(self, config: LlmConfig, transport: Transport = post) -> None:
        self.config, self.transport = config, transport

    def chat(self, messages: list[dict[str, str]], max_tokens: int = MAX_TOKENS, temperature: float = 0) -> str:
        """The reply text (a model's <think> block removed). GenerationError on timeout / failure."""
        body = json.dumps({"model": self.config.model, "messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "stream": False}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        call = _CALLS.submit(self.transport, self.config.url.rstrip("/") + "/v1/chat/completions", headers, body,
                             self.config.timeout)
        try:
            raw = call.result(timeout=self.config.timeout)  # the deadline holds whatever the transport does
        except TimeoutError as exc:  # ours, or the transport's
            call.cancel()
            raise GenerationError("timeout", f"no answer within {self.config.timeout:g}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise GenerationError("timeout", f"no answer within {self.config.timeout:g}s") from exc
            raise GenerationError("error", f"cannot reach the LLM server: {exc.reason}") from exc
        except (OSError, ValueError, OverflowError, http.client.HTTPException) as exc:  # OverflowError: timeout=inf
            raise GenerationError("error", type(exc).__name__) from exc
        try:
            data: Any = json.loads(raw.decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            cut = data["choices"][0].get("finish_reason") == "length"
        except (UnicodeDecodeError, ValueError, KeyError, IndexError, TypeError, AttributeError, RecursionError) as exc:
            raise GenerationError("error", "the LLM server's answer is not a chat completion") from exc
        if not isinstance(content, str):
            raise GenerationError("error", "the LLM server's answer has no text")
        if cut:  # stopped at max_tokens: a sentence (or the model's thinking) cut off
            raise GenerationError("too_long", f"the answer was cut off at {max_tokens} tokens")
        return strip_thinking(content)


@dataclass(frozen=True)
class Generated:
    text: str | None  # the answer to show, or None (outcome says why)
    outcome: str  # one of OUTCOMES
    ms: int
    used: tuple[str, ...]  # entry ids given to the model
    raw: str = ""  # what the model said (for logs and measurement; never shown when rejected)

    def to_log(self, model: str) -> dict[str, Any]:
        return {"model": model, "ms": self.ms, "used": list(self.used), "outcome": self.outcome}


class Generator:
    """Writes an answer from the top `depth` passages that passed the judge."""

    def __init__(self, client: LlmClient, depth: int = 3, clock: Callable[[], float] = time.monotonic) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.client, self.depth, self.clock = client, depth, clock

    @property
    def model(self) -> str:
        return self.client.config.model

    def generate(self, question: str, passages: Sequence[Entry]) -> Generated:
        used = list(passages)[: self.depth]
        start = self.clock()
        if not used:
            return Generated(None, "no_answer", 0, ())
        try:
            raw = self.client.chat(build_messages(question, used))
        except GenerationError as exc:
            return Generated(None, exc.outcome, self._ms(start), tuple(e.id for e in used))
        outcome = check(raw, used)
        return Generated(raw.strip() if outcome == "generated" else None, outcome, self._ms(start),
                         tuple(e.id for e in used), raw)

    def _ms(self, start: float) -> int:
        return max(0, round((self.clock() - start) * 1000))
