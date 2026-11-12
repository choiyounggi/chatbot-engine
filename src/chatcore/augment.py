"""LLM-written block examples and FAQ similar questions (docs/engine-core-plan.md "빌더 단계 E-H").

An LLM proposes other ways to say a block's examples or ask an FAQ question — typos, casual wording,
paraphrases, synonyms. The engine screens them without an LLM: duplicates, invalid ones and ones that
look like another block or FAQ are marked, and a person adds the rest from the builder.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Any

from chatcore.generation import GenerationError, LlmClient
from chatcore.nlu.entities import bare
from chatcore.nlu.train import PLACEHOLDER

TYPES = ("typo", "casual", "paraphrase", "synonym")
STATUSES = ("ok", "duplicate", "invalid", "conflict")
DEFAULT_COUNT, MAX_COUNT = 20, 40
TIMEOUT = 90.0  # seconds for a whole proposal (both requests, retries included); chat answers have 3.5
# its own two threads: proposals queue behind each other, never in front of chat answers
POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="augment")
TEMPERATURE = 0.7  # varied sentences, not the same one twenty times
MAX_TOKENS = 2000  # forty short sentences as JSON fit with room to spare
MAX_TEXT = 300
SAMPLE = 20  # examples shown to the model
CONFLICT_PROB = 0.5  # a block example the deployed model gives to another block this surely is a conflict
FAQ_MARGIN = 0.02  # a similar question nearer another FAQ by more than this is a conflict

SYSTEM_PROMPT = """당신은 한국어 챗봇의 학습 데이터를 만드는 도우미입니다.
주어진 문장들과 **같은 뜻**으로, 실제 사용자가 채팅창에 입력할 만한 **새** 문장을 종류별로 만드세요.
- casual: 반말·구어체·줄임말 (예: 결제 취소 어떻게 해요 → 결제 취소 어케 함)
- paraphrase: 같은 뜻의 다른 문장
- synonym: 핵심 단어를 동의어·비슷한 말로 바꾼 문장 (예: 결제 → 계산, 취소 → 철회)
규칙:
1. 뜻을 바꾸거나 새로운 요구를 더하지 마세요.
2. 주어진 문장과 똑같은 문장, 서로 같은 문장은 쓰지 마세요.
3. 문장에 {이름} 같은 중괄호 표시가 있으면 그 표시를 글자 그대로 두세요.
4. 다른 말 없이 JSON만 쓰세요: {"casual": [...], "paraphrase": [...], "synonym": [...]}"""

# a model rewrites whole words easily but rarely misspells on request: typos get their own task, one
# sentence at a time, with the kinds spelled out
TYPO_PROMPT = """당신은 오타 생성기입니다.
문장마다, 사람이 휴대폰으로 급하게 칠 때 실제로 나는 오타를 한 군데 넣은 문장을 2개씩 만드세요.
오타 종류: 모음 혼동(ㅐ↔ㅔ, ㅗ↔ㅜ, ㅓ↔ㅏ), 받침을 빠뜨리거나 틀림, 된소리↔예사소리,
이웃 자판(ㄱ↔ㄷ, ㅅ↔ㅆ 등), 띄어쓰기를 붙이거나 띄움, 글자 순서 바뀜.
예) 주문 확인해줘 → ["주믄 확인해줘", "주문 확인해저"] / 배송 언제 와요 → ["배숑 언제 와요", "배송 언제와요"]
원래 문장과 똑같으면 안 됩니다. {이름} 같은 중괄호 표시는 그대로 두세요. 다른 말 없이 JSON만: {"typo": ["...", ...]}"""


@dataclass(frozen=True)
class Proposal:
    text: str
    type: str  # one of TYPES
    status: str  # one of STATUSES
    detail: str = ""  # why it is not "ok", or a note (e.g. the conflict check was skipped)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


Messages = list[dict[str, str]]


def split_count(count: int) -> dict[str, int]:
    """How many of each type to ask for: `count` spread over TYPES, typos first."""
    return {t: count // len(TYPES) + (1 if i < count % len(TYPES) else 0) for i, t in enumerate(TYPES)}


def _tasks(sentences: Sequence[str], context: str, request: str, count: int) -> list[Messages]:
    """Two requests: typos of the given sentences (two each), and the other types."""
    n = split_count(count)
    tasks: list[Messages] = []
    if n["typo"]:
        picked = list(sentences)[:max(1, (n["typo"] + 1) // 2)]
        tasks.append([{"role": "system", "content": TYPO_PROMPT},
                      {"role": "user", "content": "[문장]\n" + "\n".join(f"- {s}" for s in picked)}])
    rest = ", ".join(f"{t} {k}개" for t, k in n.items() if t != "typo" and k)
    if rest:
        tasks.append([{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"{context}\n\n{request}: {rest}."}])
    return tasks


def intent_messages(label: str, examples: Sequence[str], count: int) -> list[Messages]:
    shown = "\n".join(f"- {e}" for e in list(examples)[:SAMPLE])
    return _tasks(examples, f"[블록] {label}\n[예문]\n{shown}", "이 블록으로 가야 하는 새 문장을 만드세요", count)


def faq_messages(question: str, answer: str, similar: Sequence[str], count: int) -> list[Messages]:
    known = "".join(f"\n- {q}" for q in list(similar)[:SAMPLE])
    return _tasks([question, *similar], f"[질문] {question}{known}\n[답] {answer}",
                  "이 답을 찾는 사용자가 물을 만한 새 질문을 만드세요", count)


def _text_of(item: Any) -> tuple[str, str] | None:
    if isinstance(item, str):
        return item, "paraphrase"
    if isinstance(item, dict) and isinstance(item.get("text"), str):
        kind = item.get("type")
        return item["text"], kind if kind in TYPES else "paraphrase"
    return None


_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


_JSONISH = re.compile(r'[{\[]\s*"')


def _items(data: Any) -> list[tuple[str, str]] | None:
    if isinstance(data, dict) and any(isinstance(data.get(t), list) for t in TYPES):  # grouped by type
        return [(x, t) for t in TYPES if isinstance(v := data.get(t), list) for x in v if isinstance(x, str)]
    items = data.get("items") if isinstance(data, dict) else data
    if isinstance(items, list):
        return [t for t in (_text_of(x) for x in items) if t is not None]
    return None


def parse_items(raw: str) -> list[tuple[str, str]]:
    """(text, type) from the model's answer: the first JSON value in it (the object asked for or a bare
    list; text after it ignored), or — only when the answer is no JSON at all — one sentence per line
    (bullets and numbers removed). Broken JSON gives nothing (asked again), never its fragments."""
    decoder = json.JSONDecoder()
    for match in _JSONISH.finditer(raw):
        try:
            data, _ = decoder.raw_decode(raw, match.start())
        except (ValueError, RecursionError):
            continue
        found = _items(data)
        if found is not None:
            return found
    if _JSONISH.search(raw):
        return []
    lines = (_BULLET.sub("", line).strip().strip('"') for line in raw.splitlines())
    return [(line, "paraphrase") for line in lines if line and not line.startswith(("{", "}", "[", "]", "```"))]


@dataclass(frozen=True)
class Unchecked:
    note: str  # why this sentence could not be checked for a conflict


Conflict = Callable[[str], "str | Unchecked | None"]  # a sentence -> why it looks like something else, or None


def screen(items: Sequence[tuple[str, str]], existing: Sequence[str], placeholders: set[str],
           conflict: Conflict | None, skipped: str = "") -> list[Proposal]:
    """Mark each proposal (deterministic, no LLM). `placeholders`: the {names} the originals use;
    `conflict` None: the check was skipped (`skipped` says why, on every ok proposal)."""
    seen = {bare(t) for t in existing}
    out: list[Proposal] = []
    for raw, kind in items:
        text = " ".join(raw.split())
        names = set(PLACEHOLDER.findall(text))
        if not bare(text) or len(text) > MAX_TEXT:
            out.append(Proposal(text, kind, "invalid", f"글자가 없거나 {MAX_TEXT}자를 넘어요"))
        elif "{" in PLACEHOLDER.sub("", text) or "}" in PLACEHOLDER.sub("", text) or not names <= placeholders:
            out.append(Proposal(text, kind, "invalid", "원래 예문에 없는 {자리} 표시가 있어요"))
        elif bare(text) in seen:
            out.append(Proposal(text, kind, "duplicate", "이미 있는 문장이에요"))
        else:
            why = conflict(text) if conflict is not None else None
            if isinstance(why, Unchecked):
                out.append(Proposal(text, kind, "ok", why.note))
            else:
                out.append(Proposal(text, kind, "conflict", why) if why else Proposal(text, kind, "ok", skipped))
        seen.add(bare(text))
    return out


def ask(client: LlmClient, messages: Messages, tries: int = 2, deadline: float | None = None
        ) -> list[tuple[str, str]]:
    """One request's sentences, asked again once when an answer has none (sampling varies), each call
    within what is left before `deadline` (time.monotonic()). GenerationError when it does not answer,
    runs out of time, or never answers anything usable."""
    for _ in range(tries):
        left = client.config.timeout if deadline is None else deadline - time.monotonic()
        if left <= 0:
            raise GenerationError("timeout", f"no proposals within {TIMEOUT:g}s")
        config = replace(client.config, timeout=min(left, client.config.timeout))
        timed = LlmClient(config, client.transport, client.pool)
        items = parse_items(timed.chat(messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE))
        if items:
            return items
    raise GenerationError("error", "the LLM's answer has no sentences")


def propose(client: LlmClient, tasks: Sequence[Messages], seconds: float = TIMEOUT) -> list[tuple[str, str]]:
    """Every task's sentences, in order, all within `seconds`."""
    deadline = time.monotonic() + seconds
    return [item for messages in tasks for item in ask(client, messages, deadline=deadline)]
