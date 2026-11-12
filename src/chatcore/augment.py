"""LLM-written block examples and FAQ similar questions (docs/engine-core-plan.md "빌더 단계 E-H").

An LLM proposes other ways to say a block's examples or ask an FAQ question — typos, casual wording,
paraphrases, synonyms. The engine screens them without an LLM: duplicates, invalid ones and ones that
look like another block or FAQ are marked, and a person adds the rest from the builder.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from chatcore.generation import GenerationError, LlmClient
from chatcore.nlu.entities import bare
from chatcore.nlu.train import PLACEHOLDER

TYPES = ("typo", "casual", "paraphrase", "synonym")
STATUSES = ("ok", "duplicate", "invalid", "conflict")
DEFAULT_COUNT, MAX_COUNT = 20, 40
TIMEOUT = 90.0  # seconds: a list of sentences takes far longer than a chat answer
TEMPERATURE = 0.7  # varied sentences, not the same one twenty times
MAX_TOKENS = 2000  # forty short sentences as JSON fit with room to spare
MAX_TEXT = 300
SAMPLE = 20  # examples shown to the model
CONFLICT_PROB = 0.5  # a block example the deployed model gives to another block this surely is a conflict
FAQ_MARGIN = 0.02  # a similar question nearer another FAQ by more than this is a conflict

SYSTEM_PROMPT = """당신은 한국어 챗봇의 학습 데이터를 만드는 도우미입니다.
주어진 문장들과 **같은 뜻**으로, 실제 사용자가 입력할 만한 다른 문장을 만드세요.
- typo: 오타(모음·받침 실수, 된소리·예사소리 혼동, 띄어쓰기 실수, 자판 실수)
- casual: 구어체·반말·줄임말·이모티콘 없는 채팅 말투
- paraphrase: 같은 뜻의 다른 표현
- synonym: 핵심 단어를 동의어·비슷한 말로 바꾼 문장
규칙:
1. 뜻을 바꾸거나 새로운 요구를 더하지 마세요.
2. 주어진 문장과 똑같은 문장은 쓰지 마세요.
3. 문장에 {이름} 같은 중괄호 표시가 있으면 그 표시를 글자 그대로 두세요.
4. 네 종류를 고르게 섞으세요.
5. 다른 말 없이 JSON만 쓰세요: {"items": [{"text": "...", "type": "typo"}, ...]}"""


@dataclass(frozen=True)
class Proposal:
    text: str
    type: str  # one of TYPES
    status: str  # one of STATUSES
    detail: str = ""  # why it is not "ok", or a note (e.g. the conflict check was skipped)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def intent_messages(label: str, examples: Sequence[str], count: int) -> list[dict[str, str]]:
    shown = "\n".join(f"- {e}" for e in list(examples)[:SAMPLE])
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"[블록] {label}\n[예문]\n{shown}\n\n이 블록으로 가야 하는 새 문장 {count}개를 만드세요."}]


def faq_messages(question: str, answer: str, similar: Sequence[str], count: int) -> list[dict[str, str]]:
    known = "".join(f"\n- {q}" for q in list(similar)[:SAMPLE])
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[질문] {question}{known}\n[답] {answer}\n\n"
                                        f"이 답을 찾는 사용자가 물을 만한 새 질문 {count}개를 만드세요."}]


def _text_of(item: Any) -> tuple[str, str] | None:
    if isinstance(item, str):
        return item, "paraphrase"
    if isinstance(item, dict) and isinstance(item.get("text"), str):
        kind = item.get("type")
        return item["text"], kind if kind in TYPES else "paraphrase"
    return None


_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def parse_items(raw: str) -> list[tuple[str, str]]:
    """(text, type) from the model's answer: the JSON asked for, a bare JSON list, or failing both,
    one sentence per line (bullets and numbers removed)."""
    for start, end in (("{", "}"), ("[", "]")):
        i, j = raw.find(start), raw.rfind(end)
        if i == -1 or j <= i:
            continue
        try:
            data: Any = json.loads(raw[i:j + 1])
        except ValueError:
            continue
        items = data.get("items") if isinstance(data, dict) else data
        if isinstance(items, list):
            return [t for t in (_text_of(x) for x in items) if t is not None]
    lines = (_BULLET.sub("", line).strip().strip('"') for line in raw.splitlines())
    return [(line, "paraphrase") for line in lines if line and not line.startswith(("{", "}", "[", "]", "```"))]


Conflict = Callable[[str], str | None]  # a sentence -> why it looks like something else, or None


def screen(items: Sequence[tuple[str, str]], existing: Sequence[str], placeholders: set[str],
           conflict: Conflict | None, skipped: str = "") -> list[Proposal]:
    """Mark each proposal (deterministic, no LLM). `placeholders`: the {names} the originals use;
    `conflict` None: the check was skipped (`skipped` says why, on every ok proposal)."""
    seen = {bare(t) for t in existing}
    out: list[Proposal] = []
    for raw, kind in items:
        text = " ".join(raw.split())
        names = set(PLACEHOLDER.findall(text))
        if not text or len(text) > MAX_TEXT:
            out.append(Proposal(text, kind, "invalid", f"1~{MAX_TEXT}자가 아니에요"))
        elif "{" in PLACEHOLDER.sub("", text) or "}" in PLACEHOLDER.sub("", text) or not names <= placeholders:
            out.append(Proposal(text, kind, "invalid", "원래 예문에 없는 {자리} 표시가 있어요"))
        elif bare(text) in seen:
            out.append(Proposal(text, kind, "duplicate", "이미 있는 문장이에요"))
        else:
            why = conflict(text) if conflict is not None else None
            out.append(Proposal(text, kind, "conflict", why) if why else Proposal(text, kind, "ok", skipped))
        seen.add(bare(text))
    return out


def propose(client: LlmClient, messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    """The model's sentences. GenerationError when it does not answer or answers nothing usable."""
    raw = client.chat(messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    items = parse_items(raw)
    if not items:
        raise GenerationError("error", "the LLM's answer has no sentences")
    return items
