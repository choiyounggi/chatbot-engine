"""Knowledge base (지식베이스): FAQ entries and documents a bot answers from when no block fits
(docs/engine-core-plan.md "빌더 단계 E-D"; measured in docs/benchmarks/kb.md).

- A bot has one knowledge base, drafted and versioned apart from the bot document (a version is
  the content hash). Its index lives in files next to the bot's models.
- FAQ questions are encoded like user sentences (the NLU's prompt); document chunks as passages
  (no prompt). Candidates are ranked by cosine in each pool, and a judge decides whether the best
  of the top JUDGE_DEPTH per pool answers the question: the cross-encoder in production, the cosine
  in hashing mode (tests, demos).
- A user sentence equal to an entry's title (an FAQ question, or a chunk's "document - section",
  which is what a suggestion button sends) is answered with that entry directly.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from chatcore.actions.builtin import NOT_FOUND
from chatcore.bot.loader import KAKAO_LABEL
from chatcore.dialogue.values import CURRENT_TURN, ActionFn
from chatcore.nlu.entities import bare, normalize
from chatcore.nlu.model import Embedder, encode_passages

CHUNK_MIN, CHUNK_MAX = 300, 500  # characters (measured: bench/kb_eval.py)
OVERLAP_MAX = 150  # the previous chunk's last sentence is repeated when at most this long
JUDGE_DEPTH = 3  # per pool
SUGGESTIONS = 3  # 답안 추천형: at most this many buttons
OVERRIDE_MARGIN = 0.05  # an FAQ beats a sure block only by this much over its nearest example
CROSS_ENCODER_THRESHOLD = 0.11  # chosen on the dev half (docs/benchmarks/kb.md)
COSINE_THRESHOLD = 0.5  # hashing mode only (surface overlap; never used to serve real users)
MAX_FAQ, MAX_DOCUMENTS, MAX_DOCUMENT_CHARS = 2000, 200, 200_000
ID = r"^[A-Za-z0-9_-]{1,64}$"
MAX_PDF_PAGES = 500
CHUNKER = 3  # stored with an index: a different chunker means the index is stale (built again)
FloatArray = NDArray[np.float32]
# a rejected generation shows the passage (docs/benchmarks/rag.md: chosen on the dev half)
ON_REJECT: Literal["extract", "fallback"] = "extract"
LLM_FAILURES = ("timeout", "error")  # generation outcomes where the LLM did not answer (chatcore.generation)

if TYPE_CHECKING:  # chatcore.generation imports this module
    from chatcore.generation import Generator


class KnowledgeError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


# ---------------------------------------------------------------- document


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FaqEntry(_Model):
    id: str = Field(pattern=ID)
    question: str = Field(min_length=1, max_length=300)
    answer: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _not_blank(self) -> FaqEntry:
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("an FAQ question and answer must not be blank")
        return self


class KbDocument(_Model):
    id: str = Field(pattern=ID)
    name: str = Field(min_length=1, max_length=200)  # shown as the source of an answer
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)  # plain text or Markdown


class KbSettings(_Model):
    mode: Literal["instant", "suggest"] = "instant"  # 즉문즉답형 / 답안 추천형
    suggest_text: str = Field("이런 답을 찾았어요. 골라 주세요.", min_length=1, max_length=200)
    generate: bool = False  # 즉문즉답형 answers written by the server's LLM from the passing passages (E-F)


class KnowledgeBase(_Model):
    faq: list[FaqEntry] = Field(default_factory=list, max_length=MAX_FAQ)
    documents: list[KbDocument] = Field(default_factory=list, max_length=MAX_DOCUMENTS)
    settings: KbSettings = Field(default_factory=KbSettings)

    @model_validator(mode="after")
    def _unique_ids(self) -> KnowledgeBase:
        for what, ids in (("FAQ", [f.id for f in self.faq]), ("document", [d.id for d in self.documents])):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            if dup:
                raise ValueError(f"duplicate {what} ids: {dup}")
        return self


def parse_kb(document: Any) -> KnowledgeBase:
    """KnowledgeError listing every problem when the document is not a valid knowledge base."""
    if not isinstance(document, dict):
        raise KnowledgeError(["the knowledge base must be an object"])
    try:
        return KnowledgeBase.model_validate(document)
    except ValidationError as exc:
        raise KnowledgeError([f"{'.'.join(str(p) for p in e['loc']) or '(top)'}: {e['msg']}"
                              for e in exc.errors()]) from exc


def extract_text(data: bytes, content_type: str) -> str:
    """The text of an uploaded file: a PDF (text layer only, no OCR), or UTF-8 plain text / Markdown.
    KnowledgeError when it cannot be read, has no text, or is too long for one document."""
    if content_type == "application/pdf" or data.startswith(b"%PDF-"):
        text = _pdf_text(data)
    elif content_type in ("text/plain", "text/markdown", "text/x-markdown", ""):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise KnowledgeError(["the file is not UTF-8 text"]) from exc
    else:
        raise KnowledgeError([f"unsupported file type {content_type!r}: use PDF, plain text or Markdown"])
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")).strip()
    if not text:
        raise KnowledgeError(["no text found in the file (a scanned PDF has none)"])
    if len(text) > MAX_DOCUMENT_CHARS:
        raise KnowledgeError([f"the text is {len(text)} characters; one document holds at most {MAX_DOCUMENT_CHARS}"])
    return text


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed files warn per object
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise KnowledgeError(["the PDF is encrypted"])
        if len(reader.pages) > MAX_PDF_PAGES:
            raise KnowledgeError([f"the PDF has more than {MAX_PDF_PAGES} pages"])
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
    except KnowledgeError:
        raise
    except Exception as exc:  # untrusted input: any parser failure means "cannot read it"
        raise KnowledgeError(["the file is not a readable PDF"]) from exc


def kb_version(document: dict[str, Any]) -> str:
    """Content hash of the document as given (12 hex characters, like a bot version)."""
    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- entries


@dataclass(frozen=True)
class Entry:
    id: str  # FAQ id, or "<document id>#<n>"
    kind: str  # "faq" | "doc"
    title: str  # unique in an index: what a suggestion button shows and sends back
    body: str  # FAQ answer, or the chunk's sentences
    source: str = ""  # the document's name (chunks)
    heading: str = ""  # a chunk's "<document title> - <section>" (its title may add " (2/3)"); FAQ: unused
    section: str = ""  # a chunk's `##` section ("" before the first one); what its suggestion button says

    @property
    def judge_text(self) -> str:
        """What is embedded (chunks) and what the judge reads (as measured: heading + body)."""
        return f"{self.heading or self.title}\n{self.body}"


def sentences(paragraph: str) -> list[str]:
    return [s for s in (p.strip() for p in re.split(r"(?<=[.!?])\s+", paragraph.strip())) if s]


def chunk_document(doc_id: str, name: str, text: str) -> list[Entry]:
    """Whole sentences packed into CHUNK_MIN..CHUNK_MAX characters within one `##` section; a chunk
    repeats the previous one's last sentence when it is at most OVERLAP_MAX characters. A sentence
    longer than CHUNK_MAX is its own chunk (never cut). Headings are "<# title or name> - <section>";
    a section in several chunks titles them "<heading> (1/3)", "(2/3)"... so each can be chosen."""
    title = name
    sections: list[tuple[str, list[str]]] = []  # (section, its sentences)

    def flush(lines: list[str]) -> None:
        body = " ".join(ln.strip() for ln in lines).strip()
        if body:
            if not sections:
                sections.append(("", []))
            sections[-1][1].extend(sentences(body))
        lines.clear()

    for block in re.split(r"\n\s*\n", text):  # paragraphs; headings may also sit inside one
        pending: list[str] = []
        for ln in block.strip().splitlines():
            if ln.startswith("#"):
                flush(pending)
                if ln.startswith("# "):
                    title = ln[2:].strip() or name
                elif ln.startswith("## "):
                    sections.append((ln[3:].strip(), []))
            else:
                pending.append(ln)
        flush(pending)
    out: list[Entry] = []
    for section, sents in sections:
        head = f"{title} - {section}" if section else title
        bodies: list[str] = []
        current: list[str] = []
        for s in sents:
            if current and len(" ".join([*current, s])) > CHUNK_MAX and len(" ".join(current)) >= CHUNK_MIN // 2:
                bodies.append(" ".join(current))
                last = current[-1]
                current = [last] if len(last) <= OVERLAP_MAX and len(last) + len(s) < CHUNK_MAX else []
            current.append(s)
        if current:
            bodies.append(" ".join(current))
        for i, body in enumerate(bodies, 1):
            shown = head if len(bodies) == 1 else f"{head} ({i}/{len(bodies)})"
            out.append(Entry(f"{doc_id}#{len(out)}", "doc", shown, body, name, head, section))
    return out


def entries(kb: KnowledgeBase) -> tuple[list[Entry], list[Entry]]:
    """(FAQ entries, document chunks)."""
    faq = [Entry(f.id, "faq", " ".join(f.question.split()), f.answer.strip()) for f in kb.faq]
    chunks = [c for d in kb.documents for c in chunk_document(d.id, d.name, d.text)]
    seen: dict[str, int] = {}
    for e in [*faq, *chunks]:
        seen[bare(e.title)] = seen.get(bare(e.title), 0) + 1
    # two documents with the same name and section: tell their chunks apart by document id
    chunks = [replace(c, title=f"{c.title} · {c.id.split('#')[0]}") if seen[bare(c.title)] > 1 else c for c in chunks]
    return faq, chunks


# ---------------------------------------------------------------- judging


@dataclass(frozen=True)
class Candidate:
    entry: Entry
    cosine: float


class Judge(Protocol):
    name: str
    threshold: float  # a candidate scoring at least this answers

    def score(self, query: str, candidates: Sequence[Candidate]) -> NDArray[np.float64]:
        """One score per candidate (higher = more likely to answer the query)."""
        ...


class CosineJudge:
    """The candidate's cosine is its score (hashing mode: no cross-encoder)."""

    name = "cosine"

    def __init__(self, threshold: float = COSINE_THRESHOLD) -> None:
        self.threshold = threshold

    def score(self, query: str, candidates: Sequence[Candidate]) -> NDArray[np.float64]:
        return np.array([c.cosine for c in candidates], dtype=np.float64)


class CrossEncoderJudge:  # pragma: no cover - needs the real model; measured by bench/kb_eval.py
    """dragonkue/bge-reranker-v2-m3-ko on the Apple GPU (fp16, 256 tokens) or CPU. Not thread-safe on
    MPS: in the server it runs on the embedder's worker thread (chatcore.serving.BatchedEmbedder)."""

    name = "cross-encoder"
    model_id = "dragonkue/bge-reranker-v2-m3-ko"

    def __init__(self, threshold: float = CROSS_ENCODER_THRESHOLD) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        mps = torch.backends.mps.is_available()
        self.threshold = threshold
        self._model = CrossEncoder(self.model_id, device="mps" if mps else "cpu", max_length=256,
                                   model_kwargs={"torch_dtype": torch.float16 if mps else torch.float32})

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float64]:
        if not pairs:
            return np.zeros(0)
        return np.asarray(self._model.predict(list(pairs), batch_size=32), dtype=np.float64)

    def score(self, query: str, candidates: Sequence[Candidate]) -> NDArray[np.float64]:
        return self.score_pairs([(query, c.entry.judge_text) for c in candidates])


# ---------------------------------------------------------------- index


@dataclass(frozen=True)
class Hit:
    entry: Entry
    cosine: float
    score: float  # the judge's


@dataclass(frozen=True)
class KbResult:
    answers: tuple[Hit, ...]  # passing candidates, best first ([] = no answer)
    candidates: tuple[Hit, ...]  # every judged candidate, best first
    exact: bool = False  # the sentence was an entry's title
    generated: dict[str, Any] | None = None  # the LLM's attempt (Generated.to_log), when one was made

    def to_log(self) -> dict[str, Any]:
        log = {"answered": [h.entry.id for h in self.answers], "exact": self.exact,
               "candidates": [[h.entry.id, round(h.cosine, 4), round(h.score, 4)] for h in self.candidates]}
        if self.generated is not None:
            log["generated"] = self.generated
        return log


def _matrix(rows: FloatArray, n: int, dims: int) -> FloatArray:
    return rows if n else np.zeros((0, dims), dtype=np.float32)


class KnowledgeIndex:
    def __init__(self, version: str, settings: KbSettings, faq: list[Entry], chunks: list[Entry],
                 faq_vectors: FloatArray, chunk_vectors: FloatArray, model_id: str, prompt: str) -> None:
        if len(faq) != len(faq_vectors) or len(chunks) != len(chunk_vectors):
            raise ValueError("entries and vectors do not match")
        self.version, self.settings, self.faq, self.chunks = version, settings, faq, chunks
        self.faq_vectors, self.chunk_vectors = faq_vectors, chunk_vectors
        self.model_id, self.prompt = model_id, prompt
        self._titles: dict[str, Entry] = {}
        for e in [*faq, *chunks]:
            self._titles.setdefault(bare(e.title), e)

    @property
    def empty(self) -> bool:
        return not self.faq and not self.chunks

    @classmethod
    def build(cls, document: dict[str, Any], embedder: Embedder) -> KnowledgeIndex:
        """KnowledgeError for an invalid document."""
        kb = parse_kb(document)
        faq, chunks = entries(kb)
        fv = embedder.encode([normalize(e.title) for e in faq]) if faq else np.zeros((0, 0), dtype=np.float32)
        cv = (encode_passages(embedder, [normalize(e.judge_text) for e in chunks]) if chunks
              else np.zeros((0, 0), dtype=np.float32))
        dims = max(fv.shape[1] if fv.ndim == 2 else 0, cv.shape[1] if cv.ndim == 2 else 0)
        return cls(kb_version(document), kb.settings, faq, chunks, _matrix(fv, len(faq), dims),
                   _matrix(cv, len(chunks), dims), embedder.model_id, embedder.prompt)

    # ------------------------------------------------------------ files

    def save(self, directory: Path) -> None:
        """Each file is written beside its final name and then renamed over it, so a reader never sees
        half a file (a crash leaves the old file or the new one)."""
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"version": self.version, "model_id": self.model_id, "prompt": self.prompt, "chunker": CHUNKER,
                "settings": self.settings.model_dump(), "faq": [asdict(e) for e in self.faq],
                "chunks": [asdict(e) for e in self.chunks]}
        vectors = io.BytesIO()
        np.savez(vectors, faq=self.faq_vectors, chunks=self.chunk_vectors)
        for name, data in (("vectors.npz", vectors.getvalue()),
                           ("index.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))):
            tmp = directory / f".{name}.{uuid.uuid4().hex}.tmp"
            try:
                tmp.write_bytes(data)
                os.replace(tmp, directory / name)
            finally:
                tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: Path, version: str, embedder: Embedder) -> KnowledgeIndex:
        """FileNotFoundError when the files are missing; ValueError when they are for another
        version or embedder, or unreadable (build it again)."""
        try:
            meta = json.loads((directory / "index.json").read_text(encoding="utf-8"))
            with np.load(directory / "vectors.npz") as data:
                fv, cv = np.asarray(data["faq"], dtype=np.float32), np.asarray(data["chunks"], dtype=np.float32)
            faq = [Entry(**e) for e in meta["faq"]]
            chunks = [Entry(**e) for e in meta["chunks"]]
            settings = KbSettings.model_validate(meta["settings"])
        except FileNotFoundError:
            raise
        except Exception as exc:  # damaged files (a truncated zip, bad JSON ...): build it again
            raise ValueError(f"unreadable knowledge-base index in {directory}: {exc}") from exc
        if meta.get("chunker") != CHUNKER:
            raise ValueError(f"the index in {directory} was made by another chunker")
        if meta.get("version") != version:
            raise ValueError(f"the index in {directory} is for {meta.get('version')}, not {version}")
        if meta.get("model_id") != embedder.model_id or meta.get("prompt") != embedder.prompt:
            raise ValueError(f"the index in {directory} was built with {meta.get('model_id')}")
        return cls(version, settings, faq, chunks, fv, cv, meta["model_id"], meta["prompt"])

    # ------------------------------------------------------------ search

    def nearest_faq(self, vector: FloatArray) -> tuple[Entry, float] | None:
        """The FAQ entry most similar to the sentence, by cosine (cheap: no judge)."""
        if not self.faq:
            return None
        cos = self.faq_vectors @ vector
        i = int(np.argmax(cos))
        return self.faq[i], float(cos[i])

    def exact(self, text: str) -> KbResult | None:
        """The entry whose title is this sentence (what a suggestion button sends), if any."""
        entry = self._titles.get(bare(text))
        if entry is None:
            return None
        hit = Hit(entry, 1.0, 1.0)  # finite: results go into the JSON event log
        return KbResult((hit,), (hit,), exact=True)

    def search(self, text: str, vector: FloatArray, judge: Judge) -> KbResult:
        """`vector`: the sentence encoded as the NLU encodes it (normalized, with its prompt)."""
        exact = self.exact(text)
        if exact is not None:
            return exact
        candidates: list[Candidate] = []
        for pool, vectors in ((self.faq, self.faq_vectors), (self.chunks, self.chunk_vectors)):
            if pool:
                cos = vectors @ vector
                for i in np.argsort(-cos, kind="stable")[:JUDGE_DEPTH]:
                    candidates.append(Candidate(pool[int(i)], float(cos[int(i)])))
        if not candidates:
            return KbResult((), ())
        scores = np.asarray(judge.score(normalize(text), candidates), dtype=np.float64)
        if len(scores) != len(candidates):
            raise ValueError(f"the judge returned {len(scores)} scores for {len(candidates)} candidates")
        if not np.isfinite(scores).all():  # they go into the JSON event log
            raise ValueError("the judge returned scores that are not finite numbers")
        order = np.argsort(-scores, kind="stable")
        hits = tuple(Hit(candidates[int(i)].entry, candidates[int(i)].cosine, float(scores[int(i)])) for i in order)
        return KbResult(tuple(h for h in hits if h.score >= judge.threshold), hits)


# ---------------------------------------------------------------- answers


@dataclass(frozen=True)
class KbAnswer:
    """What the bot says: one answer, or (답안 추천형) a prompt with suggestion buttons."""

    text: str
    replies: tuple[tuple[str, str], ...] = ()  # (button label, the message it sends)
    items: tuple[str, ...] = ()  # entry ids, for logs


def answer_text(entry: Entry) -> str:
    return entry.body if entry.kind == "faq" else f"{entry.body}\n\n출처: {entry.source}"


def button_label(text: str, suffix: str = "") -> str:
    """Kakao shows 14 characters of a button (the button still sends the whole title); `suffix` is kept."""
    room = KAKAO_LABEL - len(suffix)
    return (text if len(text) <= room else text[:room - 1].rstrip() + "…") + suffix


def _section(entry: Entry) -> str:
    """What a suggestion button says: an FAQ question, or a chunk's section (its heading without the
    document title; the "(2/3)" part of a title would not fit in 14 characters)."""
    if entry.kind == "faq":
        return entry.title
    return entry.section or entry.heading or entry.title  # no section: the document's title


def answers_nothing(outcome: str, on_reject: str) -> bool:
    """Whether a generation that was not kept leaves the turn without a knowledge-base answer: only when
    the check rejected what the model said, under "fallback". When the LLM did not answer at all
    (timeout, error) the passage is shown, as if generation were off."""
    return on_reject == "fallback" and outcome not in LLM_FAILURES


def compose(result: KbResult, settings: KbSettings) -> KbAnswer | None:
    """The message for a search result; None when nothing passed. 답안 추천형 offers one button per FAQ
    entry or document section (its best chunk), up to SUGGESTIONS, with labels that differ."""
    if not result.answers:
        return None
    best_of: dict[str, Entry] = {}
    for h in result.answers:  # best first
        e = h.entry
        best_of.setdefault(e.id if e.kind == "faq" else f"{e.id.split('#')[0]}#{e.heading or e.title}", e)
    if settings.mode == "instant" or result.exact or len(best_of) == 1:
        best = result.answers[0].entry
        return KbAnswer(answer_text(best), (), (best.id,))
    first = list(best_of.values())[:SUGGESTIONS]
    seen: dict[str, int] = {}  # by the label as shown: names that differ only after 14 characters look alike
    replies = []
    for e in first:
        name = _section(e)
        label = button_label(name)
        seen[label] = seen.get(label, 0) + 1
        replies.append((label if seen[label] == 1 else button_label(name, f" ({seen[label]})"), e.title))
    return KbAnswer(settings.suggest_text, tuple(replies), tuple(e.id for e in first))


# ---------------------------------------------------------------- in a conversation


@dataclass(frozen=True)
class Knowledge:
    """A bot's knowledge base as served: its index, the judge, the NLU's embedder and, when the server
    has an LLM, the generator (used only when the knowledge base's settings ask for it)."""

    index: KnowledgeIndex
    judge: Judge
    embedder: Embedder
    generator: Generator | None = None
    on_reject: Literal["extract", "fallback"] = ON_REJECT

    def answer(self, text: str, result: KbResult) -> tuple[KbResult, KbAnswer | None]:
        """The message for a search result: compose() as before, or — 즉문즉답형 with generation on — the
        LLM's answer from the passing passages (judge first, so it never answers what the judge refused).
        A rejected generation shows the extractive answer or nothing (`on_reject`); a failed one (the LLM
        did not answer) shows the extractive answer."""
        answer = compose(result, self.index.settings)
        settings = self.index.settings
        if (answer is None or self.generator is None or not settings.generate or settings.mode != "instant"
                or result.exact):  # a chosen button shows exactly what it offered
            return result, answer
        made = self.generator.generate(text, [h.entry for h in result.answers])
        result = replace(result, generated=made.to_log(self.generator.model))
        if made.text is None:
            return result, (None if answers_nothing(made.outcome, self.on_reject) else answer)
        used = [h.entry for h in result.answers if h.entry.id in made.used]
        sources = list(dict.fromkeys(e.source for e in used if e.kind == "doc"))
        body = made.text + (f"\n\n출처: {', '.join(sources)}" if sources else "")
        return result, KbAnswer(body, (), made.used)

    def _vector(self, text: str, vector: FloatArray | None) -> FloatArray:
        return vector if vector is not None else self.embedder.encode([normalize(text)])[0]

    def consult(self, text: str, rule: str, similarity: float,
                vector: FloatArray | None) -> tuple[KbResult | None, KbAnswer | None]:
        """Where the knowledge base sits among the dialogue rules (docs/engine-core-plan.md "빌더 단계 E-D"):
        an entry's exact title is an explicit choice (a suggestion button) and is answered, except over the
        bot's own example; a sure block ("intent") gives way only to an FAQ entry whose question is
        OVERRIDE_MARGIN closer than the block's nearest example (checked by cosine before the judge runs,
        so block turns stay fast); "clarify" and "fallback" give way to any answer.
        (search result or None, answer or None)."""
        if self.index.empty or rule not in ("intent", "clarify", "fallback"):
            return None, None
        exact = self.index.exact(text)  # a chosen suggestion: answer it, unless it is the bot's own example
        if exact is not None and not (rule == "intent" and similarity >= 1.0):
            return exact, compose(exact, self.index.settings)
        if rule == "intent":
            if similarity + OVERRIDE_MARGIN > 1.0:  # an exact example: nothing can be closer
                return None, None
            v = self._vector(text, vector)
            nearest = self.index.nearest_faq(v)
            if nearest is None or nearest[1] < similarity + OVERRIDE_MARGIN:
                return None, None
            result = self.index.search(text, v, self.judge)
            top = result.answers[0] if result.answers else None
            if top is None or top.entry.kind != "faq" or top.cosine < similarity + OVERRIDE_MARGIN:
                return result, None
            return self.answer(text, KbResult((top,), result.candidates, result.exact))
        return self.answer(text, self.index.search(text, self._vector(text, vector), self.judge))

    def action(self) -> ActionFn:
        """The `knowledge` action: search with `query` (default: the user's sentence) and return
        found ("yes"/"no"), answer, source and title of the best passing entry."""

        def run(args: dict[str, str]) -> dict[str, Any]:
            turn = CURRENT_TURN.get()
            query = " ".join(str(args.get("query") or (turn.utterance if turn else "")).split())
            if not query or self.index.empty:
                return dict(NOT_FOUND)
            result, answer = self.answer(query, self.index.search(query, self._vector(query, None), self.judge))
            if not result.answers or answer is None:
                return dict(NOT_FOUND)
            e = result.answers[0].entry
            # 답안 추천형 has no single answer text: the action gives the best entry's own text
            text = answer.text if not answer.replies else answer_text(e)
            return {"found": "yes", "answer": text, "source": e.source, "title": e.title}

        return run
