"""E-D: the knowledge base — document, chunks, index, judging, answers, conversation rules, files."""

import io
import json

import numpy as np
import pytest

from chatcore.dialogue.values import CURRENT_TURN, TurnContext
from chatcore.knowledge import (
    CHUNK_MAX,
    JUDGE_DEPTH,
    OVERRIDE_MARGIN,
    CosineJudge,
    Entry,
    KbResult,
    KbSettings,
    Knowledge,
    KnowledgeError,
    KnowledgeIndex,
    chunk_document,
    compose,
    entries,
    extract_text,
    kb_version,
    parse_kb,
)
from chatcore.nlu.entities import normalize
from chatcore.nlu.hashing import HashingEmbedder

KB = {
    "faq": [{"id": "refund", "question": "구독 환불은 어떻게 하나요", "answer": "결제 후 7일 안에 설정 > 구독에서 환불할 수 있어요."},
            {"id": "widget", "question": "위젯은 어떻게 추가하나요", "answer": "홈 화면을 길게 눌러 위젯을 추가하세요."}],
    "documents": [{"id": "policy", "name": "이용 안내",
                   "text": "# 이용 안내\n\n## 개인정보\n\n위치 정보는 날씨 조회에만 쓰고 30일 뒤에 지웁니다."}],
}
EMB = HashingEmbedder()


def vec(text):
    return EMB.encode([normalize(text)])[0]


@pytest.fixture(scope="module")
def index():
    return KnowledgeIndex.build(KB, EMB)


class CountingJudge(CosineJudge):
    def __init__(self, threshold=0.5):
        super().__init__(threshold)
        self.calls = []

    def score(self, query, candidates):
        self.calls.append((query, [c.entry.id for c in candidates]))
        return super().score(query, candidates)


# ---------------------------------------------------------------- document


def test_parse_kb_accepts_a_document_and_defaults_the_settings() -> None:
    kb = parse_kb(KB)
    assert [f.id for f in kb.faq] == ["refund", "widget"] and kb.settings.mode == "instant"
    assert parse_kb({}).faq == [] and parse_kb({}).documents == []  # an empty knowledge base is valid


@pytest.mark.parametrize("doc, problem", [
    ([], "must be an object"),
    ({"faq": [{"id": "a", "question": "q", "answer": "x"}, {"id": "a", "question": "r", "answer": "y"}]}, "duplicate FAQ"),
    ({"documents": [{"id": "d", "name": "n", "text": "t"}, {"id": "d", "name": "m", "text": "u"}]}, "duplicate document"),
    ({"faq": [{"id": "a", "question": "  ", "answer": "x"}]}, "blank"),
    ({"faq": [{"id": "a b", "question": "q", "answer": "x"}]}, "faq.0.id"),
    ({"faq": [{"id": "a", "question": "q", "answer": "x", "extra": 1}]}, "faq.0.extra"),
    ({"settings": {"mode": "random"}}, "settings.mode"),
    ({"documents": [{"id": "d", "name": "n", "text": "가" * 200_001}]}, "documents.0.text"),
])
def test_parse_kb_lists_problems(doc, problem) -> None:
    with pytest.raises(KnowledgeError) as exc:
        parse_kb(doc)
    assert any(problem in p for p in exc.value.problems), exc.value.problems


def test_kb_version_is_a_content_hash_independent_of_key_order() -> None:
    v = kb_version(KB)
    assert len(v) == 12 and int(v, 16) >= 0
    assert kb_version(dict(reversed(list(KB.items())))) == v
    assert kb_version({**KB, "faq": KB["faq"][:1]}) != v


# ---------------------------------------------------------------- chunks


def test_chunks_are_whole_sentences_within_a_section_with_a_title() -> None:
    sentence = "맑음노트는 매시 정각에 예보를 갱신합니다."
    text = f"# 설명서\n\n## 갱신\n\n{' '.join([sentence] * 40)}\n\n## 위젯\n\n위젯은 홈 화면에서 추가합니다."
    chunks = chunk_document("manual", "manual.md", text)
    assert len(chunks) >= 3 and [c.id for c in chunks] == [f"manual#{i}" for i in range(len(chunks))]
    *update, widget = chunks
    assert all(c.heading == "설명서 - 갱신" and c.body.replace(sentence, "").strip() == "" for c in update)
    assert [c.title for c in update] == [f"설명서 - 갱신 ({i}/{len(update)})" for i in range(1, len(update) + 1)]
    assert all(len(c.body) <= CHUNK_MAX for c in update)
    assert widget.title == "설명서 - 위젯" and widget.body == "위젯은 홈 화면에서 추가합니다." and widget.source == "manual.md"


def test_chunks_repeat_a_short_last_sentence_and_never_cut_a_long_one() -> None:
    sents = [f"문장 {i}번은 겹침을 확인하려고 쓴 꽤 긴 예시 문장으로 약 마흔 글자 정도입니다." for i in range(30)]
    first, second, *_ = chunk_document("a", "a", "# 제목\n\n## 절\n\n" + " ".join(sents))
    last = next(x for x in reversed(sents) if first.body.endswith(x))
    assert second.body.startswith(last + " ")  # the first chunk's last sentence opens the second
    long = "가" * (CHUNK_MAX + 100) + "."
    assert any(long in c.body for c in chunk_document("b", "b", f"짧습니다. {long} 끝입니다."))


def test_a_document_without_headings_uses_its_name_and_empty_text_has_no_chunks() -> None:
    (only,) = chunk_document("d", "안내문", "그냥 문장입니다.")
    assert only.title == "안내문" and only.body == "그냥 문장입니다."
    assert chunk_document("d", "n", "") == [] and chunk_document("d", "n", "# 제목만\n\n## 절만") == []


def test_entries_keep_faq_order_and_collapse_question_spaces() -> None:
    faq, chunks = entries(parse_kb({**KB, "faq": [{"id": "x", "question": "환불   돼요?", "answer": " 돼요 "}]}))
    assert [(e.id, e.title, e.body, e.kind) for e in faq] == [("x", "환불 돼요?", "돼요", "faq")]
    assert [c.id for c in chunks] == ["policy#0"] and chunks[0].judge_text.startswith("이용 안내 - 개인정보\n")


# ---------------------------------------------------------------- index and search


def test_build_encodes_faq_as_sentences_and_chunks_as_passages(index) -> None:
    assert index.version == kb_version(KB) and not index.empty
    assert index.faq_vectors.shape == (2, EMB.dims) and index.chunk_vectors.shape == (1, EMB.dims)
    assert np.allclose(index.faq_vectors[0], vec("구독 환불은 어떻게 하나요"))
    assert np.allclose(index.chunk_vectors[0], EMB.encode([normalize(index.chunks[0].judge_text)])[0])


def test_an_empty_knowledge_base_builds_and_finds_nothing() -> None:
    empty = KnowledgeIndex.build({}, EMB)
    assert empty.empty and empty.faq_vectors.shape[0] == 0 and empty.nearest_faq(vec("환불")) is None
    assert empty.search("환불", vec("환불"), CosineJudge()) == KbResult((), ())


def test_search_answers_with_what_passes_the_judge_best_first(index) -> None:
    judge = CountingJudge(0.5)
    found = index.search("구독 환불 어떻게 해요", vec("구독 환불 어떻게 해요"), judge)
    assert [h.entry.id for h in found.answers] == ["refund"] and not found.exact
    assert [h.entry.id for h in found.candidates] == ["refund", "widget", "policy#0"]
    assert judge.calls[0][1] == ["refund", "widget", "policy#0"]  # top JUDGE_DEPTH of each pool
    doc = index.search("위치 정보는 언제 지우나요", vec("위치 정보는 언제 지우나요"), judge)
    assert [h.entry.id for h in doc.answers] == ["policy#0"]
    assert index.search("주식 추천해줘", vec("주식 추천해줘"), judge).answers == ()


def test_search_judges_at_most_judge_depth_per_pool() -> None:
    many = {"faq": [{"id": f"f{i}", "question": f"질문 {i}번", "answer": "답"} for i in range(JUDGE_DEPTH + 3)]}
    judge = CountingJudge()
    KnowledgeIndex.build(many, EMB).search("질문", vec("질문"), judge)
    assert len(judge.calls[0][1]) == JUDGE_DEPTH


def test_a_candidate_exactly_at_the_threshold_answers(index) -> None:
    cos = float(index.chunk_vectors[0] @ vec("위치 정보는 언제 지우나요"))
    assert index.search("위치 정보는 언제 지우나요", vec("위치 정보는 언제 지우나요"), CosineJudge(cos)).answers
    assert not index.search("위치 정보는 언제 지우나요", vec("위치 정보는 언제 지우나요"), CosineJudge(cos + 1e-6)).answers


def test_an_entry_title_is_answered_directly_without_the_judge(index) -> None:
    judge = CountingJudge(0.99)
    for title, entry in (("이용 안내 - 개인정보", "policy#0"), (" 구독 환불은 어떻게 하나요? ", "refund")):
        found = index.search(title, vec(title), judge)
        assert found.exact and [h.entry.id for h in found.answers] == [entry] and found.answers[0].score == 1.0
    assert judge.calls == []
    json.dumps(found.to_log(), allow_nan=False)  # goes into the event log (jsonb has no Infinity)


def test_a_judge_returning_the_wrong_number_of_scores_is_an_error(index) -> None:
    class Broken(CosineJudge):
        def score(self, query, candidates):
            return np.zeros(1)

    with pytest.raises(ValueError, match="scores"):
        index.search("환불", vec("환불"), Broken())


def test_index_files_round_trip_and_refuse_another_version_or_embedder(index, tmp_path) -> None:
    index.save(tmp_path / "kb")
    loaded = KnowledgeIndex.load(tmp_path / "kb", index.version, EMB)
    assert loaded.faq == index.faq and loaded.chunks == index.chunks and np.array_equal(loaded.faq_vectors, index.faq_vectors)
    assert loaded.settings == index.settings
    with pytest.raises(ValueError, match="is for"):
        KnowledgeIndex.load(tmp_path / "kb", "0" * 12, EMB)

    class Other(HashingEmbedder):
        model_id = "other"

    with pytest.raises(ValueError, match="built with"):
        KnowledgeIndex.load(tmp_path / "kb", index.version, Other())
    with pytest.raises(FileNotFoundError):
        KnowledgeIndex.load(tmp_path / "missing", index.version, EMB)
    (tmp_path / "kb" / "index.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        KnowledgeIndex.load(tmp_path / "kb", index.version, EMB)


def test_an_embedder_that_needs_a_prompt_cannot_encode_passages() -> None:
    class Prompted:
        model_id, prompt = "m", "query: "

        def encode(self, texts):
            return EMB.encode(texts)

    with pytest.raises(TypeError, match="passages"):
        KnowledgeIndex.build(KB, Prompted())
    assert KnowledgeIndex.build({"faq": KB["faq"]}, Prompted()).faq  # FAQ alone needs no passages


# ---------------------------------------------------------------- answers


def hit(id_, title, kind="faq", body="답", source=""):
    from chatcore.knowledge import Hit

    return Hit(Entry(id_, kind, title, body, source), 0.9, 0.9)


def test_compose_instant_gives_the_best_answer_and_a_chunk_its_source() -> None:
    faq = hit("refund", "환불 돼요?", body="7일 안에 돼요.")
    chunk = hit("policy#0", "이용 안내 - 개인정보", kind="doc", body="30일 뒤에 지웁니다.", source="이용 안내")
    assert compose(KbResult((faq, chunk), ()), KbSettings()).text == "7일 안에 돼요."
    answer = compose(KbResult((chunk,), ()), KbSettings())
    assert answer.text == "30일 뒤에 지웁니다.\n\n출처: 이용 안내" and answer.items == ("policy#0",) and answer.replies == ()
    assert compose(KbResult((), (faq,)), KbSettings()) is None


def test_compose_suggest_offers_up_to_three_buttons_one_per_title() -> None:
    hits = (hit("a", "구독을 해지하면 환불이 되나요 연간도요"), hit("b", "짧은 질문"), hit("c#0", "문서 - 절", "doc"),
            hit("c#1", "문서 - 절", "doc"), hit("d", "네 번째"))
    answer = compose(KbResult(hits, hits), KbSettings(mode="suggest"))
    assert answer.text == KbSettings().suggest_text and answer.items == ("a", "b", "c#0")
    assert answer.replies == (("구독을 해지하면 환불이…", "구독을 해지하면 환불이 되나요 연간도요"), ("짧은 질문", "짧은 질문"),
                              ("문서 - 절", "문서 - 절"))
    assert all(len(label) <= 14 for label, _ in answer.replies)  # Kakao shows 14 characters


def test_compose_suggest_answers_directly_with_one_title_or_an_exact_title() -> None:
    one = hit("a", "한 개")
    assert compose(KbResult((one,), (one,)), KbSettings(mode="suggest")).replies == ()
    two = (hit("a", "하나"), hit("b", "둘"))
    assert compose(KbResult(two, two, exact=True), KbSettings(mode="suggest")).items == ("a",)


# ---------------------------------------------------------------- dialogue rules


def test_consult_answers_fallback_and_clarify_turns(index) -> None:
    k = Knowledge(index, CosineJudge(0.5), EMB)
    for rule in ("fallback", "clarify"):
        found, answer = k.consult("구독 환불 어떻게 해요", rule, 0.3, None)
        assert answer is not None and answer.items == ("refund",) and found.answers
    found, answer = k.consult("주식 추천해줘", "fallback", 0.3, vec("주식 추천해줘"))
    assert answer is None and found is not None and found.candidates  # searched, nothing passed


def test_consult_lets_an_faq_beat_a_sure_block_only_by_the_margin(index) -> None:
    judge = CountingJudge(0.5)
    k = Knowledge(index, judge, EMB)
    cos = float(index.faq_vectors[0] @ vec("구독 환불 어떻게 해요"))
    found, answer = k.consult("구독 환불 어떻게 해요", "intent", cos - OVERRIDE_MARGIN, None)
    assert answer is not None and answer.items == ("refund",)
    assert k.consult("구독 환불 어떻게 해요", "intent", cos - OVERRIDE_MARGIN + 1e-3, None) == (None, None)
    assert len(judge.calls) == 1  # the cheap cosine check spares the judge on block turns
    assert k.consult("구독 환불은 어떻게 하나요", "intent", 1.0, None) == (None, None)  # an exact example


def test_consult_never_lets_a_chunk_beat_a_sure_block(index) -> None:
    k = Knowledge(index, CosineJudge(0.5), EMB)
    found, answer = k.consult("위치 정보는 언제 지우나요", "intent", 0.0, None)
    assert answer is None and found is not None and found.answers[0].entry.kind == "doc"


def test_consult_skips_other_rules_and_empty_knowledge_bases(index) -> None:
    k = Knowledge(index, CosineJudge(0.0), EMB)
    for rule in ("slot", "rerun", "system", "block"):
        assert k.consult("구독 환불 어떻게 해요", rule, 0.0, None) == (None, None)
    assert Knowledge(KnowledgeIndex.build({}, EMB), CosineJudge(0.0), EMB).consult("x", "fallback", 0, None) == (None, None)


def test_knowledge_action_searches_the_query_or_the_users_sentence(index) -> None:
    run = Knowledge(index, CosineJudge(0.5), EMB).action()
    assert run({"query": "위치 정보는 언제 지우나요"}) == {
        "found": "yes", "answer": "위치 정보는 날씨 조회에만 쓰고 30일 뒤에 지웁니다.\n\n출처: 이용 안내",
        "source": "이용 안내", "title": "이용 안내 - 개인정보"}
    token = CURRENT_TURN.set(TurnContext("구독 환불 어떻게 해요", "s1"))
    try:
        assert run({})["answer"] == "결제 후 7일 안에 설정 > 구독에서 환불할 수 있어요."
    finally:
        CURRENT_TURN.reset(token)
    assert run({"query": "주식 추천해줘"})["found"] == "no"
    assert run({}) == {"found": "no", "answer": "", "source": "", "title": ""}  # no query, no turn


# ---------------------------------------------------------------- uploaded files


def pdf_with_text(text: str) -> bytes:
    """A one-page PDF whose text layer is `text` (ASCII; standard Helvetica)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref))
    return out.getvalue()


def test_extract_text_reads_pdf_text_layers_and_utf8_text() -> None:
    assert extract_text(pdf_with_text("Refunds within 7 days"), "application/pdf") == "Refunds within 7 days"
    assert extract_text(pdf_with_text("No type given"), "") == "No type given"  # sniffed by its header
    assert extract_text("﻿# 안내\r\n\r\n\r\n\r\n본문\x00".encode(), "text/markdown") == "# 안내\n\n본문"
    assert extract_text("일반 글".encode(), "text/plain") == "일반 글"


@pytest.mark.parametrize("data, kind, problem", [
    (b"\xff\xfe\x00", "text/plain", "not UTF-8"),
    (b"<html>", "text/html", "unsupported"),
    (b"   \n\n ", "text/plain", "no text"),
    (b"%PDF-1.4 broken", "application/pdf", "not a readable PDF"),
    ("가".encode() * 200_001, "text/plain", "at most"),
])
def test_extract_text_refuses_what_it_cannot_use(data, kind, problem) -> None:
    with pytest.raises(KnowledgeError) as exc:
        extract_text(data, kind)
    assert problem in exc.value.problems[0]


def test_extract_text_refuses_encrypted_and_textless_pdfs() -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    blank = io.BytesIO()
    writer.write(blank)
    with pytest.raises(KnowledgeError, match="no text"):
        extract_text(blank.getvalue(), "application/pdf")  # a scan has no text layer
    writer.encrypt("secret")
    locked = io.BytesIO()
    writer.write(locked)
    with pytest.raises(KnowledgeError, match="encrypted"):
        extract_text(locked.getvalue(), "application/pdf")


def test_extract_text_refuses_a_pdf_with_too_many_pages() -> None:
    from pypdf import PdfWriter

    from chatcore.knowledge import MAX_PDF_PAGES

    writer = PdfWriter()
    for _ in range(MAX_PDF_PAGES + 1):
        writer.add_blank_page(width=10, height=10)
    big = io.BytesIO()
    writer.write(big)
    with pytest.raises(KnowledgeError, match="pages"):
        extract_text(big.getvalue(), "application/pdf")


def test_an_index_refuses_entries_and_vectors_that_do_not_match(index) -> None:
    with pytest.raises(ValueError, match="do not match"):
        KnowledgeIndex(index.version, index.settings, index.faq, index.chunks, index.faq_vectors[:1],
                       index.chunk_vectors, EMB.model_id, EMB.prompt)


# ---------------------------------------------------------------- review fixes


def test_compact_markdown_keeps_every_section_and_text_before_a_heading() -> None:
    text = "# 안내\n머리말입니다.\n## 환불\n7일 이내 환불됩니다.\n## 배송\n배송은 3일 걸립니다."
    chunks = chunk_document("d", "안내.md", text)
    assert [(c.title, c.body) for c in chunks] == [("안내", "머리말입니다."), ("안내 - 환불", "7일 이내 환불됩니다."),
                                                   ("안내 - 배송", "배송은 3일 걸립니다.")]


def test_every_suggestion_button_brings_back_the_entry_it_offered() -> None:
    long_section = " ".join(f"환불 규정 {i}번 문장은 길이를 채우려고 쓴 꽤 긴 문장입니다." for i in range(40))
    doc = {"documents": [{"id": "a", "name": "정책", "text": f"# 정책\n\n## 환불\n\n{long_section}"},
                         {"id": "b", "name": "정책", "text": "# 정책\n\n## 환불\n\n다른 문서의 환불 문장입니다."}],
           "settings": {"mode": "suggest"}}
    idx = KnowledgeIndex.build(doc, EMB)
    assert len(idx.chunks) > 3
    hits = tuple(__import__("chatcore.knowledge", fromlist=["Hit"]).Hit(c, 0.9, 0.9) for c in idx.chunks)
    answer = compose(KbResult(hits, hits), idx.settings)
    assert len({m for _, m in answer.replies}) == len(answer.replies)  # every button sends a different message
    for (_, message), item in zip(answer.replies, answer.items, strict=True):
        assert idx.search(message, vec(message), CosineJudge(0.99)).answers[0].entry.id == item
    for c in idx.chunks:  # every chunk can be reached by its own title
        assert idx.search(c.title, vec(c.title), CosineJudge(0.99)).answers[0].entry.id == c.id


def test_an_exact_title_is_an_explicit_choice_that_beats_a_sure_block(index) -> None:
    k = Knowledge(index, CosineJudge(0.99), EMB)
    found, answer = k.consult("이용 안내 - 개인정보", "intent", 0.9, None)
    assert answer is not None and answer.items == ("policy#0",) and found.exact
    assert k.consult("구독 환불은 어떻게 하나요", "intent", 0.97, None)[1].items == ("refund",)
    assert k.consult("구독 환불은 어떻게 하나요", "intent", 1.0, None) == (None, None)  # the bot's own example


def test_a_judge_score_that_is_not_a_number_is_an_error(index) -> None:
    class NaNJudge(CosineJudge):
        def score(self, query, candidates):
            return np.full(len(candidates), np.nan)

    with pytest.raises(ValueError, match="finite"):
        index.search("환불", vec("환불"), NaNJudge())


@pytest.mark.parametrize("damage", [b"", b"PK\x03\x04 truncated"])
def test_a_damaged_vectors_file_reads_as_unreadable(index, tmp_path, damage) -> None:
    index.save(tmp_path / "kb")
    (tmp_path / "kb" / "vectors.npz").write_bytes(damage)
    with pytest.raises(ValueError, match="unreadable"):
        KnowledgeIndex.load(tmp_path / "kb", index.version, EMB)


def test_an_index_made_by_another_chunker_is_stale(index, tmp_path) -> None:
    index.save(tmp_path / "kb")
    meta = json.loads((tmp_path / "kb" / "index.json").read_text(encoding="utf-8"))
    meta["chunker"] = 0
    (tmp_path / "kb" / "index.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="chunker"):
        KnowledgeIndex.load(tmp_path / "kb", index.version, EMB)


def test_saving_leaves_no_temporary_files(index, tmp_path) -> None:
    index.save(tmp_path / "kb")
    index.save(tmp_path / "kb")  # over an existing index
    assert sorted(p.name for p in (tmp_path / "kb").iterdir()) == ["index.json", "vectors.npz"]


def test_two_documents_with_the_same_name_and_section_get_different_titles() -> None:
    doc = {"documents": [{"id": "a", "name": "정책", "text": "# 정책\n\n## 환불\n\n첫 문서의 환불 문장입니다."},
                         {"id": "b", "name": "정책", "text": "# 정책\n\n## 환불\n\n둘째 문서의 환불 문장입니다."}]}
    idx = KnowledgeIndex.build(doc, EMB)
    assert [c.title for c in idx.chunks] == ["정책 - 환불 · a", "정책 - 환불 · b"]
    assert [c.heading for c in idx.chunks] == ["정책 - 환불", "정책 - 환불"]  # what is embedded stays as measured
    for c in idx.chunks:
        assert idx.search(c.title, vec(c.title), CosineJudge(0.99)).answers[0].entry.id == c.id
