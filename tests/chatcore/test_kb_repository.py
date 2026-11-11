"""Repository contract for the knowledge base: its own draft and versions
(independent of the bot's draft/versions), and the (bot version, kb version)
deployment pair.
"""

import os
import uuid

import psycopg
import pytest

from chatcore.bot.repository import InMemoryRepository, PostgresRepository, RevisionConflict, VersionNotFound


def postgres_repository():
    url = os.environ.get("CHATCORE_TEST_DATABASE_URL")
    if not url:
        if os.environ.get("CI"):
            pytest.fail("CHATCORE_TEST_DATABASE_URL must be set in CI")
        pytest.skip("set CHATCORE_TEST_DATABASE_URL (make test-db) to run Postgres tests")
    repo = PostgresRepository(url)
    repo.ensure_schema()
    return repo


@pytest.fixture(params=["memory", "postgres"])
def repository(request):
    return InMemoryRepository() if request.param == "memory" else postgres_repository()


def bot_id() -> str:
    return f"kb-{uuid.uuid4().hex[:8]}"


def kb_doc(label="초안"):
    return {"faq": [{"id": "f1", "question": "영업시간", "answer": "9-6"}], "documents": [],
            "settings": {"mode": "instant"}, "label": label}


def test_kb_draft_is_independent_of_the_bot_draft(repository):
    bot = bot_id()
    assert repository.kb_draft(bot) is None and repository.draft(bot) is None
    repository.save_kb_draft(bot, kb_doc())
    assert repository.kb_draft(bot).document == kb_doc()
    assert repository.draft(bot) is None  # saving a KB draft never touches the bot draft


def test_kb_draft_revisions_and_conflicts(repository):
    bot = bot_id()
    assert repository.save_kb_draft(bot, kb_doc(), expected_revision=0) == 1  # create
    with pytest.raises(RevisionConflict) as exists:
        repository.save_kb_draft(bot, kb_doc(), expected_revision=0)
    assert exists.value.current == 1
    assert repository.save_kb_draft(bot, kb_doc("둘"), expected_revision=1) == 2
    with pytest.raises(RevisionConflict) as stale:
        repository.save_kb_draft(bot, kb_doc("셋"), expected_revision=1)  # the other tab
    assert stale.value.current == 2
    assert repository.kb_draft(bot).document["label"] == "둘"
    assert repository.save_kb_draft(bot, kb_doc()) == 3  # no expectation: plain overwrite


def test_kb_draft_is_none_for_an_unknown_bot(repository):
    assert repository.kb_draft(bot_id()) is None


def test_kb_versions_are_numbered_newest_first_and_idempotent(repository):
    bot = bot_id()
    assert repository.kb_versions(bot) == []
    first = repository.add_kb_version(bot, "aaaaaaaaaaaa", kb_doc(), faq=3, chunks=10)
    second = repository.add_kb_version(bot, "bbbbbbbbbbbb", kb_doc("둘"), faq=5, chunks=20)
    assert (first.number, second.number) == (1, 2)
    assert [v.version for v in repository.kb_versions(bot)] == ["bbbbbbbbbbbb", "aaaaaaaaaaaa"]
    again = repository.add_kb_version(bot, "aaaaaaaaaaaa", kb_doc("ignored"), faq=99, chunks=99)
    assert again == first  # idempotent: same number, same created_at, new document ignored
    assert repository.get_kb_version(bot, "aaaaaaaaaaaa") == kb_doc()
    assert repository.get_kb_version(bot, "bbbbbbbbbbbb")["label"] == "둘"


def test_add_kb_version_accepts_zero_faq_chunks_and_an_empty_document(repository):
    bot = bot_id()
    empty = {"faq": [], "documents": []}
    info = repository.add_kb_version(bot, "dddddddddddd", empty, faq=0, chunks=0)
    assert info.faq == 0 and info.chunks == 0
    assert repository.get_kb_version(bot, "dddddddddddd") == empty


def test_get_kb_version_unknown_raises_version_not_found(repository):
    with pytest.raises(VersionNotFound):
        repository.get_kb_version(bot_id(), "cccccccccccc")


def test_a_kb_version_for_a_bot_with_no_bot_version_is_allowed(repository):
    bot = bot_id()
    assert repository.versions(bot) == []
    repository.add_kb_version(bot, "aaaaaaaaaaaa", kb_doc(), faq=1, chunks=1)
    assert [v.version for v in repository.kb_versions(bot)] == ["aaaaaaaaaaaa"]
    assert repository.versions(bot) == []  # the bot's own versions are untouched


def test_deploy_pairs_a_bot_version_with_a_kb_version(repository):
    bot = bot_id()
    repository.add_version(bot, "aaaaaaaaaaaa", {"id": bot}, None)
    repository.add_kb_version(bot, "kkkkkkkkkkkk", kb_doc(), faq=1, chunks=2)

    repository.deploy(bot, "aaaaaaaaaaaa", "with kb", kb_version="kkkkkkkkkkkk")
    assert repository.history(bot)[0].kb_version == "kkkkkkkkkkkk"

    repository.deploy(bot, "aaaaaaaaaaaa", "without kb")
    assert repository.history(bot)[0].kb_version is None  # the default: no knowledge base

    with pytest.raises(VersionNotFound):
        repository.deploy(bot, "aaaaaaaaaaaa", "bad kb", kb_version="zzzzzzzzzzzz")
    assert len(repository.history(bot)) == 2  # the failed deploy recorded nothing
    assert [d.kb_version for d in repository.history(bot)] == [None, "kkkkkkkkkkkk"]


def test_postgres_deployments_made_before_the_kb_version_column_read_back_as_none():
    repo = postgres_repository()
    bot = bot_id()
    repo.add_version(bot, "aaaaaaaaaaaa", {"id": bot}, None)
    with psycopg.connect(repo.dsn) as conn:  # simulates a row written before ALTER TABLE added the column
        conn.execute("INSERT INTO bot_deployments (bot_id, version, note) VALUES (%s, %s, %s)",
                     (bot, "aaaaaaaaaaaa", ""))
    repo.ensure_schema()  # must stay safe to run again on a database that already has the column
    history = repo.history(bot)
    assert len(history) == 1 and history[0].kb_version is None
