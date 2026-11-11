"""Bot hosts: which runtime answers a session.

`StaticHost` serves fixed runtimes (tests, simple setups). `BotHost` serves
versions from a BotRepository and lets a builder train, deploy and roll back
without restarting (docs/engine-core-plan.md §3.1, Phase 6):

- a session stays on the version it started with while that version is
  loaded (the deployed one plus the `keep` most recently used others); a
  version is a pair (bot version, knowledge-base version or none);
- when its version has been dropped, the deployed version takes over and the
  turn processor restarts the conversation if its scenarios no longer fit.
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from chatcore.actions.builtin import builtin_registry
from chatcore.bot.changes import summarize
from chatcore.bot.loader import BotError, bot_from_document, parse_document, read_document, validate
from chatcore.bot.repository import BotRepository, Deployment, KbVersionInfo, RevisionConflict, VersionInfo
from chatcore.bot.schema import BotDefinition
from chatcore.dialogue.events import SessionStarted
from chatcore.dialogue.turn import TurnProcessor
from chatcore.dialogue.values import ActionFn, ActionRegistry
from chatcore.knowledge import (
    CosineJudge,
    Judge,
    KbAnswer,
    KbResult,
    Knowledge,
    KnowledgeError,
    KnowledgeIndex,
    compose,
    kb_version,
    parse_kb,
)
from chatcore.nlu.entities import EntityExtractor, bare, normalize
from chatcore.nlu.interpreter import NluInterpreter
from chatcore.nlu.model import CachedEmbedder, Embedder, ModelMismatch, NluModel
from chatcore.nlu.train import ConfusablePair, Report, confusable_intents, fit_only, train
from chatcore.runtime import BotRuntime
from chatcore.skills import SkillClient, SkillOutcome, Transport, error_dict, post_json, user_key
from chatcore.store.base import TrackerStore


class Host(Protocol):
    store: TrackerStore

    def bots(self) -> dict[str, str]:
        """bot id -> live version."""
        ...

    def runtime(self, bot_id: str) -> BotRuntime | None:
        """The live runtime of a bot, or None if unknown."""
        ...

    def runtime_for_session(self, bot_id: str, session_key: str) -> BotRuntime | None:
        """The runtime that should answer this session's next turn."""
        ...


class StaticHost:
    def __init__(self, runtimes: dict[str, BotRuntime], store: TrackerStore) -> None:
        self.runtimes, self.store = runtimes, store

    def bots(self) -> dict[str, str]:
        return {k: v.bot.version for k, v in self.runtimes.items()}

    def runtime(self, bot_id: str) -> BotRuntime | None:
        return self.runtimes.get(bot_id)

    def runtime_for_session(self, bot_id: str, session_key: str) -> BotRuntime | None:
        return self.runtimes.get(bot_id)


log = logging.getLogger(__name__)
TEST_DIR = "_test"  # artifacts/_test/<bot>/<version>: builder test builds (not a valid bot id)
KB_DIR = "kb"  # artifacts/<bot>/kb/<knowledge-base version>: its index (bot versions are 12 hex characters)
Key = tuple[str, str | None]  # (bot version, knowledge-base version)
KB_CACHE = 16  # knowledge-base indexes kept for lookups
PURGE_DELAY = dt.timedelta(days=7)  # how long a deleted bot can still be restored
PURGE_INTERVAL = 3600.0  # seconds between purge sweeps while serving
# An empty bot: only the responses the engine itself needs, and one unreachable scenario (a bot
# document must have at least one). check_draft() finds no errors or warnings in it.
EMPTY_BOT_RESPONSES = {
    "fallback": "무슨 말씀인지 잘 몰랐어요. 다른 말로 말씀해 주시겠어요?",
    "cancelled": "하던 작업을 멈췄어요.",
    "restarted": "처음부터 다시 시작할게요.",
    "error": "문제가 생겨서 답하지 못했어요.",
    "clarify": "혹시 이 중에 원하시는 게 있나요?",
    "input_invalid": "메시지가 비어 있거나 너무 길어요. 1,000자 이내로 다시 입력해 주세요.",
    "welcome": "안녕하세요! 무엇을 도와드릴까요?",
}


def _empty_document(bot_id: str, label: str) -> dict[str, Any]:
    return {
        "id": bot_id,
        "label": label,
        "responses": {rid: {"text": text} for rid, text in EMPTY_BOT_RESPONSES.items()},
        # a bot needs at least one block: the welcome block (shown when a user opens the chat), as Kakao starts
        "scenarios": {"welcome": {"label": "웰컴", "steps": [{"respond": "welcome"}]}},
        "system": {"welcome": "welcome"},
    }


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat(timespec="seconds")


class NoDraft(Exception):
    """The bot has no draft to train (e.g. it only has imported versions)."""


class NoEarlierDeployment(Exception):
    """Rollback asked for, but the bot has never been deployed with a different version."""


class NoKnowledgeBase(Exception):
    """The bot has no knowledge-base draft (to build) or no built version (to search)."""


class BotExists(Exception):
    """create_bot: the id already has a draft or versions, or is waiting to be deleted."""


class CopySourceNotFound(KeyError):
    """create_bot: copy_from names a bot with no draft, or one that does not exist, or is waiting
    to be deleted."""


class AlreadyDeleting(Exception):
    """delete_bot: a deletion is already pending for this bot."""


class NoDeletionPending(KeyError):
    """restore_bot: no deletion is pending for this bot."""


class BotDeleting(Exception):
    """The bot is waiting to be deleted: nothing may change it, train it or serve it until restored."""


@dataclass(frozen=True)
class KbDraftCheck:
    version: str | None  # None when the document is not a valid knowledge base
    errors: list[str]
    revision: int


def unique_sentences(texts: list[str]) -> list[tuple[str, str]]:
    """(key, text) per distinct sentence (first spelling wins), for marks."""
    seen: dict[str, str] = {}
    for t in texts:
        seen.setdefault(bare(t), t)
    return list(seen.items())


class BadExample(ValueError):
    """Sentences that cannot be examples: nothing left after trimming, or template braces."""

    def __init__(self, texts: list[str]) -> None:
        super().__init__(f"cannot be examples: {texts}")
        self.texts = texts


class ExampleTaken(ValueError):
    """Sentences that are already examples of another intent (one sentence cannot train two)."""

    def __init__(self, texts: list[str]) -> None:
        super().__init__(f"already examples of another intent: {texts}")
        self.texts = texts


class NotTrained(ValueError):
    """Deploying a version whose model is not in artifacts/ (train it first)."""


class BelowBar(ValueError):
    def __init__(self, report: Report, min_f1: float) -> None:
        super().__init__(f"macro-F1 {report.macro_f1:.3f} is below {min_f1}; not saved")
        self.report = report


@dataclass(frozen=True)
class DraftCheck:
    version: str | None  # None when the draft does not parse
    errors: list[str]
    warnings: list[str]
    revision: int = 0  # the saved draft's revision


class BotHost:
    def __init__(self, repository: BotRepository, store: TrackerStore, embedder: Embedder, artifacts: Path,
                 actions: Callable[[], ActionRegistry] = builtin_registry, keep: int = 2, min_f1: float = 0.90,
                 bulk_embedder: Embedder | None = None, test_keep: int = 2, test_ttl: float = 1800.0,
                 skill_allow_private: bool = False, skill_transport: Transport = post_json,
                 judge: Judge | None = None,
                 clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC)) -> None:
        """bulk_embedder: the same model for training (e.g. BatchedEmbedder.bulk(), so chat goes first).
        test_keep / test_ttl: builder test builds kept loaded, and for how long after their last use.
        skill_allow_private: let HTTP skills call internal addresses (development and tests only).
        judge: decides whether a knowledge-base candidate answers (default: cosine, for hashing mode).
        clock: now() for bot deletion/purge timing (tests pin it to control when purge_after is due)."""
        if keep < 1 or test_keep < 1 or test_ttl <= 0:
            raise ValueError("keep and test_keep must be >= 1 and test_ttl > 0")
        self.repository, self.store, self.embedder, self.artifacts = repository, store, embedder, artifacts
        self.bulk = bulk_embedder or embedder
        self.test_keep, self.test_ttl, self.clock = test_keep, test_ttl, clock
        # (bot, build) -> (runtime, its own directory, last use); separate from _loaded
        self._tests: OrderedDict[tuple[str, str], tuple[BotRuntime, Path, float]] = OrderedDict()
        self.actions, self.keep, self.min_f1 = actions(), keep, min_f1
        self.skill_allow_private, self.skill_transport = skill_allow_private, skill_transport
        self.judge: Judge = judge or CosineJudge()
        self._loaded: dict[str, OrderedDict[Key, BotRuntime]] = {}  # bot -> (version, kb) -> runtime (LRU)
        self._live: dict[str, Key] = {}
        self._indexes: OrderedDict[tuple[str, str], KnowledgeIndex] = OrderedDict()  # (bot, kb version), LRU
        self._deleting: set[str] = set()  # bots waiting to be deleted (mirrors repository.deletions(); _lock)
        self._index_locks: dict[tuple[str, str], threading.Lock] = {}  # one load or rebuild per version at a time
        self._lock = threading.RLock()  # guards _loaded/_live; held only briefly
        self._deploying = threading.RLock()  # one deploy/rollback at a time, so memory matches the repository

    # ------------------------------------------------------------ serving

    def bots(self) -> dict[str, str]:
        with self._lock:
            return {bot: key[0] for bot, key in self._live.items()}

    def live(self, bot_id: str) -> Key | None:
        """(bot version, knowledge-base version) that new sessions get."""
        with self._lock:
            return self._live.get(bot_id)

    def runtime(self, bot_id: str) -> BotRuntime | None:
        with self._lock:
            key = self._live.get(bot_id)
            return self._loaded[bot_id][key] if key else None

    def runtime_for_session(self, bot_id: str, session_key: str) -> BotRuntime | None:
        started = next((e for e in reversed(self.store.events(session_key)) if isinstance(e, SessionStarted)), None)
        pinned: Key | None = (started.bot_version, started.kb_version) if started is not None else None
        with self._lock:
            loaded = self._loaded.get(bot_id)
            if loaded is None or bot_id not in self._live:
                return None
            if pinned is not None and pinned in loaded:
                loaded.move_to_end(pinned)
                return loaded[pinned]
            return loaded[self._live[bot_id]]

    def start(self) -> None:
        """Permanently purge bots past their 7-day deletion grace period, then load every
        remaining bot's deployed version (server start; a bot waiting to be deleted is not
        loaded). A version whose model files are missing (a new machine on the same database) is
        trained again from its stored document. A bot that still cannot load is logged and left
        out, so it does not block the others. Test builds of an earlier run are gone with its
        memory, so their files are removed."""
        shutil.rmtree(self.artifacts / TEST_DIR, ignore_errors=True)
        self.purge_expired()
        pending = set(self.repository.deletions())
        with self._lock:
            self._deleting = set(pending)
        for bot_id in self.repository.bots():
            if bot_id not in pending:
                self._start_one(bot_id)

    def _start_one(self, bot_id: str) -> Key | None:
        """Load and activate bot_id's deployed pair; the key activated, or None if it has never
        been deployed or could not be loaded (logged)."""
        history = self.repository.history(bot_id)
        if not history:
            return None
        live = history[0]
        kb = live.kb_version
        if kb is not None:
            try:
                self.kb_index(bot_id, kb)
            except Exception:  # the bot itself still answers; its blocks do not depend on the knowledge base
                log.exception("could not load knowledge base %s@%s; serving the bot without it", bot_id, kb)
                kb = None
        try:
            try:
                runtime = self._build(bot_id, live.version, kb)
            except NotTrained:
                log.warning("%s@%s has no model files; training it again", bot_id, live.version)
                bot = bot_from_document(self.repository.get_version(bot_id, live.version), self.actions.names())
                train(bot, CachedEmbedder(self.embedder), self.artifacts)  # it cleared the bar when stored
                runtime = self._build(bot_id, live.version, kb)
        except Exception:
            log.exception("could not load %s@%s; the bot is not served", bot_id, live.version)
            return None
        key = (live.version, kb)
        self._activate(bot_id, key, runtime)
        return key

    # ------------------------------------------------------------ builder

    def check_draft(self, document: dict[str, Any]) -> DraftCheck:
        try:
            bot = parse_document(document)
        except BotError as exc:
            return DraftCheck(None, exc.problems, [])
        report = validate(bot, self.actions.names())
        return DraftCheck(bot.version, report.errors, report.warnings)

    def save_draft(self, bot_id: str, document: dict[str, Any], expected_revision: int | None = None) -> DraftCheck:
        """Save even an invalid draft (a builder saves work in progress); report its problems and
        record what changed (작업이력). RevisionConflict when expected_revision is given and the
        stored draft moved on (then the summary, made against the draft we read, is not stored)."""
        if document.get("id") != bot_id:
            raise BotError([f"the document's id must be {bot_id!r}"])
        self._refuse_if_deleting(bot_id)
        check = self.check_draft(document)
        before = self.repository.draft(bot_id)
        changes = summarize(before.document if before is not None else None, document)
        if before is not None and expected_revision is not None and before.revision != expected_revision:
            raise RevisionConflict(before.revision)
        revision = self.repository.save_draft(bot_id, document, expected_revision, [dict(c) for c in changes])
        return DraftCheck(check.version, check.errors, check.warnings, revision)

    def add_examples(self, bot_id: str, texts: list[str], intent: str, expected_revision: int) -> DraftCheck:
        """Add sentences from the training queue to an intent of the draft as examples (ones it
        already has are skipped) and mark them added. KeyError: no draft or no such intent;
        ExampleTaken: a sentence is already an example of another intent; RevisionConflict as save_draft."""
        draft = self.repository.draft(bot_id)
        if draft is None:
            raise KeyError(bot_id)
        document = copy.deepcopy(draft.document)
        intents = document.get("intents")
        if not isinstance(intents, dict):
            raise KeyError(intent)
        target = intents.get(intent)
        if not isinstance(target, dict) or not isinstance(target.get("examples"), list):
            raise KeyError(intent)
        bad = sorted({t for t in texts if not bare(t) or "{" in t or "}" in t})
        if bad:  # empty after trimming, or template syntax ("{location}") that training would expand
            raise BadExample(bad)
        owner = {bare(e): name for name, spec in intents.items()
                 if isinstance(spec, dict) and isinstance(spec.get("examples"), list)
                 for e in spec["examples"] if isinstance(e, str)}
        taken = sorted({t for t in texts if owner.get(bare(t), intent) != intent})
        if taken:
            raise ExampleTaken(taken)
        known = {bare(e) for e in target["examples"] if isinstance(e, str)}
        for text in texts:
            if bare(text) not in known:
                target["examples"].append(" ".join(text.split()))
                known.add(bare(text))
        check = self.save_draft(bot_id, document, expected_revision)
        self.repository.mark_utterances(bot_id, unique_sentences(texts), "added", intent)
        return check

    def train_draft(self, bot_id: str) -> tuple[VersionInfo, Report]:
        """Validate and train the draft; store it as a version if it clears min_f1."""
        self._refuse_if_deleting(bot_id)
        document = self.repository.get_draft(bot_id)
        if document is None:
            raise NoDraft(bot_id)
        bot = bot_from_document(document, self.actions.names())
        directory, report = train(bot, CachedEmbedder(self.bulk), self.artifacts, min_f1=self.min_f1)
        if directory is None:
            raise BelowBar(report, self.min_f1)
        self._refuse_if_deleting(bot_id)  # deleted while it trained: store nothing
        return self.repository.add_version(bot_id, bot.version, document, report.macro_f1), report

    def confusable(self, bot_id: str, build: str) -> list[ConfusablePair]:
        """Confusable intent pairs of a test build (the document it was built from, not the draft
        as it is now); [] when the build is unknown or expired."""
        runtime = self.test_runtime(bot_id, build)
        return self.confusable_of(runtime.bot) if runtime is not None else []

    def confusable_of(self, bot: BotDefinition) -> list[ConfusablePair]:
        return confusable_intents(bot, self.bulk)

    def deploy(self, bot_id: str, version: str, note: str = "", kb_version: str | None = None) -> None:
        """Build the runtime first (NotTrained / VersionNotFound / BotError leave everything as
        it was), record the deployment, then switch new sessions to it. kb_version: the
        knowledge base to answer from (None: none)."""
        with self._deploying:  # delete_bot takes the same lock: a deploy never brings a deleted bot back
            self._refuse_if_deleting(bot_id)
            runtime = self._build(bot_id, version, kb_version)
            self.repository.deploy(bot_id, version, note, kb_version)
            self._activate(bot_id, (version, kb_version), runtime)

    def rollback(self, bot_id: str) -> Deployment:
        """Re-deploy the pair (bot version, knowledge-base version) that was live before the current one."""
        with self._deploying:
            history = self.repository.history(bot_id)
            current = (history[0].version, history[0].kb_version) if history else None
            previous = next((d for d in history[1:] if (d.version, d.kb_version) != current), None)
            if previous is None:
                raise NoEarlierDeployment(bot_id)
            self.deploy(bot_id, previous.version, "rollback", previous.kb_version)
            return previous

    # ------------------------------------------------------------ create / delete / restore / purge

    def create_bot(self, bot_id: str, label: str, copy_from: str | None = None) -> DraftCheck:
        """A new draft (not deployed): empty (just the required responses) or, with copy_from, a
        deep copy of that bot's draft (and its knowledge-base draft, if any). BotExists: bot_id
        already has a draft/versions or is waiting to be deleted. CopySourceNotFound: copy_from
        has no draft, does not exist, or is waiting to be deleted."""
        with self._deploying:
            if bot_id in self.repository.bots() or bot_id in self.repository.deletions():
                raise BotExists(bot_id)
            if copy_from is None:
                document = _empty_document(bot_id, label)
            else:
                if copy_from in self.repository.deletions():
                    raise CopySourceNotFound(copy_from)
                source = self.repository.draft(copy_from)
                if source is None:
                    raise CopySourceNotFound(copy_from)
                document = copy.deepcopy(source.document)
                document["id"], document["label"] = bot_id, label
            check = self.save_draft(bot_id, document)
            if copy_from is not None:
                kb_draft = self.repository.kb_draft(copy_from)
                if kb_draft is not None:
                    self.save_kb_draft(bot_id, copy.deepcopy(kb_draft.document))
            return check

    def delete_bot(self, bot_id: str) -> str:
        """Request bot_id's deletion: out of service immediately (chat 404s, test builds dropped),
        permanently purged after PURGE_DELAY unless restored first. Returns purge_after (ISO 8601
        UTC). KeyError: unknown bot. AlreadyDeleting: a deletion is already pending."""
        with self._deploying:
            if bot_id in self.repository.deletions():
                raise AlreadyDeleting(bot_id)
            if bot_id not in self.repository.bots():
                raise KeyError(bot_id)
            purge_after = _iso(self.clock() + PURGE_DELAY)
            self.repository.request_deletion(bot_id, purge_after)
            with self._lock:
                self._deleting.add(bot_id)
            self._evict(bot_id)
            return purge_after

    def restore_bot(self, bot_id: str) -> Key | None:
        """Cancel a pending deletion and serve the bot's deployed pair again (None: it was never
        deployed). NoDeletionPending: no deletion is pending for this bot."""
        with self._deploying:
            if not self.repository.cancel_deletion(bot_id):
                raise NoDeletionPending(bot_id)
            with self._lock:
                self._deleting.discard(bot_id)
            return self._start_one(bot_id)

    def purge_expired(self) -> list[str]:
        """Permanently delete every bot whose purge_after has passed: its conversations,
        artifacts/<bot>/ and every repository row — the deletion record last, so a purge that fails
        midway is retried next time. Each bot is re-checked under the deploy lock (it may have been
        restored, or deleted again with a new date, since the list was read) and fails on its own.
        Returns the bot ids purged."""
        due = [bot_id for bot_id, purge_after in self.repository.deletions().items()
               if dt.datetime.fromisoformat(purge_after) <= self.clock()]
        purged = []
        for bot_id in due:
            try:
                with self._deploying:
                    purge_after = self.repository.deletions().get(bot_id)
                    if purge_after is None or dt.datetime.fromisoformat(purge_after) > self.clock():
                        continue
                    self._evict(bot_id)
                    self.store.delete_sessions(f"{bot_id}:")
                    self.store.delete_sessions(f"{bot_id}~")
                    shutil.rmtree(self.artifacts / bot_id, ignore_errors=True)
                    with self._lock:
                        for key in [k for k in self._indexes if k[0] == bot_id]:
                            del self._indexes[key]
                    self.repository.purge_bot(bot_id)  # last: it removes the deletion record
                    with self._lock:
                        self._deleting.discard(bot_id)
                    purged.append(bot_id)
            except Exception:  # one bot's failure must not stop the others (or the server start)
                log.exception("could not purge %s; trying again later", bot_id)
        return purged

    def start_purge_daemon(self, interval: float = PURGE_INTERVAL) -> Callable[[], None]:
        """Run purge_expired() every `interval` seconds in a daemon thread until the returned
        function is called (stops the thread)."""
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(interval):
                try:
                    self.purge_expired()
                except Exception:
                    log.exception("purge_expired failed")

        threading.Thread(target=loop, daemon=True).start()
        return stop.set

    def _refuse_if_deleting(self, bot_id: str) -> None:
        """BotDeleting when bot_id is waiting to be deleted (writes, training, deploys, test chats)."""
        with self._lock:
            if bot_id in self._deleting:
                raise BotDeleting(bot_id)

    def _evict(self, bot_id: str) -> None:
        """Take bot_id out of service right away: new sessions/chat calls see it as unknown, and
        its builder test builds are dropped. Lock _deploying expected held by the caller."""
        with self._lock:
            self._live.pop(bot_id, None)
            self._loaded.pop(bot_id, None)
        self._drop_tests_of(bot_id)

    def _drop_tests_of(self, bot_id: str) -> None:
        with self._lock:
            gone = [key for key in self._tests if key[0] == bot_id]
            roots = [self._tests.pop(key)[1] for key in gone]
        _remove(roots)

    # ------------------------------------------------------------ knowledge base

    def save_kb_draft(self, bot_id: str, document: dict[str, Any],
                      expected_revision: int | None = None) -> KbDraftCheck:
        """Save even an invalid knowledge base (work in progress) and report its problems."""
        try:
            parse_kb(document)
            version: str | None = kb_version(document)
            errors: list[str] = []
        except KnowledgeError as exc:
            version, errors = None, exc.problems
        self._refuse_if_deleting(bot_id)
        revision = self.repository.save_kb_draft(bot_id, document, expected_revision)
        return KbDraftCheck(version, errors, revision)

    def build_kb(self, bot_id: str) -> KbVersionInfo:
        """Index the knowledge-base draft and store it as a version. NoKnowledgeBase without a draft;
        KnowledgeError when the draft is invalid."""
        self._refuse_if_deleting(bot_id)
        draft = self.repository.kb_draft(bot_id)
        if draft is None:
            raise NoKnowledgeBase(bot_id)
        index = KnowledgeIndex.build(draft.document, self.bulk)
        self._refuse_if_deleting(bot_id)  # deleted while it indexed: write nothing
        index.save(self.artifacts / bot_id / KB_DIR / index.version)
        with self._lock:
            self._remember_index(bot_id, index)
        return self.repository.add_kb_version(bot_id, index.version, draft.document, len(index.faq), len(index.chunks))

    def kb_index(self, bot_id: str, version: str) -> KnowledgeIndex:
        """A stored knowledge-base version's index: from memory, its files, or built again from the stored
        document when the files are missing or stale (a new machine). VersionNotFound when unknown."""
        with self._lock:
            index = self._indexes.get((bot_id, version))
            if index is not None:
                self._indexes.move_to_end((bot_id, version))
                return index
            loading = self._index_locks.setdefault((bot_id, version), threading.Lock())
        with loading:
            with self._lock:  # another caller may have loaded it while we waited
                index = self._indexes.get((bot_id, version))
            return index if index is not None else self._load_index(bot_id, version)

    def _load_index(self, bot_id: str, version: str) -> KnowledgeIndex:
        directory = self.artifacts / bot_id / KB_DIR / version
        try:
            index = KnowledgeIndex.load(directory, version, self.embedder)
        except (FileNotFoundError, ValueError) as exc:
            document = self.repository.get_kb_version(bot_id, version)
            log.warning("knowledge base %s@%s: %s; indexing it again", bot_id, version, exc)
            index = KnowledgeIndex.build(document, self.bulk)
            if index.version != version:  # the stored document must hash to its version
                raise KnowledgeError([f"stored knowledge base {version} hashes to {index.version}"]) from exc
            index.save(directory)
        with self._lock:
            self._remember_index(bot_id, index)
        return index

    def search_kb(self, bot_id: str, query: str, version: str | None = None) -> tuple[str, KbResult, KbAnswer | None]:
        """Search a built version (default: the newest) as a conversation would: (version, result, answer).
        NoKnowledgeBase when nothing is built; VersionNotFound for an unknown version."""
        if version is None:
            versions = self.repository.kb_versions(bot_id)
            if not versions:
                raise NoKnowledgeBase(bot_id)
            version = versions[0].version
        index = self.kb_index(bot_id, version)
        text = " ".join(query.split())
        result = index.search(text, self.embedder.encode([normalize(text)])[0], self.judge)
        return version, result, compose(result, index.settings)

    def _remember_index(self, bot_id: str, index: KnowledgeIndex) -> None:
        """Lock held. A lookup cache (runtimes hold their own index): the KB_CACHE most recently used."""
        self._indexes[(bot_id, index.version)] = index
        self._indexes.move_to_end((bot_id, index.version))
        while len(self._indexes) > KB_CACHE:
            self._indexes.popitem(last=False)

    # ------------------------------------------------------------ test builds

    def build_test(self, bot_id: str) -> str:
        """Fit the current draft without evaluation (one example per intent is enough), with the
        knowledge-base draft when there is one, load it into the test cache and return its build id
        (the draft's version). BotError for an invalid draft, KnowledgeError for an invalid
        knowledge base, TrainingError when it cannot be fitted."""
        self._refuse_if_deleting(bot_id)
        document = self.repository.get_draft(bot_id)
        if document is None:
            raise NoDraft(bot_id)
        bot = bot_from_document(document, self.actions.names())
        kb_draft = self.repository.kb_draft(bot_id)
        kb = KnowledgeIndex.build(kb_draft.document, self.bulk) if kb_draft is not None else None  # KnowledgeError
        root = self.artifacts / TEST_DIR / uuid.uuid4().hex  # its own directory: nothing else deletes it
        try:
            fit_only(bot, CachedEmbedder(self.bulk), root)
            runtime = self._load(bot, root, kb)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        with self._lock:
            if bot_id in self._deleting:  # deleted while it was fitted (same lock as delete_bot's eviction)
                shutil.rmtree(root, ignore_errors=True)
                raise BotDeleting(bot_id)
            replaced = self._tests.pop((bot_id, bot.version), None)
            self._tests[(bot_id, bot.version)] = (runtime, root, time.monotonic())
            gone = self._drop_tests(lambda used: len(self._tests) > self.test_keep)
        _remove([*gone, *([replaced[1]] if replaced else [])])
        return bot.version

    def test_runtime(self, bot_id: str, build: str) -> BotRuntime | None:
        """The test build's runtime, or None when it is unknown or expired (never the live one)."""
        with self._lock:
            now = time.monotonic()
            gone = self._drop_tests(lambda used: now - used > self.test_ttl)
            entry = self._tests.get((bot_id, build))
            if entry is not None:
                self._tests[(bot_id, build)] = (entry[0], entry[1], now)
                self._tests.move_to_end((bot_id, build))
        _remove(gone)  # outside the lock: chat requests take it too
        return entry[0] if entry is not None else None

    def _drop_tests(self, drop: Callable[[float], bool]) -> list[Path]:
        """Forget test builds (oldest first) while drop(last_use) says so; returns their
        directories for the caller to delete after releasing the lock. Lock held."""
        gone = []
        for key, (_, root, used) in list(self._tests.items()):
            if drop(used):
                del self._tests[key]
                gone.append(root)
        return gone

    def seed(self, bot_dir: Path) -> str:
        """First start: import a bot directory as draft and deploy it (reusing a model from
        `make train` when there is one) unless the repository already has it. A bot waiting to be
        deleted is left alone (not recreated). Returns the bot id."""
        document = read_document(bot_dir)
        bot_id = str(document.get("id"))
        if bot_id in self.repository.deletions():
            return bot_id
        if self.repository.get_draft(bot_id) is None:
            self.repository.save_draft(bot_id, document)
        if not self.repository.deployments(bot_id):
            self.deploy(bot_id, self._trained_draft(bot_id))
        return bot_id

    # ------------------------------------------------------------ internals

    def _activate(self, bot_id: str, key: Key, runtime: BotRuntime) -> None:
        with self._lock:
            loaded = self._loaded.setdefault(bot_id, OrderedDict())
            loaded.setdefault(key, runtime)  # sessions pinned to an already loaded copy keep it
            loaded.move_to_end(key)
            self._live[bot_id] = key
            while len(loaded) > self.keep + 1:  # the live version plus `keep` others
                oldest = next(k for k in loaded if k != key)
                del loaded[oldest]

    def _trained_draft(self, bot_id: str) -> str:
        """The draft as a stored version: reuse a model already in artifacts/ (`make train`),
        otherwise train it now."""
        document = self.repository.get_draft(bot_id)
        if document is None:
            raise KeyError(bot_id)
        bot = bot_from_document(document, self.actions.names())
        try:  # only the files are checked here; deploy() loads the model once
            meta = NluModel.check(bot, self.artifacts / bot.id / bot.version, self.embedder)
        except ModelMismatch:
            return self.train_draft(bot_id)[0].version
        return self.repository.add_version(bot_id, bot.version, document, meta.get("macro_f1")).version

    def _build(self, bot_id: str, version: str, kb: str | None = None) -> BotRuntime:
        bot = bot_from_document(self.repository.get_version(bot_id, version), self.actions.names())
        return self._load(bot, None, self.kb_index(bot_id, kb) if kb is not None else None)

    def _load(self, bot: BotDefinition, root: Path | None = None, kb: KnowledgeIndex | None = None) -> BotRuntime:
        try:
            model = NluModel.load(bot, (root or self.artifacts) / bot.id / bot.version, self.embedder,
                                  EntityExtractor(bot))
        except ModelMismatch as exc:
            raise NotTrained(str(exc)) from exc
        # model.bot carries the thresholds meta.json was trained with (NluModel.load, docs/
        # engine-core-plan.md "빌더 단계 E-B4"), not necessarily the document's own numbers.
        served = model.bot
        knowledge = Knowledge(kb, self.judge, self.embedder) if kb is not None and not kb.empty else None
        processor = TurnProcessor(served, self.store, NluInterpreter(model.parse, knowledge),
                                  self.actions_for(served, knowledge), kb_version=kb.version if kb else None)
        return BotRuntime(served, processor, model)

    # ------------------------------------------------------------ HTTP skills

    def skill_client(self, bot: BotDefinition) -> SkillClient:
        return SkillClient(bot.id, bot.version, lambda name: self.repository.secret(bot.id, name),
                           lambda error: self.repository.add_skill_error(bot.id, error_dict(error)),
                           self.skill_allow_private, self.skill_transport)

    def actions_for(self, bot: BotDefinition, knowledge: Knowledge | None = None) -> ActionRegistry:
        """The built-in actions plus the bot's HTTP skills (each with its own timeout); `knowledge`
        searches the bot's knowledge base when it has one."""
        if not bot.skills and knowledge is None:
            return self.actions
        client = self.skill_client(bot)
        skills = {sid: (client.action(sid, skill), skill.timeout_ms / 1000 + 0.5) for sid, skill in bot.skills.items()}
        builtin = self.actions.names()
        replace: dict[str, tuple[ActionFn, float | None]] = {sid: fn for sid, fn in skills.items() if sid in builtin}
        if knowledge is not None and "knowledge" not in replace:  # the bot's own skill of that name wins
            replace["knowledge"] = (knowledge.action(), None)
        return self.actions.extended({sid: fn for sid, fn in skills.items() if sid not in builtin}, replace)

    def test_skill(self, bot_id: str, skill_id: str, params: dict[str, str], utterance: str = "") -> SkillOutcome:
        """Run a skill of the current draft once (its test_url when it has one). NoDraft; BotError when
        the draft does not parse; KeyError for an unknown skill."""
        document = self.repository.get_draft(bot_id)
        if document is None:
            raise NoDraft(bot_id)
        bot = parse_document(document)
        skill = bot.skills.get(skill_id)
        if skill is None:
            raise KeyError(skill_id)
        return self.skill_client(bot).call(skill_id, skill, params, utterance, user_key(f"{bot_id}~skill-test"),
                                           test=True)


def _remove(directories: list[Path]) -> None:
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)
