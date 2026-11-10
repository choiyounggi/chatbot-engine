"""Bot hosts: which runtime answers a session.

`StaticHost` serves fixed runtimes (tests, simple setups). `BotHost` serves
versions from a BotRepository and lets a builder train, deploy and roll back
without restarting (docs/engine-core-plan.md §3.1, Phase 6):

- a session stays on the version it started with while that version is
  loaded (the deployed one plus the `keep` most recently used others);
- when its version has been dropped, the deployed version takes over and the
  turn processor restarts the conversation if its scenarios no longer fit.
"""

from __future__ import annotations

import copy
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
from chatcore.bot.loader import BotError, bot_from_document, read_document, validate
from chatcore.bot.repository import BotRepository, RevisionConflict, VersionInfo
from chatcore.bot.schema import BotDefinition
from chatcore.dialogue.events import SessionStarted
from chatcore.dialogue.turn import TurnProcessor
from chatcore.dialogue.values import ActionRegistry
from chatcore.nlu.entities import EntityExtractor, bare
from chatcore.nlu.interpreter import NluInterpreter
from chatcore.nlu.model import CachedEmbedder, Embedder, ModelMismatch, NluModel
from chatcore.nlu.train import Report, fit_only, train
from chatcore.runtime import BotRuntime
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


class NoDraft(Exception):
    """The bot has no draft to train (e.g. it only has imported versions)."""


class NoEarlierDeployment(Exception):
    """Rollback asked for, but the bot has never been deployed with a different version."""


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
                 bulk_embedder: Embedder | None = None, test_keep: int = 2, test_ttl: float = 1800.0) -> None:
        """bulk_embedder: the same model for training (e.g. BatchedEmbedder.bulk(), so chat goes first).
        test_keep / test_ttl: builder test builds kept loaded, and for how long after their last use."""
        if keep < 1 or test_keep < 1 or test_ttl <= 0:
            raise ValueError("keep and test_keep must be >= 1 and test_ttl > 0")
        self.repository, self.store, self.embedder, self.artifacts = repository, store, embedder, artifacts
        self.bulk = bulk_embedder or embedder
        self.test_keep, self.test_ttl = test_keep, test_ttl
        # (bot, build) -> (runtime, its own directory, last use); separate from _loaded
        self._tests: OrderedDict[tuple[str, str], tuple[BotRuntime, Path, float]] = OrderedDict()
        self.actions, self.keep, self.min_f1 = actions(), keep, min_f1
        self._loaded: dict[str, OrderedDict[str, BotRuntime]] = {}  # bot -> version -> runtime (LRU)
        self._live: dict[str, str] = {}
        self._lock = threading.RLock()  # guards _loaded/_live; held only briefly
        self._deploying = threading.RLock()  # one deploy/rollback at a time, so memory matches the repository

    # ------------------------------------------------------------ serving

    def bots(self) -> dict[str, str]:
        with self._lock:
            return dict(self._live)

    def runtime(self, bot_id: str) -> BotRuntime | None:
        with self._lock:
            version = self._live.get(bot_id)
            return self._loaded[bot_id][version] if version else None

    def runtime_for_session(self, bot_id: str, session_key: str) -> BotRuntime | None:
        pinned = next((e.bot_version for e in reversed(self.store.events(session_key))
                       if isinstance(e, SessionStarted)), None)
        with self._lock:
            loaded = self._loaded.get(bot_id)
            if loaded is None or bot_id not in self._live:
                return None
            if pinned is not None and pinned in loaded:
                loaded.move_to_end(pinned)
                return loaded[pinned]
            return loaded[self._live[bot_id]]

    def start(self) -> None:
        """Load every bot's deployed version (server start). A version whose model files are
        missing (a new machine on the same database) is trained again from its stored document.
        A bot that still cannot load is logged and left out, so it does not block the others.
        Test builds of an earlier run are gone with its memory, so their files are removed."""
        shutil.rmtree(self.artifacts / TEST_DIR, ignore_errors=True)
        for bot_id in self.repository.bots():
            history = self.repository.deployments(bot_id)
            if not history:
                continue
            try:
                try:
                    runtime = self._build(bot_id, history[0])
                except NotTrained:
                    log.warning("%s@%s has no model files; training it again", bot_id, history[0])
                    bot = bot_from_document(self.repository.get_version(bot_id, history[0]), self.actions.names())
                    train(bot, CachedEmbedder(self.embedder), self.artifacts)  # it cleared the bar when stored
                    runtime = self._build(bot_id, history[0])
            except Exception:
                log.exception("could not load %s@%s; the bot is not served", bot_id, history[0])
                continue
            self._activate(bot_id, history[0], runtime)

    # ------------------------------------------------------------ builder

    def check_draft(self, document: dict[str, Any]) -> DraftCheck:
        from chatcore.bot.loader import parse_document

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
        owner = {bare(e): name for name, spec in intents.items() if isinstance(spec, dict)
                 for e in spec.get("examples", []) if isinstance(e, str)}
        taken = sorted({t for t in texts if owner.get(bare(t), intent) != intent})
        if taken:
            raise ExampleTaken(taken)
        known = {bare(e) for e in target["examples"] if isinstance(e, str)}
        for text in texts:
            if bare(text) not in known:
                target["examples"].append(" ".join(text.split()))
                known.add(bare(text))
        check = self.save_draft(bot_id, document, expected_revision)
        self.repository.mark_utterances(bot_id, [(bare(t), t) for t in texts], "added", intent)
        return check

    def train_draft(self, bot_id: str) -> tuple[VersionInfo, Report]:
        """Validate and train the draft; store it as a version if it clears min_f1."""
        document = self.repository.get_draft(bot_id)
        if document is None:
            raise NoDraft(bot_id)
        bot = bot_from_document(document, self.actions.names())
        directory, report = train(bot, CachedEmbedder(self.bulk), self.artifacts, min_f1=self.min_f1)
        if directory is None:
            raise BelowBar(report, self.min_f1)
        return self.repository.add_version(bot_id, bot.version, document, report.macro_f1), report

    def deploy(self, bot_id: str, version: str, note: str = "") -> None:
        """Build the runtime first (NotTrained / VersionNotFound / BotError leave everything as
        it was), record the deployment, then switch new sessions to it."""
        with self._deploying:
            runtime = self._build(bot_id, version)
            self.repository.deploy(bot_id, version, note)
            self._activate(bot_id, version, runtime)

    def rollback(self, bot_id: str) -> str:
        """Re-deploy the version that was live before the current one."""
        with self._deploying:
            history = self.repository.deployments(bot_id)
            previous = next((v for v in history[1:] if v != history[0]), None)
            if previous is None:
                raise NoEarlierDeployment(bot_id)
            self.deploy(bot_id, previous, "rollback")
            return previous

    # ------------------------------------------------------------ test builds

    def build_test(self, bot_id: str) -> str:
        """Fit the current draft without evaluation (one example per intent is enough), load it
        into the test cache and return its build id (the draft's version). BotError for an
        invalid draft, TrainingError when it cannot be fitted."""
        document = self.repository.get_draft(bot_id)
        if document is None:
            raise NoDraft(bot_id)
        bot = bot_from_document(document, self.actions.names())
        root = self.artifacts / TEST_DIR / uuid.uuid4().hex  # its own directory: nothing else deletes it
        try:
            fit_only(bot, CachedEmbedder(self.bulk), root)
            runtime = self._load(bot, root)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        with self._lock:
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
        `make train` when there is one) unless the repository already has it. Returns the bot id."""
        document = read_document(bot_dir)
        bot_id = str(document.get("id"))
        if self.repository.get_draft(bot_id) is None:
            self.repository.save_draft(bot_id, document)
        if not self.repository.deployments(bot_id):
            self.deploy(bot_id, self._trained_draft(bot_id))
        return bot_id

    # ------------------------------------------------------------ internals

    def _activate(self, bot_id: str, version: str, runtime: BotRuntime) -> None:
        with self._lock:
            loaded = self._loaded.setdefault(bot_id, OrderedDict())
            loaded.setdefault(version, runtime)  # sessions pinned to an already loaded copy keep it
            loaded.move_to_end(version)
            self._live[bot_id] = version
            while len(loaded) > self.keep + 1:  # the live version plus `keep` others
                oldest = next(v for v in loaded if v != version)
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

    def _build(self, bot_id: str, version: str) -> BotRuntime:
        return self._load(bot_from_document(self.repository.get_version(bot_id, version), self.actions.names()))

    def _load(self, bot: BotDefinition, root: Path | None = None) -> BotRuntime:
        try:
            model = NluModel.load(bot, (root or self.artifacts) / bot.id / bot.version, self.embedder,
                                  EntityExtractor(bot))
        except ModelMismatch as exc:
            raise NotTrained(str(exc)) from exc
        return BotRuntime(bot, TurnProcessor(bot, self.store, NluInterpreter(model.parse), self.actions), model)


def _remove(directories: list[Path]) -> None:
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)
