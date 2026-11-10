"""Assemble running bots: definition + trained NLU model + actions + store."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from chatcore.actions.builtin import builtin_registry
from chatcore.bot.loader import load_bot
from chatcore.bot.repository import BotRepository, InMemoryRepository, PostgresRepository
from chatcore.bot.schema import BotDefinition
from chatcore.dialogue.turn import TurnProcessor
from chatcore.dialogue.values import ActionRegistry
from chatcore.nlu.entities import EntityExtractor
from chatcore.nlu.interpreter import NluInterpreter
from chatcore.nlu.model import Embedder, NluModel
from chatcore.store.base import TrackerStore
from chatcore.store.memory import InMemoryStore


@dataclass
class BotRuntime:
    bot: BotDefinition
    processor: TurnProcessor
    model: NluModel


def build_runtime(bot_dir: Path, artifacts: Path, embedder: Embedder, store: TrackerStore,
                  actions: ActionRegistry | None = None) -> BotRuntime:
    """Load one bot and its trained model (ModelMismatch if `make train` is needed)."""
    actions = actions or builtin_registry()
    bot = load_bot(bot_dir, actions=actions.names())
    model = NluModel.load(bot, artifacts / bot.id / bot.version, embedder, EntityExtractor(bot))
    # model.bot carries the thresholds meta.json was trained with (NluModel.load).
    served = model.bot
    return BotRuntime(served, TurnProcessor(served, store, NluInterpreter(model.parse), actions), model)


def default_repository() -> BotRepository:
    """Postgres when CHATCORE_DATABASE_URL is set, else in-memory (re-seeded from --bot on every start)."""
    url = os.environ.get("CHATCORE_DATABASE_URL")
    if not url:
        return InMemoryRepository()
    repository = PostgresRepository(url)
    repository.ensure_schema()
    return repository


def default_store() -> TrackerStore:
    """Postgres when CHATCORE_DATABASE_URL is set, else in-memory (sessions vanish on restart)."""
    url = os.environ.get("CHATCORE_DATABASE_URL")
    if not url:
        return InMemoryStore()
    from chatcore.store.postgres import PostgresStore

    store = PostgresStore(url)
    store.ensure_schema()
    return store
