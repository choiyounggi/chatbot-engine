"""Print the HTTP API's OpenAPI schema (docs/openapi.json; the builder generates its types from it).

Run: `make openapi`. A test fails when docs/openapi.json is out of date.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from chatcore.api import create_app
from chatcore.bot.repository import InMemoryRepository
from chatcore.host import BotHost
from chatcore.jobs import JobRunner
from chatcore.nlu.hashing import HashingEmbedder
from chatcore.store.memory import InMemoryStore


def schema() -> dict[str, Any]:
    """The schema of the full API (chat + builder). Nothing is loaded or trained."""
    jobs = JobRunner()
    try:
        host = BotHost(InMemoryRepository(), InMemoryStore(), HashingEmbedder(), Path("artifacts"))
        return create_app(host, admin_token=None, jobs=jobs).openapi()
    finally:
        jobs.close()


def render() -> str:
    return json.dumps(schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":  # pragma: no cover
    sys.stdout.write(render())
