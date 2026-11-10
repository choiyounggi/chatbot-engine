# chatcore developer commands. Requires uv (https://docs.astral.sh/uv/).
.PHONY: setup lint typecheck test test-db check train eval serve chat bench db-up db-down

setup:  ## install Python 3.14 + locked dependencies into .venv
	uv sync --locked

lint:
	uv run ruff check .

typecheck:
	uv run mypy

CHATCORE_DB_PORT ?= 5432
TEST_DB_URL = postgresql://chatcore:chatcore-local-only@127.0.0.1:$(CHATCORE_DB_PORT)/chatcore
COV = --cov=chatcore.bot --cov=chatcore.dialogue --cov=chatcore.nlu --cov=chatcore.actions --cov=chatcore.conversations --cov=chatcore.api --cov=chatcore.cli --cov=chatcore.runtime --cov=chatcore.serving --cov=chatcore.host --cov=chatcore.jobs --cov-report=term-missing:skip-covered --cov-fail-under=90

test:  ## unit + conversation tests; Postgres tests skip unless CHATCORE_TEST_DATABASE_URL is set
	uv run pytest -q --durations=25 $(COV)

test-db: db-up  ## also run the Postgres store tests against the compose database
	CHATCORE_TEST_DATABASE_URL=$(TEST_DB_URL) uv run pytest -q --durations=25 $(COV)

check: lint typecheck test  ## what CI runs

BOT ?= bots/sample

train:  ## train + evaluate the bot's intent classifier (downloads the embedding model once); fails if macro-F1 < 0.90
	uv run python -m chatcore.nlu.train $(BOT)

eval:  ## golden conversations through the real NLU model (after make train)
	uv run python -m chatcore.conversations $(BOT)

KEEP_WARM_MS ?=
serve:  ## HTTP API on 127.0.0.1:8000 (Postgres when CHATCORE_DATABASE_URL is set; KEEP_WARM_MS=20 keeps the GPU warm)
	uv run python -m chatcore.api --bot $(BOT) $(if $(KEEP_WARM_MS),--keep-warm-ms $(KEEP_WARM_MS))

chat:  ## talk to the bot in the terminal
	uv run python -m chatcore.cli $(BOT)

bench:  ## Phase 0 NLU benchmark (Apple Silicon, downloads models, ~10 min)
	uv run python bench/phase0.py

db-up:  ## local Postgres + pgvector on 127.0.0.1:$(CHATCORE_DB_PORT)
	CHATCORE_DB_PORT=$(CHATCORE_DB_PORT) docker compose up -d --wait db

db-down:
	docker compose down
