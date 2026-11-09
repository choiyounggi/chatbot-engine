# chatcore developer commands. Requires uv (https://docs.astral.sh/uv/).
.PHONY: setup lint typecheck test check bench db-up db-down

setup:  ## install Python 3.14 + locked dependencies into .venv
	uv sync --locked

lint:
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest -q

check: lint typecheck test  ## what CI runs

bench:  ## Phase 0 NLU benchmark (Apple Silicon, downloads models, ~10 min)
	uv run python bench/phase0.py

db-up:  ## local Postgres + pgvector (used from Phase 2)
	docker compose up -d --wait db

db-down:
	docker compose down
