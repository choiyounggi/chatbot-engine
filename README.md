# chatcore

**English** | [한국어](README.ko.md)

An LLM-free, scenario-based chatbot engine that runs on your own machine. Bots are data —
intents with example sentences, entities with synonyms, scenarios, response templates — and the
engine understands Korean input with a frozen sentence-embedding model plus a small classifier,
then follows the scenario deterministically. A builder UI for editing that data comes later.

> Status: Phase 1 (project skeleton). The engine core is built from Phase 2 on.
> Design and roadmap: [docs/engine-core-plan.md](docs/engine-core-plan.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and `make`. Python 3.14 is installed by uv.

```sh
make setup   # uv sync --locked
make check   # ruff + mypy (strict, src/) + pytest — the same as CI
```

Optional:

```sh
make db-up   # Postgres 17 + pgvector on 127.0.0.1:5432 (Docker), used from Phase 2
make bench   # Phase 0 NLU benchmark — Apple Silicon, downloads models, ~10 min
```

## Layout

| Path | What |
|---|---|
| `src/chatcore/` | engine package (empty until Phase 2) |
| `bots/sample/nlu.yaml` | sample bot NLU data (source of truth) |
| `bench/` | NLU benchmarks and test sets |
| `docs/` | plan and benchmark reports |
