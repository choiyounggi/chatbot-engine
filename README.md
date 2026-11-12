# chatcore

**English** | [한국어](README.ko.md)

An LLM-free, scenario-based chatbot engine that runs on your own machine. Bots are data —
intents with example sentences, entities with synonyms, scenarios, response templates — and the
engine understands Korean input with a frozen sentence-embedding model plus a small classifier,
then follows the scenario deterministically. A builder UI for editing that data comes later.

> Status: builder step E-E — HTTP API (`make serve`), terminal chat (`make chat`) and a builder API (`/v1/builder/...`: drafts with ETag, background training jobs with SSE progress, test builds with a test chat, deploy with notes and version numbers, roll back, an edit history, a training queue of what users said grouped by sentence with "add as an example" / ignore, HTTP skills (actions a bot defines by URL, with write-only secrets, internal addresses blocked, a test run and an error log), a knowledge base (FAQ, documents and PDFs answered when no block fits, or offered as suggestions; a cross-encoder decides whether to answer; versioned apart from the bot and deployed with it), analytics (users, sessions, block calls, knowledge-base answer rate, retention, by Korean day), creating bots (empty or a copy) and deleting them after a 7-day wait, and ML settings: a 5-step similarity level that sets thresholds per bot, exact-only intents, confusable-intent warnings, optional keyword match). Bots can have Kakao-style welcome / escape blocks, a session timeout, optional and validated slots, quick replies and buttons that start a block directly, image / card / list / carousel responses, and 16 system entities (dates, times, numbers, periods, ordinals, ages, percents, URLs, units, free text) over the scenario engine with Korean NLU. Set `CHATCORE_ADMIN_TOKEN` to protect the builder API. The builder UI lives in [chatbot-builder](https://github.com/choiyounggi/chatbot-builder).
> Design and roadmap: [docs/engine-core-plan.md](docs/engine-core-plan.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and `make`. Python 3.14 is installed by uv.

```sh
make setup   # uv sync --locked
make check   # ruff + mypy (strict, src/) + pytest with coverage — the same as CI
make test-db # also the Postgres store tests (CHATCORE_DB_PORT=… if 5432 is taken)
```

Optional:

```sh
make db-up   # Postgres 17 + pgvector on 127.0.0.1:5432 (Docker), used from Phase 2
make bench   # Phase 0 NLU benchmark — Apple Silicon, downloads models, ~10 min
make openapi # write docs/openapi.json — the builder generates its API types from it
# a fast test server without the model (CI, end-to-end tests; NOT for real users):
uv run python -m chatcore.api --bot bots/sample --embedder hashing --min-f1 0 --artifacts .e2e-artifacts
```

## Layout

| Path | What |
|---|---|
| `src/chatcore/` | engine package: `bot/` (definition, loader), `dialogue/` (commands, events, state, scenario engine, turns), `store/` (in-memory, Postgres) |
| `bots/sample/` | sample bot: `bot.yaml` (scenarios, responses), `nlu.yaml` (intents, entities) |
| `bench/` | NLU benchmarks and test sets |
| `docs/` | plan and benchmark reports |
