# Chess RAG — Project Context

Portfolio project for Solutions Engineering / Forward Deployed job applications.
Goal: demoable hybrid structured-search + RAG chess assistant, built in 1-2 weeks.

## Architecture (decided, do not re-litigate without reason)

Four layers, called by an agent via native tool-calling (not a hand-rolled classifier):

- **Layer 1 — Structured search**: `python-chess` + SQL over `games`/`moves` tables.
  Deterministic, no LLM judgment. Handles move/position/pattern queries and
  statistical aggregation ("where does the knight usually land in this ECO code").
- **Layer 2 — Vector RAG**: pgvector over `chunks` table (game annotations + public
  domain book text). Handles conceptual/strategic questions and cross-source synthesis.
- **Layer 3 — Stockfish**: `python-chess` UCI integration for ground-truth eval.
  Used to keep LLM commentary grounded instead of letting it judge soundness itself.
- **Layer 4 — Personalized analysis**: user uploads a PGN, we run structural
  similarity search against the corpus, synthesize a comparison.

Also supported, no new infra:
- **Opening profile queries**: Layer 1 stats (piece-placement frequency) + Layer 2
  prose (plans), synthesized together. This is the flagship demo query. Built.
- **Corpus/study recommendations**: a separate small agent (`src/recommendation/`)
  decides whether a Lichess study or a ChessBase master game is worth surfacing
  alongside a chat answer, and a trained quality classifier
  (`scripts/train_quality_classifier.py`, `models/quality_classifier.joblib`)
  filters the Lichess study pool before it's ever searchable. Built.
- **Recommended-game replay**: a surfaced ChessBase game can be played through
  move by move on the board (arrow keys or on-screen buttons), read-only, using
  the same board component as free play. Built.

Designed but not yet implemented — mentioned here because the schema and some
supporting code already anticipate them, not because they're live features today:
- **Trend synthesis**: Layer 2 filtered by `year` column across decade buckets.
  `chunks.year NOT NULL` and the minimum-chunk-count-per-bucket caveat below are
  both already enforced in the data layer for when this gets built, but no
  decade-bucketing query or UI exists yet.
- **Book corpus**: `chunks.source_type` already accepts `'book'` (see schema),
  but no ingestion path produces book-sourced chunks yet -- only
  `'game_annotation'` rows exist today, from ChessBase exports.
- **Annotated PGN export**: attach Stockfish evals + synthesized commentary as PGN
  comments, export via `python-chess`. Label AI-generated commentary as such.

## Stack decisions (already made — don't re-ask)

- **DB**: Postgres + pgvector, hosted on a free tier (Neon or Supabase) — need a
  live shareable demo link, not just local.
- **LLM**: Anthropic Claude Sonnet 5 (`claude-sonnet-5`) as primary via native tool
  use / function calling. OpenAI is an acceptable fallback adapter, not primary.
- **Embeddings**: Voyage AI `voyage-4` (Anthropic's recommended embeddings
  partner; Anthropic has no first-party embedding model). 1024-dim vectors —
  `chunks.embedding` is `VECTOR(1024)`. 200M free tokens/model covers this
  corpus; batch requests (up to 128 texts/call) during full-corpus runs.
- **RAG orchestration**: raw SQL + API calls, NOT LangChain/LlamaIndex — deliberately,
  so every step is explainable in an interview without hiding behind a framework.
- **Frontend**: Streamlit.
- **Board input**: a draggable board (`src/ui/board_component/`), a custom
  `st.components.v2.component` wrapping chessboard.js, replacing the original
  click-grid v1. Deliberately not paired with chess.js — move legality stays
  entirely in python-chess (`_attempt_move` in `src/ui/chat.py`), the same
  source of truth the rest of the app already uses; chessboard.js is wired
  purely as a visual/drag layer, optimistic-UI style (see that function's
  docstring). Piece art is generated from `chess.svg.piece()` via
  `scripts/generate_board_piece_images.py`, not chessboard.js's stock
  Wikipedia set, for visual consistency with the rest of the app. Do NOT use
  the `streamlit-chess` / `streamlit-chess-board` PyPI packages — unmaintained,
  and the latter explicitly doesn't work when deployed to Streamlit Cloud.
- **Linting/formatting**: ruff (not black/flake8 separately).

## Data sources

- Master game annotations exported from ChessBase 17 (user owns the license).
  Exports may need encoding cleanup (sometimes Windows-1252, not UTF-8) and a
  NAG-to-symbol mapping if the numeric glyphs need to become !/?! in prose.
- Public domain chess literature (Project Gutenberg / archive.org) for the book
  corpus — keep in the public repo. Any modern copyrighted books the user owns
  personally are for local/private ingestion ONLY — never commit that text or
  distribute it; keep those paths gitignored.
- Lichess open database for raw move-level data (Layer 1 volume).

## Schema

See `scripts/init_db.sql` for the authoritative DDL. Key point: every `chunks` row
must have `year` populated — it's required for trend synthesis, not optional.

## Known reliability caveats (be honest about these in the demo/README, don't hide them)

- LLM-generated chess commentary is a synthesis of retrieved human text, NOT
  independent tactical judgment — do not let the agent generate "what should I
  play" style advice without Stockfish grounding.
- Personalized structural similarity (Layer 4) is approximate, not exact — UI
  copy should say "illustrative comparison."
- Trend synthesis should require a minimum chunk count per decade bucket before
  claiming a trend exists, to avoid overstating sparse data.

## Current status / what's next

All four layers, the recommendation/quality-classifier subsystem, the
draggable board, the guided tour, and CI are built and tested. What's left
before this is fully resume-ready: a security pass and a README (neither
written yet). The "designed but not yet implemented" list above (trend
synthesis, book corpus, annotated PGN export) is the honest list of what
to build next if picking this back up as a feature project rather than a
polish pass.

## Commands

- `streamlit run src/app.py` — run the app
- `psql "$DATABASE_URL" -f scripts/init_db.sql` — init schema
- `ruff check --fix . && ruff format .` — lint/format
- `pytest -v tests` — run tests. Two things self-skip with a clear reason rather
  than failing when their dependency isn't set up locally: the draggable-board
  test needs Playwright/Chromium (see below), and several DB-backed tests
  (Layer 1/4 structured-search and similarity, the schema constraint tests,
  the study-index build test) need a local Postgres with pgvector -- CI
  provisions both automatically (`.github/workflows/ci.yml`'s `postgres`
  service), so a passing CI run always exercises the real thing even when a
  local run without Postgres set up doesn't.
- `pip install -e ".[dev]" && playwright install chromium` — one-time setup for
  the real-browser board test (`tests/test_board_component.py`)
