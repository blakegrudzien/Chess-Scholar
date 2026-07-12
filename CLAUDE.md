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
  prose (plans), synthesized together. This is the flagship demo query.
- **Trend synthesis**: Layer 2 filtered by `year` column across decade buckets.
- **Annotated PGN export**: attach Stockfish evals + synthesized commentary as PGN
  comments, export via `python-chess`. Label AI-generated commentary as such.

## Stack decisions (already made — don't re-ask)

- **DB**: Postgres + pgvector, hosted on a free tier (Neon or Supabase) — need a
  live shareable demo link, not just local.
- **LLM**: Anthropic Claude Sonnet 5 (`claude-sonnet-5`) as primary via native tool
  use / function calling. OpenAI is an acceptable fallback adapter, not primary.
- **Embeddings**: OpenAI `text-embedding-3-small` (Anthropic has no first-party
  embedding model). Use Batch API for full-corpus runs, real-time during dev.
- **RAG orchestration**: raw SQL + API calls, NOT LangChain/LlamaIndex — deliberately,
  so every step is explainable in an interview without hiding behind a framework.
- **Frontend**: Streamlit.
- **Board input v1**: click-based board using `chess.svg` rendering + Streamlit
  session state (no JS). A drag-and-drop JS component (chessboard.js + chess.js via
  Streamlit's Components API) is a stretch goal for later — do NOT use the existing
  `streamlit-chess` / `streamlit-chess-board` PyPI packages, they're unmaintained
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

Scaffolding + `.vscode` config done. Next: `src/ingestion/pgn_parser.py`,
`annotation_extractor.py`, `db_loader.py` — this is the foundation, build and test
this before touching Layer 2/3/4.

## Commands

- `streamlit run src/app.py` — run the app
- `psql "$DATABASE_URL" -f scripts/init_db.sql` — init schema
- `ruff check --fix . && ruff format .` — lint/format
- `pytest -v tests` — run tests
