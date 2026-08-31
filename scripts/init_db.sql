-- Chess RAG schema. Run via: psql "$DATABASE_URL" -f scripts/init_db.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    white TEXT,
    black TEXT,
    event TEXT,
    year INTEGER,
    eco_code TEXT,
    result TEXT,
    source TEXT CHECK (source IN ('chessbase', 'lichess'))
);

CREATE TABLE IF NOT EXISTS moves (
    game_id TEXT REFERENCES games(game_id) ON DELETE CASCADE,
    ply INTEGER,
    move_san TEXT,
    move_uci TEXT,
    from_sq TEXT,
    to_sq TEXT,
    piece TEXT,
    is_capture BOOLEAN,
    captured_piece TEXT,
    material_delta INTEGER,
    fen_after TEXT,
    PRIMARY KEY (game_id, ply)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id SERIAL PRIMARY KEY,
    chunk_hash TEXT NOT NULL UNIQUE,  -- content hash; natural key for idempotent loads
    source_type TEXT NOT NULL CHECK (source_type IN ('game_annotation', 'book')),
    game_id TEXT,                -- nullable, populated for game_annotation chunks
    source_title TEXT,
    author TEXT,
    year INTEGER NOT NULL,       -- required for trend synthesis, do not skip
    eco_code TEXT,
    ply_or_page TEXT,
    text TEXT NOT NULL,
    embedding VECTOR(1024)  -- voyage-4 / voyage-4-lite default dimension
);

-- Recommendation feature: quality-passing Lichess study candidates, ready
-- for embedding similarity search. Populated by scripts/build_study_index.py,
-- not by ingestion -- title/likes are duplicated from the (gitignored, local-
-- only) study_candidates.jsonl file so the deployed app is fully
-- self-sufficient from this table alone, the same reason models/*.joblib is
-- committed rather than regenerated at deploy time.
--
-- No IVFFlat index here, unlike chunks.embedding -- IVFFlat is an
-- approximate index that only pays for itself at real scale (thousands of
-- rows); at the row counts this table will actually hold (tens to low
-- hundreds), a plain `ORDER BY embedding <=> query LIMIT k` exact scan is
-- both simpler and faster. See chunks.embedding's own comment below for the
-- concrete failure mode this avoids repeating.
CREATE TABLE IF NOT EXISTS lichess_study_cache (
    study_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    likes INTEGER NOT NULL,
    quality_probability REAL NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for Layer 1 structured search
CREATE INDEX IF NOT EXISTS idx_games_eco ON games(eco_code);
CREATE INDEX IF NOT EXISTS idx_games_year ON games(year);
CREATE INDEX IF NOT EXISTS idx_moves_piece_tosq ON moves(piece, to_sq);
CREATE INDEX IF NOT EXISTS idx_moves_capture ON moves(is_capture) WHERE is_capture = true;

-- Index for Layer 4 structural similarity (matches moves against a target
-- opening prefix by ply across all games, not scoped to one game_id)
CREATE INDEX IF NOT EXISTS idx_moves_ply_san ON moves(ply, move_san);

-- Index for Layer 2 vector search (IVFFlat). This script runs before the
-- corpus is loaded, so `chunks` is empty here and `lists` can't be sized
-- correctly yet -- IVFFlat's clustering quality depends on the row count at
-- BUILD time, not query time. Building it against an empty table produces a
-- degenerate index (effectively one giant list) that silently falls back to
-- scanning nearly the whole table on every query, no error, just ~100-300x
-- slower lookups. This index is intentionally left out of the initial
-- schema; after loading the corpus (and any time its size changes by an
-- order of magnitude), build or rebuild it explicitly with `lists` set to
-- roughly (row count / 1000) for up to ~1M rows:
--
--   SET maintenance_work_mem = '256MB';  -- lists this size needs more than Neon's default 64MB
--   CREATE INDEX idx_chunks_embedding ON chunks
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 470);  -- ~468k rows
--   ANALYZE chunks;
CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
