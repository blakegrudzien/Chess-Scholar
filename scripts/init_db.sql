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

-- Lichess studies eligible for recommendation, with embeddings for
-- similarity search. Filled by scripts/build_study_index.py. Title and
-- likes are copied in from the local candidates file so the deployed app
-- needs only this table.
--
-- No vector index: at tens to hundreds of rows an exact scan is simpler
-- and faster than IVFFlat.
CREATE TABLE IF NOT EXISTS lichess_study_cache (
    study_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    likes INTEGER NOT NULL,
    quality_probability REAL NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every question asked in the deployed app, kept for later review (see
-- src/ui/conversation_log.py). fen_context is null for questions not asked
-- about the board.
CREATE TABLE IF NOT EXISTS conversation_log (
    id SERIAL PRIMARY KEY,
    asked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    question TEXT NOT NULL,
    fen_context TEXT,
    answer TEXT NOT NULL
);

-- Saved answers for the app's example questions, so a demo click is instant
-- and free. Written by scripts/generate_example_answers.py. Kept here rather
-- than in the repo, since an answer can paraphrase licensed annotations.
CREATE TABLE IF NOT EXISTS example_answers (
    question TEXT PRIMARY KEY,
    answer TEXT NOT NULL,
    touched_fens JSONB NOT NULL,
    recommendations JSONB NOT NULL,
    -- When each status line appeared, when the answer began streaming and
    -- the total time, so a saved answer can be replayed at a live pace.
    timeline JSONB,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- For tables created before the timeline column existed.
ALTER TABLE example_answers ADD COLUMN IF NOT EXISTS timeline JSONB;

-- Indexes for Layer 1 structured search
CREATE INDEX IF NOT EXISTS idx_games_eco ON games(eco_code);
CREATE INDEX IF NOT EXISTS idx_games_year ON games(year);
CREATE INDEX IF NOT EXISTS idx_moves_piece_tosq ON moves(piece, to_sq);
CREATE INDEX IF NOT EXISTS idx_moves_capture ON moves(is_capture) WHERE is_capture = true;

-- Index for Layer 4 structural similarity (matches moves against a target
-- opening prefix by ply across all games, not scoped to one game_id)
CREATE INDEX IF NOT EXISTS idx_moves_ply_san ON moves(ply, move_san);

-- The Layer 2 vector index (IVFFlat) is not created here. Its clustering is
-- fixed when it is built, and built on the empty table it degenerates into
-- one big list that scans nearly everything (~100-300x slower, with no
-- error). After loading the corpus, and whenever its size changes by an
-- order of magnitude, build it with `lists` at about row count / 1000:
--
--   SET maintenance_work_mem = '256MB';  -- lists this size needs more than Neon's default 64MB
--   CREATE INDEX idx_chunks_embedding ON chunks
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 470);  -- ~468k rows
--   ANALYZE chunks;
CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
