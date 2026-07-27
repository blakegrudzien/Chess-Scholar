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
    source TEXT  -- 'chessbase' | 'lichess'
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
    source_type TEXT NOT NULL,   -- 'game_annotation' | 'book'
    game_id TEXT,                -- nullable, populated for game_annotation chunks
    source_title TEXT,
    author TEXT,
    year INTEGER NOT NULL,       -- required for trend synthesis, do not skip
    eco_code TEXT,
    ply_or_page TEXT,
    text TEXT NOT NULL,
    embedding VECTOR(1024)  -- voyage-4 / voyage-4-lite default dimension
);

CREATE TABLE IF NOT EXISTS opening_notes (
    note_id SERIAL PRIMARY KEY,
    eco_code TEXT,
    opening_name TEXT,
    note_text TEXT,
    updated_at TIMESTAMP DEFAULT now()
);

-- Indexes for Layer 1 structured search
CREATE INDEX IF NOT EXISTS idx_games_eco ON games(eco_code);
CREATE INDEX IF NOT EXISTS idx_games_year ON games(year);
CREATE INDEX IF NOT EXISTS idx_moves_piece_tosq ON moves(piece, to_sq);
CREATE INDEX IF NOT EXISTS idx_moves_capture ON moves(is_capture) WHERE is_capture = true;

-- Index for Layer 2 vector search (IVFFlat; fine for a few thousand-tens of thousands of chunks)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON chunks(source_type);
