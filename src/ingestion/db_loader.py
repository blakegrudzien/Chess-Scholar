"""Load parsed PGN records (pgn_parser.py, annotation_extractor.py) into the
games/moves/chunks tables defined in scripts/init_db.sql.

Inserts are idempotent:
- game_id is a deterministic content hash (see pgn_parser.compute_game_id),
  so re-running ingestion on the same PGN file hits ON CONFLICT DO NOTHING
  instead of creating duplicate games/moves rows.
- chunks has no natural key otherwise (chunk_id is a bare SERIAL), so chunk
  rows are keyed on chunk_hash, a content hash computed here.

Inserts are batched (flushed + committed every GAMES_BATCH_SIZE games /
CHUNKS_BATCH_SIZE chunks) rather than materializing an entire large PGN file's
rows in memory before one insert -- a single ChessBase export can be
hundreds of thousands of games/chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from collections.abc import Iterable
from urllib.parse import urlparse

import psycopg2
import psycopg2.extensions
import psycopg2.pool
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from src.ingestion.annotation_extractor import AnnotationChunk, extract_annotations
from src.ingestion.hash_utils import ID_DELIMITER, check_no_delimiter
from src.ingestion.pgn_parser import GameRecord, parse_pgn

# getLogger(__name__), not the root logger: log records carry this module's
# dotted path, so anything consuming these logs can tell where a message
# came from, and can configure this module's verbosity independently of
# every other module's.
logger = logging.getLogger(__name__)

GAMES_INSERT = """
    INSERT INTO games (game_id, white, black, event, year, eco_code, result, source)
    VALUES %s
    ON CONFLICT (game_id) DO NOTHING
"""

MOVES_INSERT = """
    INSERT INTO moves (
        game_id, ply, move_san, move_uci, from_sq, to_sq,
        piece, is_capture, captured_piece, material_delta, fen_after
    )
    VALUES %s
    ON CONFLICT (game_id, ply) DO NOTHING
"""

CHUNKS_INSERT = """
    INSERT INTO chunks (
        chunk_hash, source_type, game_id, source_title, author, year, eco_code, ply_or_page, text
    )
    VALUES %s
    ON CONFLICT (chunk_hash) DO NOTHING
"""

GAMES_BATCH_SIZE = 500  # games per flush (also bounds moves held in memory at once)
CHUNKS_BATCH_SIZE = 5000  # chunks per flush


def _neon_connect_kwargs(database_url: str) -> dict[str, str]:
    hostname = urlparse(database_url).hostname or ""
    if not hostname.endswith(".neon.tech"):
        return {}
    # psycopg2-binary's bundled libpq predates Neon's SNI-based routing, so
    # the endpoint ID must be passed explicitly or connections fail with
    # "Endpoint ID is not specified". See https://neon.tech/sni
    endpoint = hostname.split(".", 1)[0]
    return {"options": f"endpoint={endpoint}"}


def get_connection() -> psycopg2.extensions.connection:
    # override=True: .env must win over anything already in os.environ, since
    # Streamlit auto-loads .streamlit/secrets.toml into the environment before
    # this runs (see get_engine_path in stockfish_eval.py for the full story).
    # No effect in deployment, where .env doesn't exist -- nothing to override.
    load_dotenv(override=True)
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url, **_neon_connect_kwargs(database_url))


# Sized for a handful of concurrent Streamlit sessions. DB round-trips here
# are fast (indexed lookups), so this pool exists to stop concurrent sessions
# serializing on one shared connection -- not because queries are slow.
DB_POOL_MIN_CONN = 2
DB_POOL_MAX_CONN = 10


def get_connection_pool() -> psycopg2.pool.ThreadedConnectionPool:
    load_dotenv(override=True)  # see get_connection above for why override=True
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.pool.ThreadedConnectionPool(
        DB_POOL_MIN_CONN, DB_POOL_MAX_CONN, database_url, **_neon_connect_kwargs(database_url)
    )


class DatabaseBusyError(Exception):
    """Raised by get_connection_with_timeout() when no pooled connection
    becomes available within the wait budget.
    """


# psycopg2's pool has no built-in wait/timeout option: getconn() either
# returns a connection immediately or raises PoolError right away. A DB
# checkout is fast and cheap (unlike a Stockfish search, which pins a CPU
# core for the whole request), so a caller arriving a moment after the pool
# was momentarily exhausted is likely to succeed if it just waits briefly --
# worth doing here in a way it wouldn't be worth doing for the engine pool.
DB_POOL_CHECKOUT_TIMEOUT_SECONDS = 2.0
DB_POOL_CHECKOUT_POLL_INTERVAL_SECONDS = 0.05


def get_connection_with_timeout(
    db_pool: psycopg2.pool.ThreadedConnectionPool,
) -> psycopg2.extensions.connection:
    """Check a connection out of db_pool, retrying briefly if it's
    momentarily exhausted instead of failing on the first attempt.
    """
    deadline = time.monotonic() + DB_POOL_CHECKOUT_TIMEOUT_SECONDS
    while True:
        try:
            return db_pool.getconn()
        except psycopg2.pool.PoolError:
            if time.monotonic() >= deadline:
                raise DatabaseBusyError(
                    "The database is handling too many requests right now. "
                    "Please try again in a moment."
                ) from None
            time.sleep(DB_POOL_CHECKOUT_POLL_INTERVAL_SECONDS)


def load_games(
    games: Iterable[GameRecord], conn: psycopg2.extensions.connection
) -> tuple[int, int]:
    """Insert games and their moves, flushing every GAMES_BATCH_SIZE games.
    Returns (games_inserted, moves_inserted), counting only rows that weren't
    already present.
    """
    games_inserted = 0
    moves_inserted = 0
    game_rows: list[tuple] = []
    move_rows: list[tuple] = []

    def flush() -> None:
        nonlocal games_inserted, moves_inserted, game_rows, move_rows
        with conn.cursor() as cur:
            if game_rows:
                # page_size=len(rows): without this, execute_values silently
                # sub-pages in groups of 100, and cur.rowcount afterward only
                # reflects the *last* internal page, not the true total.
                execute_values(cur, GAMES_INSERT, game_rows, page_size=len(game_rows))
                games_inserted += cur.rowcount
            if move_rows:
                execute_values(cur, MOVES_INSERT, move_rows, page_size=len(move_rows))
                moves_inserted += cur.rowcount
        conn.commit()
        game_rows = []
        move_rows = []

    for game in games:
        game_rows.append(
            (
                game.game_id,
                game.white,
                game.black,
                game.event,
                game.year,
                game.eco_code,
                game.result,
                game.source,
            )
        )
        move_rows.extend(
            (
                game.game_id,
                move.ply,
                move.move_san,
                move.move_uci,
                move.from_sq,
                move.to_sq,
                move.piece,
                move.is_capture,
                move.captured_piece,
                move.material_delta,
                move.fen_after,
            )
            for move in game.moves
        )
        if len(game_rows) >= GAMES_BATCH_SIZE:
            flush()

    flush()  # remaining partial batch
    return games_inserted, moves_inserted


def _chunk_hash(chunk: AnnotationChunk) -> str:
    fields = [
        chunk.source_type,
        chunk.game_id or "",
        chunk.source_title or "",
        chunk.author or "",
        chunk.ply_or_page or "",
        chunk.text,
    ]
    check_no_delimiter(*fields)
    canonical = ID_DELIMITER.join(fields)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_chunks(chunks: Iterable[AnnotationChunk], conn: psycopg2.extensions.connection) -> int:
    """Insert annotation/book chunks, flushing every CHUNKS_BATCH_SIZE chunks.
    Returns the number of rows actually inserted (rows skipped by ON CONFLICT
    don't count).
    """
    chunks_inserted = 0
    rows: list[tuple] = []

    def flush() -> None:
        nonlocal chunks_inserted, rows
        if rows:
            with conn.cursor() as cur:
                execute_values(cur, CHUNKS_INSERT, rows, page_size=len(rows))
                chunks_inserted += cur.rowcount
        conn.commit()
        rows = []

    for chunk in chunks:
        rows.append(
            (
                _chunk_hash(chunk),
                chunk.source_type,
                chunk.game_id,
                chunk.source_title,
                chunk.author,
                chunk.year,
                chunk.eco_code,
                chunk.ply_or_page,
                chunk.text,
            )
        )
        if len(rows) >= CHUNKS_BATCH_SIZE:
            flush()

    flush()  # remaining partial batch
    return chunks_inserted


def _main() -> None:
    parser = argparse.ArgumentParser(description="Load PGN-derived records into Postgres.")
    parser.add_argument("--input", required=True, help="Path to a PGN file")
    parser.add_argument(
        "--mode",
        choices=["games", "chunks"],
        default="games",
        help="'games' loads games+moves via pgn_parser; "
        "'chunks' loads annotation chunks via annotation_extractor",
    )
    parser.add_argument("--source", default="lichess", help="games mode: games.source column")
    parser.add_argument("--source-title", default=None, help="chunks mode: chunks.source_title")
    parser.add_argument("--author", default=None, help="chunks mode: chunks.author")
    args = parser.parse_args()

    conn = get_connection()
    try:
        if args.mode == "games":
            games = parse_pgn(args.input, source=args.source)
            games_inserted, moves_inserted = load_games(games, conn)
            logger.info(
                "Inserted %d new game(s), %d new move(s) from %s",
                games_inserted,
                moves_inserted,
                args.input,
            )
        else:
            chunks = extract_annotations(
                args.input, source_title=args.source_title, author=args.author
            )
            chunks_inserted = load_chunks(chunks, conn)
            logger.info("Inserted %d new chunk(s) from %s", chunks_inserted, args.input)
    finally:
        conn.close()


if __name__ == "__main__":
    # basicConfig belongs here, not at module level: it's a global side
    # effect (it configures the root logger's handlers), so only the
    # script entry point should trigger it. A module that configures
    # logging as a side effect of being imported would surprise (and
    # override the choices of) whatever imports it.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _main()
