"""Load parsed PGN records (pgn_parser.py, annotation_extractor.py) into the
games/moves/chunks tables defined in scripts/init_db.sql.

Inserts are idempotent:
- game_id is a deterministic content hash (see pgn_parser.compute_game_id),
  so re-running ingestion on the same PGN file hits ON CONFLICT DO NOTHING
  instead of creating duplicate games/moves rows.
- chunks has no natural key otherwise (chunk_id is a bare SERIAL), so chunk
  rows are keyed on chunk_hash, a content hash computed here.

Inserts are committed in batches (GAMES_BATCH_SIZE games, CHUNKS_BATCH_SIZE
chunks), since one ChessBase export can hold hundreds of thousands of rows.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from urllib.parse import urlparse

import psycopg2
import psycopg2.extensions
import psycopg2.pool
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from src.ingestion.annotation_extractor import AnnotationChunk, extract_annotations
from src.ingestion.hash_utils import ID_DELIMITER, check_no_delimiter
from src.ingestion.pgn_parser import GameRecord, parse_pgn

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
    # override=True: locally, .env must win over values Streamlit loads from
    # .streamlit/secrets.toml (see get_engine_path in stockfish_eval.py).
    load_dotenv(override=True)
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url, **_neon_connect_kwargs(database_url))


# Enough for a handful of concurrent sessions; queries are fast, so the pool
# exists to keep sessions from sharing one connection.
DB_POOL_MIN_CONN = 2
DB_POOL_MAX_CONN = 10


# Neon's free tier suspends compute after 5 minutes without queries, which
# closes every open connection server-side while the pool still holds them.
# A connection idle for less than that can't have been closed by a suspend,
# so only connections idle at least this long are tested before use.
STALE_CONNECTION_AGE_SECONDS = 270


def _select_one(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
    # psycopg2 opens a transaction on any statement; close it so the
    # connection doesn't sit "idle in transaction".
    conn.rollback()


class HealthCheckedConnectionPool(psycopg2.pool.ThreadedConnectionPool):
    """A ThreadedConnectionPool that tests a connection with SELECT 1 before
    handing it out, if it has sat idle for STALE_CONNECTION_AGE_SECONDS or
    longer. Recently used connections skip the test, so an active session
    pays nothing for it.

    A failed test closes every idle connection, not just the one tested: a
    suspend kills them all at once, so the next one in the pool would fail
    the same way. The caller then gets a freshly opened connection.
    """

    # psycopg2 pool internals this class builds on. They exist at runtime
    # (psycopg2/pool.py) but psycopg2's type stubs don't declare them.
    _lock: threading.Lock
    _pool: list[psycopg2.extensions.connection]
    _getconn: Callable[..., psycopg2.extensions.connection]
    _putconn: Callable[..., None]

    def __init__(self, minconn: int, maxconn: int, *args, **kwargs):
        super().__init__(minconn, maxconn, *args, **kwargs)
        # id(conn) -> time.monotonic() when it last entered the idle pool,
        # starting with the minconn connections opened above.
        now = time.monotonic()
        self._idle_since: dict[int, float] = {id(conn): now for conn in self._pool}

    def getconn(self, key=None):
        with self._lock:
            conn = self._getconn(key)
            idle_since = self._idle_since.pop(id(conn), None)
        if idle_since is None or time.monotonic() - idle_since < STALE_CONNECTION_AGE_SECONDS:
            return conn
        try:
            _select_one(conn)
            return conn
        except DB_RETRYABLE_ERRORS as exc:
            logger.warning(
                "Pooled connection idle %.0fs failed its health check (%s); "
                "replacing all idle connections",
                time.monotonic() - idle_since,
                str(exc).strip(),
            )
        with self._lock:
            for idle_conn in self._pool:
                idle_conn.close()
            self._pool.clear()
            self._idle_since.clear()
            self._putconn(conn, key, close=True)
            return self._getconn(key)

    def putconn(self, conn=None, key=None, close=False):
        with self._lock:
            self._putconn(conn, key, close)
            # _putconn closes a connection instead of pooling it when the
            # pool already holds minconn idle ones, so check where it went.
            if any(idle_conn is conn for idle_conn in self._pool):
                self._idle_since[id(conn)] = time.monotonic()
            else:
                self._idle_since.pop(id(conn), None)


def get_connection_pool() -> HealthCheckedConnectionPool:
    load_dotenv(override=True)  # see get_connection above for why override=True
    database_url = os.environ["DATABASE_URL"]
    return HealthCheckedConnectionPool(
        DB_POOL_MIN_CONN, DB_POOL_MAX_CONN, database_url, **_neon_connect_kwargs(database_url)
    )


def ping_database(db_pool: psycopg2.pool.ThreadedConnectionPool) -> None:
    """Run SELECT 1 through the pool, which resets Neon's inactivity timer
    and wakes a suspended compute."""
    query_with_retry(db_pool, _select_one)


class DatabaseBusyError(Exception):
    """Raised by get_connection_with_timeout() when no pooled connection
    becomes available within the wait budget.
    """


# psycopg2's getconn() fails at once when the pool is empty. Connections free
# up quickly (unlike engines, which are busy for a whole search), so callers
# wait briefly instead.
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


DB_RETRYABLE_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)

# The first attempt plus one retry on a dropped connection.
MAX_QUERY_ATTEMPTS = 2


def query_with_retry(db_pool: psycopg2.pool.ThreadedConnectionPool, fn: Callable, *args, **kwargs):
    """Run fn(conn, *args, **kwargs) with a pooled connection, retrying once
    with another connection if the first has dropped. A connection that
    failed is closed rather than returned to the pool. Used by every
    database-backed agent tool.
    """
    conn = get_connection_with_timeout(db_pool)
    last_error: Exception | None = None
    for attempt in range(MAX_QUERY_ATTEMPTS):
        try:
            result = fn(conn, *args, **kwargs)
        except DB_RETRYABLE_ERRORS as exc:
            db_pool.putconn(conn, close=True)
            if attempt == MAX_QUERY_ATTEMPTS - 1:
                raise
            last_error = exc
            conn = get_connection_with_timeout(db_pool)
            continue
        except Exception:
            # Not a connection problem: the connection is fine to reuse.
            db_pool.putconn(conn)
            raise
        else:
            db_pool.putconn(conn)
            # The only record that a pooled connection had died; the error
            # text tells a stale connection from an unreachable database.
            if attempt > 0:
                logger.warning(
                    "%s succeeded on attempt %d after a dropped connection: %s",
                    getattr(fn, "__name__", repr(fn)),
                    attempt + 1,
                    str(last_error).strip(),
                )
            return result
    # Unreachable: every attempt returns or raises. Stated so a type checker
    # doesn't see an implicit None return.
    raise AssertionError("unreachable: every query_with_retry attempt returns or raises")


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
                # One page, or cur.rowcount only counts execute_values' last
                # internal page of 100.
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
    """Insert annotation chunks in batches of CHUNKS_BATCH_SIZE. Returns the
    number of rows inserted; duplicates and undated chunks don't count.

    chunks.year is NOT NULL (trend synthesis needs it), but ChessBase marks
    unknown dates "????.??.??", which parses to None. Undated chunks are
    skipped and counted: one would otherwise fail the whole unflushed batch.
    """
    chunks_inserted = 0
    chunks_skipped_missing_year = 0
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
        if chunk.year is None:
            chunks_skipped_missing_year += 1
            continue
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
    if chunks_skipped_missing_year:
        logger.warning(
            "Skipped %d chunk(s) with no determinable year (chunks.year is NOT NULL)",
            chunks_skipped_missing_year,
        )
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
    # Configured here, not at import, so importing the module never changes
    # the caller's logging.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _main()
