"""Applies the real scripts/init_db.sql against a throwaway database and
checks the constraints it declares actually hold -- games.source and
chunks.source_type both have CHECK constraints that no other test ever
exercises: test_structured_search.py and test_similarity.py deliberately
build their own simplified games/chunks tables without them (documented in
test_structured_search.py's own fixture -- those tests only ever query
game_id/source_type, not the constraint itself), so a regression that let
an invalid value through db_loader.load_games/load_chunks would previously
have been caught by nothing short of a production `psql` insert failing.

This also incidentally verifies scripts/init_db.sql is valid, executable
SQL -- nothing else in the test suite ever runs the real schema file at
all.
"""

from pathlib import Path

import psycopg2
import pytest

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "scripts" / "init_db.sql"
TEST_DB = "chess_rag_schema_test"


def _postgres_available() -> bool:
    try:
        psycopg2.connect(dbname="postgres").close()
        return True
    except psycopg2.OperationalError:
        return False


def _pgvector_available() -> bool:
    try:
        conn = psycopg2.connect(dbname="postgres")
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.close()
        return True
    except psycopg2.Error:
        return False


pytestmark = pytest.mark.skipif(
    not (_postgres_available() and _pgvector_available()),
    reason="requires a local Postgres with the pgvector extension",
)


@pytest.fixture
def conn():
    admin = psycopg2.connect(dbname="postgres")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        cur.execute(f"CREATE DATABASE {TEST_DB}")
    admin.close()

    test_conn = psycopg2.connect(dbname=TEST_DB)
    with test_conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    test_conn.commit()

    yield test_conn

    test_conn.close()
    admin = psycopg2.connect(dbname="postgres")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
    admin.close()


def _insert_game(conn, game_id: str, source: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO games (game_id, source) VALUES (%s, %s)",
            (game_id, source),
        )
    conn.commit()


def test_games_source_check_accepts_the_two_documented_values(conn):
    _insert_game(conn, "g1", "chessbase")
    _insert_game(conn, "g2", "lichess")  # must not raise either


def test_games_source_check_rejects_an_invalid_value(conn):
    """Regression coverage: nothing in Python validates games.source
    before insert (pgn_parser.GameRecord.source is a bare `str`) -- the
    database CHECK constraint is the only thing enforcing this at all.
    """
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert_game(conn, "g1", "not_a_real_source")


def _insert_chunk(conn, chunk_hash: str, source_type: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chunks (chunk_hash, source_type, year, text) VALUES (%s, %s, %s, %s)",
            (chunk_hash, source_type, 2021, "some text"),
        )
    conn.commit()


def test_chunks_source_type_check_accepts_the_two_documented_values(conn):
    _insert_chunk(conn, "h1", "game_annotation")
    _insert_chunk(conn, "h2", "book")  # must not raise either


def test_chunks_source_type_check_rejects_an_invalid_value(conn):
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert_chunk(conn, "h1", "not_a_real_source_type")
