import psycopg2
import pytest

from src.ingestion.db_loader import load_games
from src.ingestion.pgn_parser import parse_pgn
from src.personalization.similarity import find_similar_games

TEST_DB = "chess_rag_similarity_test"

# A and B share the first 4 plies (e4 e5 Nf3 Nc6), diverging at ply 5.
# C shares nothing (different first move).
GAME_A = (
    '[Event "A"]\n[Date "2020.01.01"]\n[White "Alice"]\n[Black "Bob"]\n'
    '[Result "1-0"]\n[ECO "C50"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 1-0\n'
)
GAME_B = (
    '[Event "B"]\n[Date "2021.01.01"]\n[White "Carol"]\n[Black "Dave"]\n'
    '[Result "0-1"]\n[ECO "C60"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 0-1\n'
)
GAME_C = (
    '[Event "C"]\n[Date "2019.01.01"]\n[White "Eve"]\n[Black "Frank"]\n'
    '[Result "1/2-1/2"]\n[ECO "D10"]\n\n1. d4 d5 2. c4 c6 1/2-1/2\n'
)

USER_MOVES = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]  # matches A exactly


def _postgres_available() -> bool:
    try:
        psycopg2.connect(dbname="postgres").close()
        return True
    except psycopg2.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _postgres_available(), reason="requires a local Postgres")


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    admin = psycopg2.connect(dbname="postgres")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        cur.execute(f"CREATE DATABASE {TEST_DB}")
    admin.close()

    test_conn = psycopg2.connect(dbname=TEST_DB)
    with test_conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY, white TEXT, black TEXT, event TEXT,
                year INTEGER, eco_code TEXT, result TEXT, source TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE moves (
                game_id TEXT REFERENCES games(game_id) ON DELETE CASCADE,
                ply INTEGER, move_san TEXT, move_uci TEXT, from_sq TEXT, to_sq TEXT,
                piece TEXT, is_capture BOOLEAN, captured_piece TEXT,
                material_delta INTEGER, fen_after TEXT,
                PRIMARY KEY (game_id, ply)
            )
        """)
    test_conn.commit()

    tmp_dir = tmp_path_factory.mktemp("pgn")
    for name, text in [("a.pgn", GAME_A), ("b.pgn", GAME_B), ("c.pgn", GAME_C)]:
        path = tmp_dir / name
        path.write_text(text, encoding="utf-8")
        load_games(parse_pgn(path, source="test"), test_conn)

    yield test_conn

    test_conn.close()
    admin = psycopg2.connect(dbname="postgres")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
    admin.close()


def test_ranks_by_longest_matching_opening_prefix(conn):
    results = find_similar_games(conn, USER_MOVES, max_ply=20, limit=10)
    by_event = {r.white: r.matching_plies for r in results}

    assert by_event["Alice"] == 6  # exact full match
    assert by_event["Carol"] == 4  # diverges at ply 5 (Bb5 vs Bc4)
    assert by_event["Eve"] == 0  # diverges at ply 1 (d4 vs e4)


def test_results_ordered_best_match_first(conn):
    results = find_similar_games(conn, USER_MOVES, max_ply=20, limit=10)
    assert [r.matching_plies for r in results] == sorted(
        (r.matching_plies for r in results), reverse=True
    )
    assert results[0].white == "Alice"


def test_respects_limit(conn):
    results = find_similar_games(conn, USER_MOVES, max_ply=20, limit=1)
    assert len(results) == 1
    assert results[0].white == "Alice"


def test_max_ply_caps_how_far_the_comparison_looks(conn):
    # Only compare the first 3 plies -- A and B both match fully within that window.
    results = find_similar_games(conn, USER_MOVES, max_ply=3, limit=10)
    by_white = {r.white: r.matching_plies for r in results}
    assert by_white["Alice"] == 3
    assert by_white["Carol"] == 3


def test_rejects_empty_moves(conn):
    with pytest.raises(ValueError):
        find_similar_games(conn, [], max_ply=20, limit=10)
