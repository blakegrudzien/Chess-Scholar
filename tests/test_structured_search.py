import psycopg2
import pytest

from src.ingestion.db_loader import load_games
from src.ingestion.pgn_parser import parse_pgn
from src.search.structured_search import (
    MoveFrequency,
    SquareFrequency,
    common_moves_at_ply,
    eco_summary,
    game_moves_as_pgn,
    piece_placement_frequency,
    select_narrative_game,
)

TEST_DB = "chess_rag_structured_search_test"

# Two C50 games sharing the same first 3 moves (so Nc6/f3 land twice, at ply
# 3/4), diverging after -- gives deterministic tie-breaking to assert on.
GAME_A = (
    '[Event "A"]\n[Date "2020.01.01"]\n[White "Alice"]\n[Black "Bob"]\n'
    '[Result "1-0"]\n[ECO "C50"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 1-0\n'
)
GAME_B = (
    '[Event "B"]\n[Date "2021.01.01"]\n[White "Carol"]\n[Black "Dave"]\n'
    '[Result "0-1"]\n[ECO "C50"]\n\n1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 0-1\n'
)
GAME_C = (
    '[Event "C"]\n[Date "2019.01.01"]\n[White "Eve"]\n[Black "Frank"]\n'
    '[Result "1/2-1/2"]\n[ECO "D10"]\n\n1. d4 d5 2. c4 c6 3. Nf3 Nf6 1/2-1/2\n'
)
# Both chessbase-sourced (A/B/C above are source="test"), both C50, so the
# narrative-game tests can assert selection prefers the more-annotated one.
GAME_D = (
    '[Event "D"]\n[Date "2018.01.01"]\n[White "Grace"]\n[Black "Heidi"]\n'
    '[Result "1-0"]\n[ECO "C60"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 1-0\n'
)
GAME_E = (
    '[Event "E"]\n[Date "2017.01.01"]\n[White "Ivan"]\n[Black "Judy"]\n'
    '[Result "0-1"]\n[ECO "C60"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 0-1\n'
)


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
        # Simplified vs. the real chunks table (no embedding column, no
        # source_type CHECK constraint) -- these tests only ever query
        # game_id and source_type, and skipping embedding avoids requiring
        # the pgvector extension in a test database that doesn't need it.
        cur.execute("""
            CREATE TABLE chunks (
                chunk_id SERIAL PRIMARY KEY,
                chunk_hash TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL,
                game_id TEXT,
                year INTEGER NOT NULL,
                text TEXT NOT NULL
            )
        """)
    test_conn.commit()

    tmp_dir = tmp_path_factory.mktemp("pgn")
    for name, text in [("a.pgn", GAME_A), ("b.pgn", GAME_B), ("c.pgn", GAME_C)]:
        path = tmp_dir / name
        path.write_text(text, encoding="utf-8")
        load_games(parse_pgn(path, source="test"), test_conn)

    # D and E are the only source="chessbase" games -- select_narrative_game
    # filters to that source, so A/B/C (source="test") must never match.
    for name, text in [("d.pgn", GAME_D), ("e.pgn", GAME_E)]:
        path = tmp_dir / name
        path.write_text(text, encoding="utf-8")
        load_games(parse_pgn(path, source="chessbase"), test_conn)

    with test_conn.cursor() as cur:
        cur.execute("SELECT event, game_id FROM games WHERE source = 'chessbase'")
        game_id_by_event = dict(cur.fetchall())
        # Two annotation chunks on D, zero on E: D should win the C50 match.
        cur.executemany(
            "INSERT INTO chunks (chunk_hash, source_type, game_id, year, text) "
            "VALUES (%s, 'game_annotation', %s, 2018, %s)",
            [
                ("chunk-d-1", game_id_by_event["D"], "Note 1"),
                ("chunk-d-2", game_id_by_event["D"], "Note 2"),
            ],
        )
    test_conn.commit()

    yield test_conn

    test_conn.close()
    admin = psycopg2.connect(dbname="postgres")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
    admin.close()


def test_eco_summary_counts_games_and_results(conn):
    summary = eco_summary(conn, "C50")
    assert summary.game_count == 2
    assert summary.white_wins == 1
    assert summary.black_wins == 1
    assert summary.draws == 0
    assert summary.avg_ply_count == 6.0


def test_eco_summary_different_eco_code_is_independent(conn):
    summary = eco_summary(conn, "D10")
    assert summary.game_count == 1
    assert summary.draws == 1
    assert summary.avg_ply_count == 6.0


def test_eco_summary_unknown_code_returns_zero_and_none(conn):
    summary = eco_summary(conn, "Z99")
    assert summary.game_count == 0
    assert summary.avg_ply_count is None


def test_piece_placement_frequency_both_colors_orders_by_count_then_square(conn):
    result = piece_placement_frequency(conn, "C50", "N", max_ply=None)
    assert result == [
        SquareFrequency("c6", 2),
        SquareFrequency("f3", 2),
        SquareFrequency("c3", 1),
        SquareFrequency("f6", 1),
    ]


def test_piece_placement_frequency_filters_by_color(conn):
    white_only = piece_placement_frequency(conn, "C50", "N", color="white", max_ply=None)
    assert white_only == [SquareFrequency("f3", 2), SquareFrequency("c3", 1)]

    black_only = piece_placement_frequency(conn, "C50", "N", color="black", max_ply=None)
    assert black_only == [SquareFrequency("c6", 2), SquareFrequency("f6", 1)]


def test_piece_placement_frequency_respects_max_ply(conn):
    # Only ply <= 3 counted: both games' 2.Nf3 (ply 3), nothing later.
    result = piece_placement_frequency(conn, "C50", "N", max_ply=3)
    assert result == [SquareFrequency("f3", 2)]


def test_piece_placement_frequency_rejects_invalid_piece(conn):
    with pytest.raises(ValueError):
        piece_placement_frequency(conn, "C50", "X")


def test_piece_placement_frequency_rejects_invalid_color(conn):
    with pytest.raises(ValueError):
        piece_placement_frequency(conn, "C50", "N", color="purple")  # type: ignore[arg-type]


def test_common_moves_at_ply_merges_identical_moves(conn):
    # Both games play 2...Nc6 at ply 4.
    result = common_moves_at_ply(conn, "C50", 4)
    assert result == [MoveFrequency("Nc6", 2)]


def test_common_moves_at_ply_breaks_ties_alphabetically(conn):
    # Ply 6 diverges: game A plays Bc5, game B plays Nf6 -- tied at count 1.
    result = common_moves_at_ply(conn, "C50", 6)
    assert result == [MoveFrequency("Bc5", 1), MoveFrequency("Nf6", 1)]


def test_select_narrative_game_prefers_more_annotated_game(conn):
    # D and E are both chessbase-sourced C60 games; only D has chunks.
    candidate = select_narrative_game(conn, ["C60"])
    assert candidate is not None
    assert candidate.event == "D"
    assert candidate.annotation_chunk_count == 2


def test_select_narrative_game_excludes_non_chessbase_source(conn):
    # A and B (source="test") are C50, not C60, so they can't leak into a
    # C60 query -- this specifically checks the *other* filter, source:
    # querying A/B's own C50 code must find nothing, since no chessbase
    # game exists at C50 (only D/E at C60 are chessbase-sourced).
    assert select_narrative_game(conn, ["C50"]) is None


def test_select_narrative_game_no_chessbase_match_returns_none(conn):
    # C is D10 but source="test"; no chessbase game exists for D10.
    assert select_narrative_game(conn, ["D10"]) is None


def test_select_narrative_game_unknown_eco_returns_none(conn):
    assert select_narrative_game(conn, ["Z99"]) is None


def test_select_narrative_game_rejects_empty_eco_list(conn):
    with pytest.raises(ValueError):
        select_narrative_game(conn, [])


def test_game_moves_as_pgn_contains_headers_and_moves_no_comments(conn):
    candidate = select_narrative_game(conn, ["C60"])
    assert candidate is not None
    pgn_text = game_moves_as_pgn(conn, candidate.game_id)
    assert pgn_text is not None

    assert '[White "Grace"]' in pgn_text
    assert '[Black "Heidi"]' in pgn_text
    assert '[ECO "C60"]' in pgn_text
    assert "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6" in pgn_text
    assert "1-0" in pgn_text
    assert "{" not in pgn_text  # no PGN comments anywhere


def test_game_moves_as_pgn_unknown_game_id_returns_none(conn):
    assert game_moves_as_pgn(conn, "does-not-exist") is None
