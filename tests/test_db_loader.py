from unittest.mock import MagicMock, patch

from src.ingestion.db_loader import get_connection, load_games
from src.ingestion.pgn_parser import GameRecord, MoveRecord


def _sample_game() -> GameRecord:
    move = MoveRecord(
        ply=1,
        move_san="e4",
        move_uci="e2e4",
        from_sq="e2",
        to_sq="e4",
        piece="P",
        is_capture=False,
        captured_piece=None,
        material_delta=0,
        fen_after="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    )
    return GameRecord(
        game_id="abc123",
        white="Alice",
        black="Bob",
        event="Test Event",
        year=2021,
        eco_code="C00",
        result="1-0",
        source="lichess",
        moves=[move],
    )


def test_get_connection_reads_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/db")
    with patch("src.ingestion.db_loader.psycopg2.connect") as mock_connect:
        get_connection()
        mock_connect.assert_called_once_with("postgresql://u:p@host:5432/db")


def test_load_games_inserts_games_and_moves():
    game = _sample_game()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.rowcount = 1

    with patch("src.ingestion.db_loader.execute_values") as mock_execute_values:
        games_inserted, moves_inserted = load_games([game], conn)

    assert games_inserted == 1
    assert moves_inserted == 1
    assert mock_execute_values.call_count == 2

    games_call, moves_call = mock_execute_values.call_args_list
    games_sql, games_rows = games_call.args[1], games_call.args[2]
    assert "INSERT INTO games" in games_sql
    assert "ON CONFLICT (game_id) DO NOTHING" in games_sql
    assert games_rows == [("abc123", "Alice", "Bob", "Test Event", 2021, "C00", "1-0", "lichess")]

    moves_sql, moves_rows = moves_call.args[1], moves_call.args[2]
    assert "INSERT INTO moves" in moves_sql
    assert "ON CONFLICT (game_id, ply) DO NOTHING" in moves_sql
    assert moves_rows == [
        ("abc123", 1, "e4", "e2e4", "e2", "e4", "P", False, None, 0, game.moves[0].fen_after)
    ]

    conn.commit.assert_called_once()


def test_load_games_skips_moves_insert_when_no_games():
    conn = MagicMock()
    with patch("src.ingestion.db_loader.execute_values") as mock_execute_values:
        games_inserted, moves_inserted = load_games([], conn)

    assert (games_inserted, moves_inserted) == (0, 0)
    mock_execute_values.assert_not_called()
    conn.commit.assert_called_once()


def test_load_games_reports_zero_when_conflict_skips_row():
    game = _sample_game()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.rowcount = 0  # simulates ON CONFLICT DO NOTHING skipping an existing game

    with patch("src.ingestion.db_loader.execute_values"):
        games_inserted, moves_inserted = load_games([game], conn)

    assert (games_inserted, moves_inserted) == (0, 0)
