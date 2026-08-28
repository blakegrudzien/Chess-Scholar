from unittest.mock import MagicMock, patch

from src.ingestion.annotation_extractor import AnnotationChunk
from src.ingestion.db_loader import get_connection, load_chunks, load_games
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
    # load_dotenv(override=True) would otherwise clobber the monkeypatched
    # value above with whatever's in this machine's real .env file -- not
    # a concern for get_connection() itself, just for keeping this test
    # isolated from real local config.
    with (
        patch("src.ingestion.db_loader.load_dotenv"),
        patch("src.ingestion.db_loader.psycopg2.connect") as mock_connect,
    ):
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
    # page_size must match the batch, or cur.rowcount silently reflects only
    # execute_values' last internal page (default page_size=100) instead of
    # the true total -- this is exactly the bug that slipped through before.
    assert games_call.kwargs["page_size"] == len(games_rows)

    moves_sql, moves_rows = moves_call.args[1], moves_call.args[2]
    assert "INSERT INTO moves" in moves_sql
    assert "ON CONFLICT (game_id, ply) DO NOTHING" in moves_sql
    assert moves_rows == [
        ("abc123", 1, "e4", "e2e4", "e2", "e4", "P", False, None, 0, game.moves[0].fen_after)
    ]
    assert moves_call.kwargs["page_size"] == len(moves_rows)

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


def test_load_games_flushes_in_batches():
    # A large PGN file shouldn't hold every row in memory until the end --
    # verify flush actually happens mid-stream, not just once at completion.
    games = [
        GameRecord(
            game_id=f"game{i}",
            white="Alice",
            black="Bob",
            event="Test",
            year=2021,
            eco_code="C00",
            result="1-0",
            source="lichess",
            moves=[],
        )
        for i in range(5)
    ]
    conn = MagicMock()

    def fake_execute_values(cur, _sql, rows, **_kwargs):
        cur.rowcount = len(rows)  # mirror real psycopg2: rowcount reflects this call's batch

    with (
        patch("src.ingestion.db_loader.GAMES_BATCH_SIZE", 2),
        patch(
            "src.ingestion.db_loader.execute_values", side_effect=fake_execute_values
        ) as mock_execute_values,
    ):
        games_inserted, _ = load_games(games, conn)

    assert games_inserted == 5
    assert mock_execute_values.call_count == 3  # batches of 2, 2, 1
    assert conn.commit.call_count == 3


def _sample_chunk() -> AnnotationChunk:
    return AnnotationChunk(
        source_type="game_annotation",
        game_id="abc123",
        source_title=None,
        author=None,
        year=2021,
        eco_code="C00",
        ply_or_page="8",
        text="Bxb4!?: Accepting the gambit.",
    )


def test_load_chunks_inserts_with_content_hash_key():
    chunk = _sample_chunk()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.rowcount = 1

    with patch("src.ingestion.db_loader.execute_values") as mock_execute_values:
        chunks_inserted = load_chunks([chunk], conn)

    assert chunks_inserted == 1
    mock_execute_values.assert_called_once()

    sql, rows = mock_execute_values.call_args.args[1], mock_execute_values.call_args.args[2]
    assert "INSERT INTO chunks" in sql
    assert "ON CONFLICT (chunk_hash) DO NOTHING" in sql
    assert len(rows) == 1
    assert mock_execute_values.call_args.kwargs["page_size"] == len(rows)
    row = rows[0]
    assert row[1:] == (
        "game_annotation",
        "abc123",
        None,
        None,
        2021,
        "C00",
        "8",
        "Bxb4!?: Accepting the gambit.",
    )
    assert isinstance(row[0], str) and len(row[0]) == 64  # sha256 hex digest

    conn.commit.assert_called_once()


def test_load_chunks_hash_is_deterministic_across_calls():
    chunk = _sample_chunk()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.rowcount = 1

    with patch("src.ingestion.db_loader.execute_values") as mock_execute_values:
        load_chunks([chunk], conn)
        first_hash = mock_execute_values.call_args.args[2][0][0]

        load_chunks([chunk], conn)
        second_hash = mock_execute_values.call_args.args[2][0][0]

    assert first_hash == second_hash


def test_load_chunks_skips_insert_when_no_chunks():
    conn = MagicMock()
    with patch("src.ingestion.db_loader.execute_values") as mock_execute_values:
        chunks_inserted = load_chunks([], conn)

    assert chunks_inserted == 0
    mock_execute_values.assert_not_called()
    conn.commit.assert_called_once()


def test_load_chunks_flushes_in_batches():
    chunks = [
        AnnotationChunk(
            source_type="game_annotation",
            game_id="abc123",
            source_title=None,
            author=None,
            year=2021,
            eco_code="C00",
            ply_or_page=str(i),
            text=f"chunk {i}",
        )
        for i in range(5)
    ]
    conn = MagicMock()

    def fake_execute_values(cur, _sql, rows, **_kwargs):
        cur.rowcount = len(rows)  # mirror real psycopg2: rowcount reflects this call's batch

    with (
        patch("src.ingestion.db_loader.CHUNKS_BATCH_SIZE", 2),
        patch(
            "src.ingestion.db_loader.execute_values", side_effect=fake_execute_values
        ) as mock_execute_values,
    ):
        chunks_inserted = load_chunks(chunks, conn)

    assert chunks_inserted == 5
    assert mock_execute_values.call_count == 3  # batches of 2, 2, 1
    assert conn.commit.call_count == 3
