import chess
import pytest

from src.ingestion.pgn_parser import parse_pgn

SAMPLE_PGN = """[Event "Test Game"]
[Site "?"]
[Date "2021.05.01"]
[Round "1"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[ECO "C00"]

1. e4 d5 2. exd5 Qxd5 1-0
"""


@pytest.fixture
def pgn_file(tmp_path):
    path = tmp_path / "sample.pgn"
    path.write_text(SAMPLE_PGN, encoding="utf-8")
    return path


def test_parse_pgn_game_metadata(pgn_file):
    games = list(parse_pgn(pgn_file, source="lichess"))
    assert len(games) == 1

    game = games[0]
    assert game.white == "Alice"
    assert game.black == "Bob"
    assert game.event == "Test Game"
    assert game.year == 2021
    assert game.eco_code == "C00"
    assert game.result == "1-0"
    assert game.source == "lichess"
    assert game.game_id  # non-empty, unique id assigned per game


def test_parse_pgn_move_records(pgn_file):
    game = next(parse_pgn(pgn_file, source="lichess"))
    assert len(game.moves) == 4

    m1, m2, m3, m4 = game.moves

    assert (m1.ply, m1.move_san, m1.move_uci) == (1, "e4", "e2e4")
    assert (m1.from_sq, m1.to_sq, m1.piece) == ("e2", "e4", "P")
    assert m1.is_capture is False
    assert m1.captured_piece is None
    assert m1.material_delta == 0

    assert (m2.move_san, m2.piece) == ("d5", "p")
    assert m2.is_capture is False

    # exd5: white pawn captures the black pawn on d5.
    assert m3.move_san == "exd5"
    assert m3.piece == "P"
    assert m3.is_capture is True
    assert m3.captured_piece == "p"
    assert m3.material_delta == 1

    # Qxd5: black queen recaptures the white pawn on d5.
    assert m4.move_san == "Qxd5"
    assert m4.piece == "q"
    assert m4.is_capture is True
    assert m4.captured_piece == "P"
    assert m4.material_delta == 1


def test_fen_after_reflects_position_at_each_ply(pgn_file):
    game = next(parse_pgn(pgn_file, source="lichess"))
    final_board = chess.Board(game.moves[-1].fen_after)
    assert final_board.piece_at(chess.D5).symbol() == "q"
    assert final_board.piece_at(chess.E4) is None


def test_game_id_is_deterministic_across_reparses(pgn_file):
    # db_loader relies on game_id being stable across runs so that
    # ON CONFLICT DO NOTHING actually dedupes re-ingested games.
    first_pass = next(parse_pgn(pgn_file, source="lichess")).game_id
    second_pass = next(parse_pgn(pgn_file, source="lichess")).game_id
    assert first_pass == second_pass


def test_game_id_differs_for_different_games(tmp_path):
    original_path = tmp_path / "sample.pgn"
    original_path.write_text(SAMPLE_PGN, encoding="utf-8")

    other_path = tmp_path / "other.pgn"
    other_path.write_text(SAMPLE_PGN.replace("Alice", "Carol"), encoding="utf-8")

    first_id = next(parse_pgn(original_path, source="lichess")).game_id
    second_id = next(parse_pgn(other_path, source="lichess")).game_id
    assert first_id != second_id
