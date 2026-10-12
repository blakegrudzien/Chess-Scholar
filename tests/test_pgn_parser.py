import chess
import chess.pgn
import pytest

from src.ingestion.pgn_parser import compute_game_id, parse_game, parse_pgn, parse_year

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


def test_compute_game_id_rejects_a_header_containing_the_delimiter():
    # A "|" in a header could make two different games hash the same, and
    # one would be silently dropped at insert.
    headers = chess.pgn.Headers(
        {
            "White": "Weird|Name",
            "Black": "Bob",
            "Event": "Test",
            "Date": "2021.05.01",
            "Round": "1",
            "Result": "1-0",
        }
    )
    with pytest.raises(ValueError, match=r"\|"):
        compute_game_id(headers, ["e4", "d5"])


def test_parse_game_stops_at_the_first_illegal_move_instead_of_crashing():
    """python-chess parses a mainline move without checking it's legal, and
    board.san() then fails. User uploads reach parse_game, so it returns
    the game up to that move instead of crashing."""
    game = chess.pgn.Game()
    node = game.add_variation(chess.Move.from_uci("e2e4"))
    node = node.add_variation(chess.Move.from_uci("e7e5"))
    # White moves twice in a row, which is illegal.
    node.add_variation(chess.Move.from_uci("e2e4"))

    record = parse_game(game, source="user_upload")

    assert [m.move_san for m in record.moves] == ["e4", "e5"]


def test_parse_game_truncates_at_threefold_repetition_when_requested():
    """An uploaded PGN can shuffle pieces back and forth legally forever.
    truncate_at_repetition stops at a position's third occurrence (where a
    draw can be claimed); it's off by default so ingested games that play
    on past a repetition are kept whole."""
    game = chess.pgn.Game()
    node = game
    # Nf3 Nf6 Ng1 Ng8 three times: the start position occurs for the third
    # time after 8 plies, so parsing stops there, short of all 12.
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in cycle * 3:
        node = node.add_variation(chess.Move.from_uci(uci))

    record = parse_game(game, source="user_upload", truncate_at_repetition=True)

    assert [m.move_san for m in record.moves] == ["Nf3", "Nf6", "Ng1", "Ng8"] * 2


def test_parse_game_does_not_truncate_at_repetition_by_default():
    """The corpus-ingestion default: a real game can play on past a
    repeated position (the draw is claimable, not automatic), so this must
    not silently drop moves unless a caller opts in.
    """
    game = chess.pgn.Game()
    node = game
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for uci in cycle * 3:
        node = node.add_variation(chess.Move.from_uci(uci))

    record = parse_game(game, source="lichess")

    assert len(record.moves) == 12


def test_parse_year_accepts_a_normal_date():
    assert parse_year("1995.03.10") == 1995


def test_parse_year_returns_none_for_missing_or_unknown_date():
    # "????.??.??" is ChessBase's normal placeholder for an unknown date.
    assert parse_year(None) is None
    assert parse_year("") is None
    assert parse_year("????.??.??") is None


def test_parse_year_rejects_an_implausible_year():
    """A malformed date like "202.01.01" is rejected instead of becoming the
    year 202."""
    assert parse_year("202.01.01") is None  # truncated
    assert parse_year("0001.01.01") is None  # before chess as we know it existed
    assert parse_year("9999.01.01") is None  # implausibly far future
