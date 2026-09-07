import chess

from src.ui.board_panel import game_status


def _board_after(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def test_reports_nothing_while_the_game_is_in_progress():
    assert game_status(chess.Board()) is None
    assert game_status(_board_after("e4", "e5")) is None


def test_names_the_winner_on_checkmate():
    # Fool's mate.
    status = game_status(_board_after("f3", "e5", "g4", "Qh4"))

    assert status == "Checkmate -- Black wins."


def test_reports_stalemate_as_a_draw():
    status = game_status(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))

    assert status == "Draw by stalemate."


def test_reports_insufficient_material():
    status = game_status(chess.Board("7k/8/6K1/8/8/8/8/8 w - - 0 1"))

    assert status == "Draw by insufficient material."


def test_threefold_repetition_is_reported_as_claimable_not_as_a_result():
    """Under FIDE rules threefold repetition is a player's right to claim,
    not an automatic end to the game. Reporting it as a finished draw is a
    common chess-app bug, and it matters here because the board stays
    playable afterward -- the game genuinely has not ended.
    """
    board = chess.Board()
    for _ in range(2):
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            board.push_san(san)

    assert not board.is_game_over()
    assert game_status(board) == "A draw can be claimed here, by threefold repetition."


def test_detects_terminal_positions_loaded_from_a_bare_fen():
    """Game replay rebuilds each ply from a FEN, so board.move_stack is
    empty there. Checkmate is a property of the position alone and still has
    to be detected without any move history to inspect.
    """
    checkmated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")

    assert not checkmated.move_stack
    assert game_status(checkmated) == "Checkmate -- Black wins."
