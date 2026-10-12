import tomllib
from pathlib import Path

import chess
import chess.pgn
from streamlit.testing.v1 import AppTest

from src.ui.input_limits import MAX_QUESTION_CHARS, SharedQuestionLimit

ROOT = Path(__file__).resolve().parent.parent


def test_a_question_over_the_length_limit_is_refused_with_a_message():
    at = AppTest.from_string(f"""
import streamlit as st
from src.ui.input_limits import accept_question_length
st.session_state["_ok"] = accept_question_length("x" * {MAX_QUESTION_CHARS + 1})
st.session_state["_short_ok"] = accept_question_length("Is the Najdorf sound?")
""")
    at.run()
    assert not at.exception
    assert at.session_state["_ok"] is False
    assert at.session_state["_short_ok"] is True
    assert any("too long" in w.value for w in at.warning)


def test_shared_limit_allows_up_to_the_limit_then_refuses_until_the_window_passes():
    limit = SharedQuestionLimit(limit=3, window_seconds=100)
    assert [limit.try_acquire(now=t) for t in (0, 1, 2)] == [True, True, True]
    assert limit.try_acquire(now=3) is False
    assert limit.try_acquire(now=100.5) is True  # the first question has aged out


def test_submit_question_refuses_when_the_shared_limit_is_used_up():
    at = AppTest.from_string("""
from unittest.mock import MagicMock, patch
import streamlit as st
from src.ui import chat

st.session_state.chat_history = []
exhausted = MagicMock()
exhausted.try_acquire.return_value = False
with (
    patch("src.ui.chat.get_shared_question_limit", return_value=exhausted),
    patch("src.ui.chat.ask_with_status") as ask,
):
    submitted = chat._submit_question("How should White meet the Sicilian?")
st.session_state["_submitted"] = submitted
st.session_state["_asked"] = ask.called
""")
    at.run()
    assert not at.exception
    assert at.session_state["_submitted"] is False
    assert at.session_state["_asked"] is False
    assert at.session_state.chat_history == []
    assert any("today" in w.value for w in at.warning)


def _long_game_pgn(plies: int) -> bytes:
    # A fixed sequence of legal, non-repeating moves (the first legal move in
    # a stable order that doesn't repeat a position).
    board = chess.Board()
    seen = {board.board_fen()}
    while len(board.move_stack) < plies:
        for move in sorted(board.legal_moves, key=lambda m: m.uci()):
            board.push(move)
            if board.board_fen() not in seen and not board.is_game_over():
                seen.add(board.board_fen())
                break
            board.pop()
        else:
            break
    assert len(board.move_stack) >= plies
    return str(chess.pgn.Game.from_board(board)).encode()


def test_only_the_opening_of_an_uploaded_game_goes_into_the_question():
    from unittest.mock import MagicMock

    from src.agent.chess_agent import MAX_PLY_WINDOW
    from src.ui.chat import _describe_uploaded_game

    upload = MagicMock()
    upload.getvalue.return_value = _long_game_pgn(MAX_PLY_WINDOW + 20)
    question, _ = _describe_uploaded_game(upload, "")
    move_list = question.split(": ", 1)[1].split(".")[0]
    assert len(move_list.split(", ")) == MAX_PLY_WINDOW


def test_uploads_are_capped_at_one_megabyte():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["server"]["maxUploadSize"] == 1
