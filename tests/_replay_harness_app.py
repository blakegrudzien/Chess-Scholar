"""A Streamlit app for test_board_component.py's replay tests: it seeds a
fixed 4-move game into st.session_state.game_path, so replay can be tested
without a real resource lookup. Not named test_*.py, so pytest doesn't
collect it.
"""

import sys
from pathlib import Path

# `streamlit run` puts tests/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import streamlit as st  # noqa: E402

from src.ui.chat import render_main_screen  # noqa: E402
from src.ui.styles import apply_global_styles  # noqa: E402

st.set_page_config(page_title="Chess Scholar", layout="wide")
apply_global_styles()

if "game_path" not in st.session_state:
    board = chess.Board()
    path = [board.fen()]
    for san in ["e4", "e5", "Nf3", "Nc6"]:
        board.push_san(san)
        path.append(board.fen())
    st.session_state.game_path = path
    st.session_state.game_path_index = 0
    st.session_state.game_path_label = "Test Game"

st.title("Chess Scholar")
render_main_screen()
