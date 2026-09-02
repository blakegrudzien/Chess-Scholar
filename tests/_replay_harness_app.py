"""A minimal harness for test_board_component.py's game-replay tests --
seeds st.session_state.game_path directly (a short, fixed 4-move game) so
the read-only-board and Prev/Next behavior can be exercised in a real
browser without needing a real recommend_resources() call (which needs
live Anthropic/DB credentials and isn't guaranteed to recommend a
Chessbase game for any given question).

Not named test_*.py deliberately, so pytest doesn't try to collect this as
a test module -- it's a Streamlit script, not a test file, and only makes
sense run via `streamlit run`.
"""

import sys
from pathlib import Path

# `streamlit run tests/_replay_harness_app.py` puts tests/ itself on
# sys.path, not the project root -- same issue src/app.py's own comment
# documents and works around, for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402
import streamlit as st  # noqa: E402

from src.ui.chat import render_main_screen  # noqa: E402
from src.ui.styles import apply_global_styles  # noqa: E402

st.set_page_config(page_title="Chess RAG Assistant", layout="wide")
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

st.title("Chess RAG Assistant")
render_main_screen()
