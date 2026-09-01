"""Streamlit frontend for the chess RAG assistant. Ties together all four
layers via the agent in src/agent/chess_agent.py. Page setup and tab wiring
only -- the tabs themselves live in src/ui/.

Known reliability caveats (see CLAUDE.md) surfaced directly in the UI:
- Chat answers synthesize retrieved human text and engine output; they are
  not the model's own independent tactical judgment.
- The "find similar games" comparison (Layer 4) is an approximate,
  illustrative match on exact opening moves, not a positional analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run src/app.py` puts src/ itself on sys.path, not the project
# root, so the absolute `from src....` imports below need this first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.ui.chat import render_chat_tab  # noqa: E402
from src.ui.styles import apply_global_styles  # noqa: E402
from src.ui.upload import render_pgn_upload_tab  # noqa: E402

st.set_page_config(page_title="Chess RAG Assistant", layout="wide")
apply_global_styles()


def main() -> None:
    st.title("Chess RAG Assistant")
    chat_tab, upload_tab = st.tabs(["Chat", "Analyze Your Game"])
    with chat_tab:
        render_chat_tab()
    with upload_tab:
        render_pgn_upload_tab()


main()
