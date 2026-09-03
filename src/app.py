"""Streamlit frontend for the chess RAG assistant. Ties together all four
layers via the agent in src/agent/chess_agent.py. Page setup only -- the
one screen itself (chat, board, game upload, resource recommendations)
lives in src/ui/chat.py; no tabs, everything is reachable on one scroll.

Known reliability caveats (see CLAUDE.md) surfaced directly in the UI:
- Chat answers synthesize retrieved human text and engine output; they are
  not the model's own independent tactical judgment.
- The "find similar games" comparison (Layer 4) is an approximate,
  illustrative match on exact opening moves, not a positional analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# `streamlit run src/app.py` puts src/ itself on sys.path, not the project
# root, so the absolute `from src....` imports below need this first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.ui.chat import render_main_screen  # noqa: E402
from src.ui.styles import apply_global_styles  # noqa: E402
from src.ui.tutorial_overlay import render_tutorial_trigger  # noqa: E402

# Without this, every logger.info/exception call anywhere in the app --
# chess_agent.ask()'s per-turn timing (built specifically to answer "is
# this actually slow" instead of guessing, see its own comment) and
# chat.py's new logger.exception calls on a failed API call included --
# goes nowhere: Python's logging module only guarantees a WARNING-or-above
# fallback handler when nothing has configured one (confirmed in
# engine_pool.py's own docstring), and INFO-level calls sit below that.
# basicConfig() is a safe no-op on Streamlit's per-interaction script
# reruns (it only configures the root logger's handlers once, unless
# force=True is passed, which this doesn't). Streamlit Cloud captures
# stdout/stderr into its own log viewer, so this is also what makes these
# logs visible at all on the deployed demo, not just a local run.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Chess RAG Assistant", layout="wide")
apply_global_styles()


def main() -> None:
    # Title and the tour trigger share one row, not stacked -- reclaims a
    # full line of vertical space toward fitting an unscrolled, fresh page
    # inside a laptop viewport (see styles.py's stMainBlockContainer rule
    # for the rest of that budget).
    title_col, tutorial_col = st.columns([6, 1], vertical_alignment="bottom")
    with title_col:
        st.title("Chess RAG Assistant")
    with tutorial_col:
        render_tutorial_trigger()
    render_main_screen()


main()
