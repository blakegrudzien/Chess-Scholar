"""Entry point for the Streamlit app: page setup, logging and the database
keep-warm ping. The screen itself is built in src/ui/chat.py.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# `streamlit run src/app.py` puts src/ itself on sys.path, not the project
# root, so the absolute `from src....` imports below need this first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.ui.activity_listener import keystroke_after_inactivity  # noqa: E402
from src.ui.chat import render_main_screen  # noqa: E402
from src.ui.resources import (  # noqa: E402
    keep_warm_ping_due,
    ping_database_in_background,
    session_inactive,
)
from src.ui.styles import apply_global_styles  # noqa: E402
from src.ui.tutorial_overlay import render_tutorial_trigger  # noqa: E402

# Without a handler, Python only prints WARNING and above, so INFO logs
# (per-turn timing, for one) would be lost. basicConfig does nothing on
# later reruns. Streamlit Cloud shows stdout and stderr in its log viewer.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Chess Scholar", layout="wide")
apply_global_styles()


# How often the keep-warm fragment checks whether a ping is due. Every full
# run restarts a fragment's timer, so a 270s timer would never fire for a
# visitor who keeps clicking; checking often keeps pings at most
# KEEP_WARM_PING_INTERVAL_SECONDS plus this apart.
KEEP_WARM_CHECK_SECONDS = 15


@st.fragment(run_every=KEEP_WARM_CHECK_SECONDS)
def _keep_database_warm() -> None:
    # Renders only the invisible keystroke listener. Runs with every full
    # run (the first is the page-load warmup) and on its own timer, which
    # never interrupts a full run.
    now = time.monotonic()
    # Typing only reaches Python on submit, so the first keystroke after a
    # long idle is reported separately to wake the database early.
    if keystroke_after_inactivity(armed=session_inactive(now, st.session_state.last_interaction)):
        st.session_state.last_interaction = now
    if keep_warm_ping_due(
        now, st.session_state.get("last_db_ping"), st.session_state.last_interaction
    ):
        st.session_state.last_db_ping = now
        ping_database_in_background()


def main() -> None:
    # Every full run follows an interaction or the page load; the keep-warm
    # timer reruns only its fragment, so it doesn't count.
    st.session_state.last_interaction = time.monotonic()
    # Title and tour button share a row to save vertical space.
    title_col, tutorial_col = st.columns([6, 1], vertical_alignment="bottom")
    with title_col:
        st.title("Chess Scholar", anchor=False)
    with tutorial_col:
        render_tutorial_trigger()
    render_main_screen()
    # Last, so the page has rendered before the warmup ping starts.
    _keep_database_warm()


main()
