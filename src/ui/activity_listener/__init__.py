"""Detects the first keystroke in a text box after a session has gone
inactive, built as an st.components.v2 component like the board.

Typing in st.chat_input or st.text_input reaches Python only on submit, so
without this a visitor returning after a long break would wake the
database only when they sent their question. Rendered inside the
keep-warm fragment in app.py, so a keystroke reruns that fragment alone
and leaves the rest of the page, and the text being typed, untouched.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_JS = (Path(__file__).parent / "listener.js").read_text(encoding="utf-8")

_activity_listener_component = st.components.v2.component("chess_rag_activity_listener", js=_JS)


def keystroke_after_inactivity(*, armed: bool) -> bool:
    """Render the listener. Returns True on the rerun triggered by the
    first keystroke while `armed`, False otherwise. `armed` should be True
    only while the session counts as inactive.
    """
    result = _activity_listener_component(
        data={"armed": armed},
        height=1,
        on_keystroke_change=lambda: None,
    )
    return bool(result.keystroke)
