"""A small harness for tests/test_activity_listener.py: renders the
activity listener inside a fragment, armed until its first keystroke, next
to a plain text input. Counters on the page show whether a keystroke was
recorded and whether it caused a full run or only a fragment run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.ui.activity_listener import keystroke_after_inactivity  # noqa: E402

st.session_state.setdefault("full_runs", 0)
st.session_state.setdefault("keystrokes_recorded", 0)
st.session_state.setdefault("armed", True)
st.session_state.full_runs += 1

st.text_input("Question", key="question")


@st.fragment
def activity() -> None:
    if keystroke_after_inactivity(armed=st.session_state.armed):
        st.session_state.keystrokes_recorded += 1
        st.session_state.armed = False
    st.write(f"keystrokes recorded: {st.session_state.keystrokes_recorded}")


activity()
st.write(f"full runs: {st.session_state.full_runs}")
