"""A quick guided tour, built as an st.components.v2 component -- spotlights
six existing UI elements in turn with a short tooltip, launched from a
single "How this works" button (see app.py's call site).

isolate_styles=False is required, not a style choice: wiring.js appends its
backdrop/spotlight/tooltip elements straight to document.body rather than to
the parentElement Streamlit hands it (see that file's own comment for why a
position: fixed overlay needs to live there instead), so a Shadow DOM around
parentElement's subtree -- isolate_styles=True, the component default --
would scope this file's CSS to a subtree those elements sit outside of, and
none of it would apply to them at all.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_DIR = Path(__file__).parent

_JS = (_DIR / "wiring.js").read_text(encoding="utf-8")
_CSS = (_DIR / "style.css").read_text(encoding="utf-8")

_tutorial_component = st.components.v2.component(
    "chess_rag_tutorial",
    css=_CSS,
    js=_JS,
    isolate_styles=False,
)

# Each selector targets the `st-key-<key>` CSS class Streamlit generates for
# any widget or container given a `key=` (confirmed in the installed
# Streamlit source, elements/layouts.py and elements/widgets/button.py),
# except the chat input, which needs no key of its own -- it's the only
# st.chat_input on the page, so its default data-testid is already unique.
_STEPS = [
    {
        "selector": ".st-key-chat_panel",
        "title": "Ask anything about chess",
        "text": (
            "Openings, positions, strategy, history -- one chat handles "
            "all of it and figures out what to draw on for each question. "
            "There's no mode to pick."
        ),
    },
    {
        "selector": ".st-key-tutorial_board_target",
        "title": "Play on the board",
        "text": (
            "Drag pieces to make moves. Legality is checked by "
            "python-chess, not the board itself, so an illegal drop just "
            "snaps back."
        ),
    },
    {
        "selector": ".st-key-position_question_form",
        "title": "Ask about a position",
        "text": (
            "Type a question about the exact position shown on the board "
            "and it's answered in the chat, with that position attached."
        ),
    },
    {
        "selector": ".st-key-evaluate_position",
        "title": "Analyze with Stockfish",
        "text": (
            "Get an engine evaluation and best move for the current "
            "position, grounded in Stockfish rather than model guesswork."
        ),
    },
    {
        # Grayed out until eligible (see _render_resource_recommendations'
        # own docstring) rather than absent -- unlike the old design, this
        # selector always resolves, on a fresh page included.
        "selector": ".st-key-find_resources",
        "title": "Find related resources",
        "text": (
            "Grayed out for now -- once you've asked something, this "
            "pulls matching Lichess studies and annotated games from the "
            "corpus for that question."
        ),
    },
    {
        "selector": '[data-testid="stChatInput"]',
        "title": "Upload a game",
        "text": (
            "Attach a PGN from the chat box for an illustrative comparison "
            "against similar games in the corpus."
        ),
    },
]


def render_tutorial_trigger() -> None:
    """A small "How this works" button that launches the tour.

    st.session_state.tutorial_launch_id is a counter, not a bool: this
    component is unkeyed, so Streamlit remounts it (a fresh JS instance,
    calling the previous one's cleanup first) only when its serialized
    `data` payload changes -- the same mechanism chat.py's board_generation
    counter relies on for chess_board(). A bool would stay True forever
    after the first click, so a second click meant to reopen the tour after
    closing it would send the same payload and never remount anything.
    Incrementing on every click guarantees a new payload each time.
    """
    if "tutorial_launch_id" not in st.session_state:
        st.session_state.tutorial_launch_id = 0

    if st.button("How this works", key="tutorial_trigger", type="tertiary"):
        st.session_state.tutorial_launch_id += 1

    _tutorial_component(
        data={"launchId": st.session_state.tutorial_launch_id, "steps": _STEPS},
        height=1,
    )
