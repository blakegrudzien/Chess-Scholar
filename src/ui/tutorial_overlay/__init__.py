"""The "How this works" tour: an st.components.v2 component that spotlights
six parts of the page in turn, each with a short tooltip.

isolate_styles=False because wiring.js adds its overlay to document.body,
outside the component's own element, where a shadow DOM would keep this
CSS from reaching it.
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

# Selectors use the `st-key-<key>` class Streamlit adds for a `key=`; the
# chat input is the page's only one, so its data-testid is enough.
_STEPS = [
    {
        "selector": ".st-key-chat_panel",
        "title": "Ask anything about chess",
        "text": (
            "Openings, positions, strategy, history: one chat handles "
            "all of it and figures out what to draw on for each question. "
            "There's no mode to pick."
        ),
    },
    {
        "selector": ".st-key-tutorial_board_target",
        "title": "Play on the board",
        "text": (
            "Drag pieces to make moves. Illegal moves are rejected right on "
            "the board, and python-chess on the server double-checks every move."
        ),
        # Phones move pieces by tapping (see board_component/wiring.js).
        "phoneText": (
            "Tap a piece, then tap where it should go. Illegal moves are "
            "rejected right on the board, and python-chess on the server "
            "double-checks every move."
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
        # Grayed out until its results are ready (see _render_resource_
        # recommendations' own docstring) rather than absent, so this
        # selector always resolves, on a fresh page included.
        "selector": ".st-key-find_resources",
        "title": "Find related resources",
        "text": (
            "After each answer, matching Lichess studies and master games "
            "from the corpus are found in the background while you read. "
            "This unlocks when they're ready."
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


# On phones the board column comes first and the chat last (see theme.css),
# so the tour follows that top-to-bottom order instead of the desktop one:
# board, ask about a position, Evaluate, resources, chat, upload.
_PHONE_STEP_ORDER = [1, 2, 3, 4, 0, 5]


def render_tutorial_trigger() -> None:
    """The "How this works" button and the tour it launches.

    tutorial_launch_id counts clicks rather than being a flag: the
    component restarts only when its data changes, so a flag would never
    reopen the tour after the first time.
    """
    if "tutorial_launch_id" not in st.session_state:
        st.session_state.tutorial_launch_id = 0

    # secondary, not tertiary: tertiary renders as bare text with no
    # border, which reads as a stray label rather than a
    # clickable control once it's sitting alone in the header with nothing
    # else nearby to anchor it. A bordered button matches Reset board/Undo
    # last move and every other secondary action already on the page.
    if st.button("How this works", key="tutorial_trigger", type="secondary"):
        st.session_state.tutorial_launch_id += 1

    _tutorial_component(
        data={
            "launchId": st.session_state.tutorial_launch_id,
            "steps": _STEPS,
            "phoneStepOrder": _PHONE_STEP_ORDER,
        },
        height=1,
    )
