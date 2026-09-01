"""A draggable chess board, built as an st.components.v2 component around
chessboard.js -- replaces the old static chess.svg image plus a separate
click-grid with a single interactive widget.

Deliberately not paired with chess.js: move legality stays entirely in
python-chess, the same source of truth the rest of this app already uses.
See chat.py's _render_board_panel docstring for the optimistic-UI flow this
implies.

isolate_styles=False is required, not a style choice: chessboard.js uses
jQuery ID-based lookups (`$("#" + squareId)`) against the *document* to
manage its own square/piece elements internally. Those lookups can't reach
inside a Shadow DOM, which is what isolate_styles=True (the component
default) would mount this in -- confirmed by reading chessboard.js's own
vendored source before wiring this up, not assumed. The tradeoff is that
chessboard.js's CSS (loaded via the css= string below) applies to the whole
page rather than being sandboxed to just this component; its class names
already carry library-generated hash suffixes (e.g. "board-b72b1"), making a
collision with this app's own CSS unlikely.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_DIR = Path(__file__).parent

_JS = "\n".join(
    (
        (_DIR / "vendor" / "jquery-3.7.1.min.js").read_text(encoding="utf-8"),
        (_DIR / "vendor" / "chessboard-1.0.0.min.js").read_text(encoding="utf-8"),
        (_DIR / "generated" / "piece_images.js").read_text(encoding="utf-8"),
        (_DIR / "wiring.js").read_text(encoding="utf-8"),
    )
)
_CSS = (_DIR / "vendor" / "chessboard-1.0.0.min.css").read_text(encoding="utf-8")

_chess_board_component = st.components.v2.component(
    "chess_rag_board",
    css=_CSS,
    js=_JS,
    isolate_styles=False,
)


def chess_board(
    fen: str, *, size: int = 300, generation: int = 0, key: str | None = None
) -> dict[str, str] | None:
    """Render a draggable board at `fen`. Returns {"from": sq, "to": sq} for
    the drop that just happened, or None if nothing new was dropped since
    the last script run -- "drop" is a trigger value (see wiring.js), so it
    resets to None automatically after one rerun rather than replaying.

    `generation` must change on every call where the board should visually
    re-sync, independent of whether `fen` itself changed. An illegal drop is
    the reason this exists: chessboard.js optimistically shows the piece at
    the dropped square, but when python-chess rejects the move,
    st.session_state.board -- and therefore `fen` -- is exactly what it was
    *before* the drop, so nothing about `data` would otherwise differ from
    the previous call. Without a distinct generation value, this component
    has no signal to re-render, and the piece is left stuck at the illegal
    square instead of snapping back. See chat.py's board_generation counter.
    """
    result = _chess_board_component(
        data={"fen": fen, "size": size, "generation": generation},
        key=key,
        height=size,
        on_drop_change=lambda: None,
    )
    return result.drop
