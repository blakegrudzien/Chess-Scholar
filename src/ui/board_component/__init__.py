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
# A fix layered on top of the vendored CSS rather than edited into that
# file directly, so re-vendoring a future chessboard.js release doesn't
# silently drop it. chessboard.js's own coordinate labels (.notation-322f9)
# sit on the corner of every edge square, position:absolute; a piece image
# there is plain position:static -- confirmed live (piece computed
# position: static, notation: absolute) that a positioned element always
# paints above a static sibling regardless of z-index or DOM order, which
# is why the label was covering the piece rather than the reverse.
# Explicitly positioning the piece with a z-index gives it something to
# actually win the stacking comparison against, rather than trying to push
# the label behind it with a negative z-index, which would escape this
# square's local stacking and interact unpredictably with the *board's*
# own stacking context instead. The label's own font-size is also reduced
# so it reads as a small corner mark instead of competing for the same
# visual space as the piece art even where the two still meet.
_PIECE_STACKING_FIX_CSS = """
.notation-322f9 { font-size: 9px; }
img[class*="piece-"] { position: relative; z-index: 1; }
"""
_CSS = (_DIR / "vendor" / "chessboard-1.0.0.min.css").read_text(encoding="utf-8") + (
    _PIECE_STACKING_FIX_CSS
)

_chess_board_component = st.components.v2.component(
    "chess_rag_board",
    css=_CSS,
    js=_JS,
    isolate_styles=False,
)

# Streamlit gives the element container Python's `height=` renders into
# overflow-y: auto (confirmed by inspecting the live DOM: stElementContainer,
# not this component itself, is what has the scrollbar) -- fine as long as
# the actual chessboard.js content never exceeds that exact pixel height,
# but a brief mismatch (the container's declared height and the JS side's
# actual rendered content settling on different frames, right after a
# rebuild) is enough to trip it into showing a scrollbar for a moment before
# it corrects itself. A small buffer between the requested container height
# and the board's own true pixel size (still exactly `size`, unaffected)
# gives that transient mismatch somewhere to go instead of overflowing.
_HEIGHT_BUFFER_PX = 10


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
        height=size + _HEIGHT_BUFFER_PX,
        on_drop_change=lambda: None,
    )
    return result.drop
