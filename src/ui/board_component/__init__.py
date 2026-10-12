"""A chess board, built as an st.components.v2 component around
chessboard.js, with chess.js so the browser knows the rules.

The browser plays legal moves instantly and rejects illegal ones without a
server round trip; python-chess on the server still checks and records
every move and has the final say. See wiring.js for how the two stay in
sync, and board_panel.py's render_board_panel docstring for the Python side.

isolate_styles=False is required, not a style choice: chessboard.js uses
jQuery ID-based lookups (`$("#" + squareId)`) against the *document* to
manage its own square/piece elements, which can't reach inside a Shadow DOM
(isolate_styles=True, the component default). The tradeoff is that
chessboard.js's CSS applies to the whole page rather than being sandboxed to
this component; its class names carry library-generated hash suffixes
(e.g. "board-b72b1"), making a collision with this app's own CSS unlikely.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_DIR = Path(__file__).parent

# chess.js ships an ES module with named exports and no imports, so it can
# share this one module with wiring.js's default export. Its source map isn't
# vendored, so the reference to it is dropped to avoid a 404 in devtools.
_CHESS_JS = (
    (_DIR / "vendor" / "chess-1.4.0.esm.js")
    .read_text(encoding="utf-8")
    .replace("//# sourceMappingURL=chess.js.map", "")
)

_JS = "\n".join(
    (
        (_DIR / "vendor" / "jquery-3.7.1.min.js").read_text(encoding="utf-8"),
        (_DIR / "vendor" / "chessboard-1.0.0.min.js").read_text(encoding="utf-8"),
        _CHESS_JS,
        (_DIR / "generated" / "piece_images.js").read_text(encoding="utf-8"),
        (_DIR / "wiring.js").read_text(encoding="utf-8"),
    )
)
# A fix layered on top of the vendored CSS rather than edited into that
# file directly, so re-vendoring a future chessboard.js release doesn't
# silently drop it. chessboard.js's own coordinate labels (.notation-322f9)
# sit on the corner of every edge square, position:absolute; a piece image
# there is plain position:static. A positioned element always paints above
# a static sibling regardless of z-index or DOM order, which is why the
# label covers the piece rather than the reverse.
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


def chess_board(
    fen: str,
    *,
    size: int = 300,
    generation: int = 0,
    draggable: bool = True,
    key: str | None = None,
) -> dict[str, str] | None:
    """Render a board at `fen`, `size` pixels wide at most and narrower
    when the screen is. Returns {"from": sq, "to": sq} for a move made
    since the last run, or None. "drop" is a trigger value, so it resets
    after one rerun.

    Pass a `key` so the board persists across reruns: wiring.js then keeps
    the moves it drew itself and redraws only when `fen` differs from what
    it shows. `generation` must change after every move Python processes,
    so the board hears back even when Python's FEN matches the one it
    already drew (its confirmation of a move) or when a rejected move leaves
    `fen` unchanged (its signal to put the pieces back).

    draggable=False makes the board read-only (game replay, and while an
    answer generates). Callers still check their own state before acting
    on a move.
    """
    result = _chess_board_component(
        data={"fen": fen, "size": size, "generation": generation, "draggable": draggable},
        key=key,
        # Left at Streamlit's default, height="content": the board shrinks
        # on narrow screens (see wiring.js), and its frame follows.
        on_drop_change=lambda: None,
    )
    return result.drop
