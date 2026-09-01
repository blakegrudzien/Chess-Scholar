"""Generate the draggable board's piece image set as a JS data-URI map, so
the piece art matches the chess.svg pieces used everywhere else in the app
(the chat avatars, the old static board) instead of chessboard.js's stock
Wikipedia piece images.

The custom Streamlit component's JS is passed to st.components.v2.component
as an inline string, not served from a static directory, so there's no
relative URL a browser could fetch a vendored image file from. Baking each
piece as a base64 SVG data URI directly into a JS object literal sidesteps
that entirely -- no network/file fetch at render time.

Regenerate whenever python-chess's SVG piece rendering changes:
    .venv/bin/python -m scripts.generate_board_piece_images
"""

from __future__ import annotations

import base64
from pathlib import Path

import chess
import chess.svg

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "ui"
    / "board_component"
    / "generated"
    / "piece_images.js"
)

# chessboard.js's own two-letter piece-code convention (color + piece
# letter), matched here so the generated map can be indexed directly by
# whatever code chessboard.js's pieceTheme callback receives.
_PIECES = [
    (chess.PAWN, chess.WHITE, "wP"),
    (chess.KNIGHT, chess.WHITE, "wN"),
    (chess.BISHOP, chess.WHITE, "wB"),
    (chess.ROOK, chess.WHITE, "wR"),
    (chess.QUEEN, chess.WHITE, "wQ"),
    (chess.KING, chess.WHITE, "wK"),
    (chess.PAWN, chess.BLACK, "bP"),
    (chess.KNIGHT, chess.BLACK, "bN"),
    (chess.BISHOP, chess.BLACK, "bB"),
    (chess.ROOK, chess.BLACK, "bR"),
    (chess.QUEEN, chess.BLACK, "bQ"),
    (chess.KING, chess.BLACK, "bK"),
]


def main() -> None:
    entries = []
    for piece_type, color, code in _PIECES:
        svg = chess.svg.piece(chess.Piece(piece_type, color), size=45)
        data_uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode(
            "ascii"
        )
        entries.append(f'  "{code}": "{data_uri}"')

    js = "const CHESS_RAG_PIECE_IMAGES = {\n" + ",\n".join(entries) + "\n};\n"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(js, encoding="utf-8")
    print(f"Wrote {len(_PIECES)} piece images to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
