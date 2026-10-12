"""Split a Lichess study's PGN export into chapters. Each chapter is one
PGN game in the export, a Lichess convention, so this lives here rather
than in the ChessBase ingestion parser. Used by the labeling app's preview
and the classifier's feature extraction.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import chess.pgn

logger = logging.getLogger(__name__)


def iter_study_chapters(pgn_text: str) -> Iterator[chess.pgn.Game]:
    """Yield one chess.pgn.Game per chapter in a study's PGN export, in
    chapter order.

    Stops at the first chapter that fails to parse (rare, since
    python-chess is lenient) rather than raising.
    """
    stream = io.StringIO(pgn_text)
    while True:
        try:
            game = chess.pgn.read_game(stream)
        except Exception:
            # Logged: a dropped chapter weakens training data and search
            # text, and should leave a trace.
            logger.warning("Skipping the rest of a study's chapters after a PGN parse error")
            return
        if game is None:
            return
        yield game
