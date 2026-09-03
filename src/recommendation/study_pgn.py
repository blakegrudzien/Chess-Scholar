"""Split a Lichess study's PGN export into its constituent chapters.

Lichess's per-study PGN export (LichessClient.fetch_study_pgn) encodes each
chapter as one consecutive PGN game block within the same text blob --
that's a Lichess study convention, not a general PGN-file property, so this
lives in src/recommendation/ rather than src/ingestion/pgn_parser.py, which
is the ChessBase ingestion pipeline's own file-based parser and has no
reason to know about it.

Both the labeling app's comment preview and the quality classifier's
feature extraction need this same split, so it's written once here instead
of twice.
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

    Stops (without raising) at the first chapter that fails to parse.
    python-chess is normally lenient about malformed PGN, so a hard parse
    failure here is rare, but a caller working through arbitrary scraped
    content shouldn't crash over one bad chapter in an otherwise fine
    study -- it just sees fewer chapters than the study actually has.
    """
    stream = io.StringIO(pgn_text)
    while True:
        try:
            game = chess.pgn.read_game(stream)
        except Exception:
            # Logged, not silently swallowed -- this function backs both
            # the quality classifier's training features and the search
            # index's embedding text (see module docstring), so a chapter
            # silently dropped here quietly degrades training data or
            # search relevance with nothing pointing at why. The other
            # "skip and log" spots in this subsystem (e.g. lichess_
            # scraper._parse_card's "unexpected markup" warning) already
            # follow this convention; this was the one place that didn't.
            logger.warning("Skipping the rest of a study's chapters after a PGN parse error")
            return
        if game is None:
            return
        yield game
