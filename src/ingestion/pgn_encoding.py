"""PGN decoding shared by pgn_parser.py and annotation_extractor.py.

ChessBase exports are sometimes Windows-1252 instead of UTF-8, and some mix
the two in one file. Re-decoding a whole file as cp1252 would garble its
valid UTF-8, so only the bytes that fail UTF-8 are read as Windows-1252. A
file that is entirely cp1252 still decodes correctly that way.
"""

from __future__ import annotations

import codecs
from pathlib import Path

_HANDLER_NAME = "chessbase_cp1252_fallback"


def _cp1252_fallback(error: UnicodeDecodeError) -> tuple[str, int]:
    bad_bytes = error.object[error.start : error.end]
    return bad_bytes.decode("cp1252", errors="replace"), error.end


codecs.register_error(_HANDLER_NAME, _cp1252_fallback)

PGN_DECODE_ERRORS = _HANDLER_NAME


def read_pgn_text(path: str | Path) -> str:
    """Decode a whole PGN file as UTF-8 with per-byte Windows-1252 fallback."""
    return Path(path).read_bytes().decode("utf-8", errors=PGN_DECODE_ERRORS)
