"""Shared PGN decoding for pgn_parser.py and annotation_extractor.py.

ChessBase exports are sometimes Windows-1252 rather than UTF-8 -- and some
exports mix the two within a single file (observed in practice: a file that's
otherwise valid UTF-8 with a handful of raw Windows-1252 bytes embedded in
comments, likely from concatenating exports written by different annotators).

A whole-file "try UTF-8, else re-decode everything as cp1252" fallback is
wrong for the mixed case: re-decoding the entire file as cp1252 would mangle
every correctly-encoded multi-byte UTF-8 character elsewhere in the file.
Instead, decode UTF-8 normally and only reinterpret the specific byte(s) that
fail UTF-8 validation as Windows-1252, leaving everything else untouched.
This also degrades correctly to "whole file is cp1252": every non-ASCII byte
then fails UTF-8 validation and falls back individually.
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
