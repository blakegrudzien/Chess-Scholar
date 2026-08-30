"""Shared helper for the content-hash natural keys used across ingestion
(pgn_parser.compute_game_id, db_loader._chunk_hash): both build a SHA-256
hash over several fields joined with a delimiter.

A delimiter-joined string is only an unambiguous encoding of its fields --
meaning two different sets of field values can never join to the same
string -- if none of the fields contain the delimiter itself. Otherwise a
"|" inside one field can shift where a reader thinks one field ends and the
next begins, so two genuinely different records could hash identically.
Both hash functions call check_no_delimiter on their fields before joining
them, rather than assuming real data never contains a pipe character.
"""

from __future__ import annotations

ID_DELIMITER = "|"


def check_no_delimiter(*fields: str) -> None:
    """Raise ValueError if any field contains ID_DELIMITER.

    Deliberately a raise, not an assert: assert statements are removed
    entirely when Python runs with the -O optimization flag, so an assert
    here would silently stop protecting against hash collisions in
    whatever environment happens to run with optimizations enabled. This
    check has to hold unconditionally, not just in the default case.
    """
    for field in fields:
        if ID_DELIMITER in field:
            raise ValueError(
                f"Field contains the natural-key delimiter {ID_DELIMITER!r} and cannot be "
                f"hashed unambiguously as part of a delimiter-joined key: {field!r}"
            )
