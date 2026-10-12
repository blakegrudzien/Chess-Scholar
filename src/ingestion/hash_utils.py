"""Helpers for the content-hash keys (pgn_parser.compute_game_id,
db_loader._chunk_hash), which hash several fields joined by a delimiter.

The joined string only identifies its fields if no field contains the
delimiter; otherwise two different records could hash the same. Both hash
functions check their fields first instead of assuming no "|" ever appears.
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
