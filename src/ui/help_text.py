"""Shared tooltip copy for the chess notation this UI shows on screen.

Both the chat transcript's position captions and the board panel's own
FEN/ply captions surface real notation terms that mean nothing to a reader
who doesn't already play. st.caption's help= renders these as a hover
tooltip, which keeps the caption itself short and glanceable instead of
spelling the term out inline every time it appears.

They live here rather than in either module that uses them so the wording
stays identical in both places, and so neither module has to import the
other just to reuse a string.
"""

from __future__ import annotations

FEN_HELP = (
    "FEN (Forsyth-Edwards Notation): a compact text format that fully "
    "encodes a chess position -- where every piece is, whose turn it is, "
    "and a few other rules-relevant details."
)

PLY_HELP = "A ply is one player's move -- White's 1st move is ply 1, Black's reply is ply 2, etc."
