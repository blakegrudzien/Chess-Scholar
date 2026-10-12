"""Tooltip text for the chess notation shown on screen (FEN, ply), shared
by the chat transcript and the board panel so the wording matches.
"""

from __future__ import annotations

FEN_HELP = (
    "FEN (Forsyth-Edwards Notation): a compact text format that fully "
    "encodes a chess position: where every piece is, whose turn it is, "
    "and a few other rules-relevant details."
)

PLY_HELP = "A ply is one player's move: White's 1st move is ply 1, Black's reply is ply 2, etc."
