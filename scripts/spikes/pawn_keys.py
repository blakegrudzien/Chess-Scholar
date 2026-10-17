"""Throwaway spike code for the structure discovery run: turn positions into
pawn keys and find where a key stays unchanged long enough to count as a
structure. No structure names are assumed; names are attached afterwards by
looking at the most common keys."""

from __future__ import annotations

from dataclasses import dataclass

import chess

# A key must stay unchanged this many plies to count as a structure, so
# passing through a pawn formation for a move or two is ignored.
MIN_STRUCTURE_PLIES = 10

KINDS = ("white", "black", "both", "full")

CENTRAL_FILES = "cdef"
_CENTRAL_MASK = chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F


@dataclass(frozen=True)
class PositionKeys:
    white: str  # White's central pawns, e.g. "W:d4"
    black: str  # Black's central pawns, e.g. "B:e6"
    both: str  # both sides' central pawns, e.g. "W:d4 / B:e6"
    full: str  # every pawn on the board


@dataclass(frozen=True)
class Interval:
    key: str
    start: int  # index into the per-ply lists; index 0 is the position after ply 1
    end: int  # inclusive
    # The move that changed the key, e.g. "W b5"; None if the game ended first.
    exit_move: str | None
    next_key: str | None


def _squares(bitboard: int) -> str:
    """Square names ordered by file, then rank (c2 d2 e4 f2), so keys read
    left to right across the board."""
    squares = sorted(
        chess.SquareSet(bitboard), key=lambda sq: (chess.square_file(sq), chess.square_rank(sq))
    )
    return " ".join(chess.square_name(sq) for sq in squares) or "-"


def position_keys(board: chess.Board) -> PositionKeys:
    white = board.pawns & board.occupied_co[chess.WHITE]
    black = board.pawns & board.occupied_co[chess.BLACK]
    white_central = _squares(white & _CENTRAL_MASK)
    black_central = _squares(black & _CENTRAL_MASK)
    return PositionKeys(
        white=f"W:{white_central}",
        black=f"B:{black_central}",
        both=f"W:{white_central} / B:{black_central}",
        full=f"W:{_squares(white)} / B:{_squares(black)}",
    )


def family_of(full_key: str) -> str:
    """The central both-sides key contained in a full-board key."""

    def central(part: str) -> str:
        prefix, squares = part.split(":", 1)
        kept = [sq for sq in squares.split() if sq[0] in CENTRAL_FILES]
        return f"{prefix}:{' '.join(kept) or '-'}"

    white_part, black_part = full_key.split(" / ")
    return f"{central(white_part)} / {central(black_part)}"


def game_keys(sans: list[str]) -> dict[str, list[str]]:
    """Every kind of key for the position after each ply. Raises ValueError
    on an illegal or unreadable move."""
    board = chess.Board()
    keys: dict[str, list[str]] = {kind: [] for kind in KINDS}
    for san in sans:
        board.push_san(san)
        position = position_keys(board)
        for kind in KINDS:
            keys[kind].append(getattr(position, kind))
    return keys


def find_intervals(
    keys: list[str], sans: list[str], min_plies: int = MIN_STRUCTURE_PLIES
) -> list[Interval]:
    """Runs of an unchanged key lasting at least `min_plies` plies. keys[i]
    is the key after sans[i], so a run ending at index i - 1 was ended by
    sans[i]; even indexes are White's moves."""
    intervals = []
    start = 0
    for i in range(1, len(keys) + 1):
        if i < len(keys) and keys[i] == keys[start]:
            continue
        if i - start >= min_plies:
            if i < len(keys):
                mover = "W" if i % 2 == 0 else "B"
                # Check and mate marks are dropped so "exd5" and "exd5+" count
                # as the same exit.
                exit_move, next_key = f"{mover} {sans[i].rstrip('+#')}", keys[i]
            else:
                exit_move = next_key = None
            intervals.append(Interval(keys[start], start, i - 1, exit_move, next_key))
        start = i
    return intervals


def owner_result(result: str, kind: str) -> str | None:
    """The game's outcome for the side that owns a key: Black for "black"
    keys, White otherwise. None when the result is unknown."""
    outcomes = {"1-0": ("win", "loss"), "0-1": ("loss", "win"), "1/2-1/2": ("draw", "draw")}
    if result not in outcomes:
        return None
    white_outcome, black_outcome = outcomes[result]
    return black_outcome if kind == "black" else white_outcome
