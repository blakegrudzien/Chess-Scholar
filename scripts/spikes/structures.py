"""Throwaway spike code for Phase 0 of the game report roadmap: simple pawn
structure checks and one plan check (the minority attack in the Carlsbad
structure). Definitions are deliberately minimal; Phase 3 replaces them with
tested, documented detectors."""

from __future__ import annotations

from collections.abc import Callable

Pawns = frozenset[str]

_FILES = "abcdefgh"


def pawn_squares(fen: str) -> tuple[Pawns, Pawns]:
    """White and Black pawn squares, read straight from the FEN's board field
    (faster than building a board, which matters over thousands of games)."""
    white: set[str] = set()
    black: set[str] = set()
    for rank_index, row in enumerate(fen.split()[0].split("/")):
        rank = 8 - rank_index
        file = 0
        for char in row:
            if char.isdigit():
                file += int(char)
                continue
            if char == "P":
                white.add(f"{_FILES[file]}{rank}")
            elif char == "p":
                black.add(f"{_FILES[file]}{rank}")
            file += 1
    return frozenset(white), frozenset(black)


def _has_pawn_on_file(pawns: Pawns, file: str) -> bool:
    return any(square[0] == file for square in pawns)


def is_carlsbad(white: Pawns, black: Pawns) -> bool:
    """White d4 with no c-pawn against Black d5 and c6 with no e-pawn, as
    after the Queen's Gambit Declined exchange."""
    return (
        "d4" in white
        and "d5" in black
        and "c6" in black
        and not _has_pawn_on_file(white, "c")
        and not _has_pawn_on_file(black, "e")
    )


def is_iqp_white(white: Pawns, black: Pawns) -> bool:
    return "d4" in white and not _has_pawn_on_file(white, "c") and not _has_pawn_on_file(white, "e")


def is_iqp_black(white: Pawns, black: Pawns) -> bool:
    return "d5" in black and not _has_pawn_on_file(black, "c") and not _has_pawn_on_file(black, "e")


def is_maroczy(white: Pawns, black: Pawns) -> bool:
    """White pawns on c4 and e4 with no d-pawn, Black without a c-pawn."""
    return (
        "c4" in white
        and "e4" in white
        and not _has_pawn_on_file(white, "d")
        and not _has_pawn_on_file(black, "c")
    )


STRUCTURES: dict[str, Callable[[Pawns, Pawns], bool]] = {
    "carlsbad": is_carlsbad,
    "iqp_white": is_iqp_white,
    "iqp_black": is_iqp_black,
    "maroczy": is_maroczy,
}


def first_ply_with(fens: list[str], predicate: Callable[[Pawns, Pawns], bool]) -> int | None:
    for index, fen in enumerate(fens):
        if predicate(*pawn_squares(fen)):
            return index
    return None


def minority_attack_played(fens: list[str], start: int, window: int = 40) -> bool:
    """Whether White's b-pawn reaches b5 while Black still has c6, within
    `window` plies of `start`: a simple marker of the minority attack."""
    for fen in fens[start : start + window + 1]:
        white, black = pawn_squares(fen)
        if "b5" in white and "c6" in black:
            return True
    return False
