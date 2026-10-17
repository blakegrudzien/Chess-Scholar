"""Throwaway spike code for the structure discovery run: the two sources of
games. Club games stream from a Lichess monthly PGN export (read line by
line, filtered, never stored); master games stream from the project's moves
table through a server-side cursor."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import psycopg2.extensions

from scripts.spikes.fetch_games import MIN_PLIES
from scripts.spikes.structure_census import is_standard_start

MIN_ELO = 1500
MAX_ELO = 2200
MAX_RATING_GAP = 100

_COMMENT = re.compile(r"\{[^}]*\}")
_MOVE_NUMBER = re.compile(r"^\d+\.+$")
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
_BAD_TERMINATIONS = {"Abandoned", "Rules infraction"}

MASTER_QUERY = """
SELECT m.game_id, g.result, m.move_san, CASE WHEN m.ply = 1 THEN m.fen_after END
FROM moves m JOIN games g USING (game_id)
WHERE g.source = 'chessbase'
ORDER BY m.game_id, m.ply
"""


@dataclass
class SourceGame:
    result: str
    sans: list[str]
    white_elo: int | None = None
    black_elo: int | None = None


@dataclass
class ClubStreamStats:
    seen: int = 0
    qualified: int = 0


def movetext_to_sans(movetext: str) -> list[str]:
    sans = []
    for token in _COMMENT.sub(" ", movetext).split():
        if _MOVE_NUMBER.match(token) or token in _RESULTS or token.startswith("$"):
            continue
        sans.append(token.rstrip("?!"))
    return sans


def iter_pgn_games(lines: Iterable[str]) -> Iterator[tuple[dict[str, str], str]]:
    """Split a PGN stream into (headers, movetext) pairs without parsing the
    moves, which keeps filtering millions of games fast."""
    headers: dict[str, str] = {}
    movetext: list[str] = []
    for raw in lines:
        line = raw.strip()
        if line.startswith("["):
            if movetext:
                yield headers, " ".join(movetext)
                headers, movetext = {}, []
            name, _, value = line[1:-1].partition(" ")
            headers[name] = value.strip('"')
        elif line:
            movetext.append(line)
    if headers or movetext:
        yield headers, " ".join(movetext)


def qualifies(headers: dict[str, str]) -> bool:
    event = headers.get("Event", "")
    if not (event.startswith("Rated Rapid") or event.startswith("Rated Classical")):
        return False
    if headers.get("Variant", "Standard") != "Standard" or "FEN" in headers:
        return False
    if headers.get("Termination") in _BAD_TERMINATIONS:
        return False
    try:
        white, black = int(headers["WhiteElo"]), int(headers["BlackElo"])
    except (KeyError, ValueError):
        return False
    return (
        MIN_ELO <= white <= MAX_ELO
        and MIN_ELO <= black <= MAX_ELO
        and abs(white - black) <= MAX_RATING_GAP
    )


def read_club_games(
    lines: Iterable[str], limit: int, stats: ClubStreamStats
) -> Iterator[SourceGame]:
    for headers, movetext in iter_pgn_games(lines):
        stats.seen += 1
        if stats.seen % 500_000 == 0:
            print(
                f"club stream: {stats.seen} read, {stats.qualified} qualified",
                file=sys.stderr,
                flush=True,
            )
        if not qualifies(headers):
            continue
        sans = movetext_to_sans(movetext)
        if len(sans) < MIN_PLIES:
            continue
        stats.qualified += 1
        yield SourceGame(
            result=headers.get("Result", "*"),
            sans=sans,
            white_elo=int(headers["WhiteElo"]),
            black_elo=int(headers["BlackElo"]),
        )
        if stats.qualified >= limit:
            return


def group_master_rows(rows: Iterable[tuple]) -> Iterator[SourceGame]:
    """Rows of (game_id, result, move_san, ply-1 FEN or None), ordered by
    game and ply, grouped into games. Games that don't start from the
    standard position, or are too short, are skipped."""
    current_id = None
    result = ""
    sans: list[str] = []
    standard = False
    for game_id, game_result, san, ply1_fen in rows:
        if game_id != current_id:
            if current_id is not None and standard and len(sans) >= MIN_PLIES:
                yield SourceGame(result=result, sans=sans)
            current_id, result, sans = game_id, game_result, []
            standard = ply1_fen is not None and is_standard_start(ply1_fen)
        sans.append(san)
    if current_id is not None and standard and len(sans) >= MIN_PLIES:
        yield SourceGame(result=result, sans=sans)


def iter_master_games(conn: psycopg2.extensions.connection) -> Iterator[SourceGame]:
    # A named cursor streams rows from the server in batches instead of
    # loading ten million rows into memory at once.
    with conn.cursor(name="structure_discovery") as cur:
        cur.itersize = 20_000
        cur.execute(MASTER_QUERY)
        yield from group_master_rows(cur)
