"""Throwaway spike for the structure discovery run: which pawn structures do
club players reach, how often do masters reach the same ones, how does each
side score in them, and how do they usually end?

Club games come from a Lichess monthly export piped through stdin and are
never stored; master games come from the moves table. The output is a
markdown report plus a JSON dump of the counts. Nothing is written to the
database.

Run from the repo root, for example:
    curl -s https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst \\
      | zstd -dc \\
      | .venv/bin/python -m scripts.spikes.structure_discovery --club-pgn - \\
          --report structure-discovery-report.md --json data/processed/structure_discovery.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from scripts.spikes.discovery_sources import (
    MAX_ELO,
    MAX_RATING_GAP,
    MIN_ELO,
    ClubStreamStats,
    iter_master_games,
    read_club_games,
)
from scripts.spikes.fetch_games import MIN_PLIES
from scripts.spikes.pawn_keys import (
    KINDS,
    MIN_STRUCTURE_PLIES,
    family_of,
    find_intervals,
    game_keys,
    owner_result,
)
from src.ingestion.db_loader import get_connection_pool

_SECTION_TITLES = {
    "both": "Both sides' central pawns",
    "white": "White's central pawns",
    "black": "Black's central pawns",
}


@dataclass
class KeyStats:
    games: int = 0  # games that reached this key for at least the minimum length
    intervals: int = 0
    plies: int = 0
    results: Counter = field(default_factory=Counter)  # owning side, once per game
    exits: Counter = field(default_factory=Counter)
    next_keys: Counter = field(default_factory=Counter)


@dataclass
class Tally:
    games: int = 0
    skipped: int = 0
    stats: dict[str, dict[str, KeyStats]] = field(
        default_factory=lambda: {kind: {} for kind in KINDS}
    )

    def add_game(self, sans: list[str], result: str) -> bool:
        try:
            keys = game_keys(sans)
        except Exception:
            # Any failure replaying the game (an illegal or unreadable move)
            # skips that game, so one bad record cannot stop a long run.
            self.skipped += 1
            return False
        self.games += 1
        for kind in KINDS:
            seen: set[str] = set()
            for interval in find_intervals(keys[kind], sans):
                stats = self.stats[kind].setdefault(interval.key, KeyStats())
                stats.intervals += 1
                stats.plies += interval.end - interval.start + 1
                if interval.exit_move is not None:
                    stats.exits[interval.exit_move] += 1
                    stats.next_keys[interval.next_key] += 1
                if interval.key not in seen:
                    seen.add(interval.key)
                    stats.games += 1
                    outcome = owner_result(result, kind)
                    if outcome is not None:
                        stats.results[outcome] += 1
        return True

    def to_json(self) -> dict:
        return {
            "games": self.games,
            "skipped": self.skipped,
            "stats": {
                kind: {
                    key: {
                        "games": s.games,
                        "intervals": s.intervals,
                        "plies": s.plies,
                        "results": dict(s.results),
                        "exits": dict(s.exits),
                        "next_keys": dict(s.next_keys),
                    }
                    for key, s in keys.items()
                }
                for kind, keys in self.stats.items()
            },
        }


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "-"


def _wdl(stats: KeyStats | None) -> str:
    if stats is None:
        return "-"
    known = sum(stats.results.values())
    if not known:
        return "-"
    return " / ".join(
        f"{stats.results[outcome] / known:.0%}" for outcome in ("win", "draw", "loss")
    )


def _top_exits(stats: KeyStats | None, n: int = 3) -> str:
    if stats is None or not stats.intervals:
        return "-"
    parts = [
        f"{move} ({count / stats.intervals:.0%})" for move, count in stats.exits.most_common(n)
    ]
    # A run with no exit move is one the game ended in.
    game_ends = stats.intervals - sum(stats.exits.values())
    if game_ends > 0:
        parts.append(f"game end ({game_ends / stats.intervals:.0%})")
    return ", ".join(parts)


def _games(stats: KeyStats | None) -> int:
    return stats.games if stats is not None else 0


def render_report(
    club: Tally,
    masters: Tally,
    notes: list[str],
    top: int = 20,
    min_master_games: int = 100,
    masters_complete: bool = True,
) -> str:
    lines = ["# Pawn structure discovery", ""]
    if not masters_complete:
        # Placed first so no verdict below is read without it.
        lines += [
            "Warning: the master tally is incomplete, so master columns and coverage "
            "are not final.",
            "",
        ]
    lines += [f"- {note}" for note in notes]
    lines += [
        f"- Club games analyzed: {club.games} (skipped {club.skipped} unreadable)",
        f"- Master games analyzed: {masters.games} (skipped {masters.skipped} unreadable)",
        f"- A key counts in a game when it lasts at least {MIN_STRUCTURE_PLIES} plies.",
        f"- Exits are shares of runs (a run is a stretch of at least {MIN_STRUCTURE_PLIES} "
        "plies with the key unchanged). A key can come back later in the same game after "
        "a change, such as a capture and a recapture, so one game can add more than one "
        "run. Games and W/D/L still count each game once per key.",
        "- W/D/L is for the side that owns the key (White for White and both-sides "
        "keys, Black for Black keys), as a share of games with a known result.",
        f"- Coverage is 'thin' when fewer than {min_master_games} master games reach the key.",
        "",
    ]
    if not club.games:
        lines.append("No club games were analyzed, so there is nothing to rank.")
        return "\n".join(lines) + "\n"

    for kind in ("both", "white", "black"):
        lines += [f"## {_SECTION_TITLES[kind]}", ""]
        lines.append(
            "| Key | Club games | Club W/D/L | Master games | Master W/D/L | Coverage "
            "| Club exits (% of runs) | Master exits (% of runs) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        ranked = sorted(club.stats[kind].items(), key=lambda item: -item[1].games)[:top]
        for key, club_stats in ranked:
            master_stats = masters.stats[kind].get(key)
            if not masters_complete:
                coverage = "incomplete"
            elif _games(master_stats) < min_master_games:
                coverage = "thin"
            else:
                coverage = ""
            lines.append(
                f"| {key} | {club_stats.games} ({_pct(club_stats.games, club.games)}) "
                f"| {_wdl(club_stats)} "
                f"| {_games(master_stats)} ({_pct(_games(master_stats), masters.games)}) "
                f"| {_wdl(master_stats)} | {coverage} "
                f"| {_top_exits(club_stats)} | {_top_exits(master_stats)} |"
            )
        lines.append("")

    lines += ["## Full-board variants of the top both-sides keys", ""]
    top_families = sorted(club.stats["both"].items(), key=lambda item: -item[1].games)[:5]
    for family, _ in top_families:
        lines += [f"### {family}", "", "| Full key | Club games | Master games |", "|---|---|---|"]
        variants = [
            (key, stats) for key, stats in club.stats["full"].items() if family_of(key) == family
        ]
        for key, stats in sorted(variants, key=lambda item: -item[1].games)[:5]:
            lines.append(f"| {key} | {stats.games} | {_games(masters.stats['full'].get(key))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _read_club(stream: TextIO, limit: int, notes: list[str]) -> Tally:
    club = Tally()
    stats = ClubStreamStats()
    for game in read_club_games(stream, limit, stats):
        club.add_game(game.sans, game.result)
    # The stream ending early (a dropped download, a short file) looks the same
    # as a finished run unless the note says which of the two happened.
    if stats.qualified >= limit:
        ending = f"stopped at the {limit}-game limit"
    else:
        ending = f"the stream ended before the {limit}-game limit"
    note = (
        f"Club sample: {stats.qualified} qualifying games out of {stats.seen} read "
        f"(rapid and classical, both players {MIN_ELO} to {MAX_ELO}, "
        f"rating gap at most {MAX_RATING_GAP}); {ending}."
    )
    notes.append(note)
    print(note, file=sys.stderr, flush=True)
    return club


def _read_masters(pool, notes: list[str]) -> tuple[Tally, bool]:
    """Tally the master games. The second value is False when the read failed
    part way, so the report can say its master columns are not final."""
    masters = Tally()
    complete = False
    count = 0
    conn = None
    try:
        conn = pool.getconn()
        for count, game in enumerate(iter_master_games(conn), start=1):
            masters.add_game(game.sans, game.result)
            if count % 10_000 == 0:
                print(f"master games: {count}", file=sys.stderr, flush=True)
        complete = True
        notes.append(
            "Master sample: every standard-start ChessBase game of at least "
            f"{MIN_PLIES} plies in the database."
        )
    except Exception as exc:
        # Keep the games already tallied; a failure this late should not
        # discard the club run or the master games read before it.
        note = f"Master tally incomplete after {count} games: {type(exc).__name__}: {exc}"
        notes.append(note)
        print(note, file=sys.stderr, flush=True)
    finally:
        if conn is not None:
            pool.putconn(conn)
    return masters, complete


def main() -> None:
    parser = argparse.ArgumentParser(description="Pawn structure discovery spike")
    parser.add_argument("--club-pgn", help="Lichess PGN export to read; '-' reads stdin")
    parser.add_argument("--club-games", type=int, default=200_000)
    parser.add_argument("--skip-masters", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    notes: list[str] = []
    pool = None if args.skip_masters else get_connection_pool()
    try:
        if pool is not None:
            # Check the database before the club stream is read, so a missing
            # DATABASE_URL or an unreachable database fails now rather than
            # hours later. The working connection is taken after the club read,
            # so the pool can health-check it first; a connection held through
            # the whole read would be used unchecked.
            pool.putconn(pool.getconn())

        club = Tally()
        if args.club_pgn == "-":
            # Undecodable bytes become replacement characters instead of
            # stopping a long run.
            stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
            try:
                club = _read_club(stdin, args.club_games, notes)
            finally:
                # Closing the pipe lets the upstream curl and zstd processes
                # exit now instead of waiting through the master pass.
                stdin.close()
        elif args.club_pgn:
            with open(args.club_pgn, encoding="utf-8", errors="replace") as stream:
                club = _read_club(stream, args.club_games, notes)

        masters_complete = True
        if pool is None:
            masters = Tally()
            notes.append("Master games were skipped in this run, so master columns are empty.")
        else:
            masters, masters_complete = _read_masters(pool, notes)
    finally:
        if pool is not None:
            pool.closeall()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"club": club.to_json(), "masters": masters.to_json(), "notes": notes}
        # Dumping straight into the file avoids building the whole JSON text
        # as one large string in memory first.
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    report = render_report(club, masters, notes, masters_complete=masters_complete)
    args.report.write_text(report, encoding="utf-8")
    print(f"Wrote {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
