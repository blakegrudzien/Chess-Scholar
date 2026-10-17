"""Throwaway spike for Phase 0 of the game report roadmap: do a player's
games reach common pawn structures often enough to say anything, and does a
simple plan check show a difference from strong players?

Compares the player's games with a random sample of standard-start master games
from the games and moves tables (moves and positions only).

Run from the repo root:
    .venv/bin/python -m scripts.spikes.structure_census --site lichess --username NAME
"""

from __future__ import annotations

import argparse

import chess
import psycopg2.extensions

from scripts.spikes.fetch_games import MIN_PLIES, fens_after_each_ply, fetch_games
from scripts.spikes.structures import (
    STRUCTURES,
    first_ply_with,
    is_carlsbad,
    minority_attack_played,
)
from src.ingestion.db_loader import get_connection_pool, query_with_retry

# Both sides are compared only over the first MAX_PLY plies.
MAX_PLY = 80


def structure_counts(games_fens: list[list[str]]) -> dict[str, int]:
    return {
        name: sum(1 for fens in games_fens if first_ply_with(fens, predicate) is not None)
        for name, predicate in STRUCTURES.items()
    }


def minority_attack_rate(games_fens: list[list[str]]) -> tuple[int, int]:
    reached = played = 0
    for fens in games_fens:
        start = first_ply_with(fens, is_carlsbad)
        if start is None:
            continue
        reached += 1
        if minority_attack_played(fens, start):
            played += 1
    return reached, played


def format_rate(part: int, whole: int) -> str:
    if whole == 0:
        return "0 of 0"
    return f"{part} of {whole} ({part / whole:.0%})"


def is_standard_start(first_fen: str) -> bool:
    """True when the position after ply 1 could come from the standard start:
    Black to move, fullmove number 1, and all 32 pieces still on the board."""
    board_field, side, _castling, _ep, _half, fullmove = first_fen.split()
    pieces = sum(1 for ch in board_field if ch.isalpha())
    return side == "b" and fullmove == "1" and pieces == 32


def load_master_fens(
    conn: psycopg2.extensions.connection, sample_size: int, max_ply: int = MAX_PLY
) -> list[list[str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT setseed(0.42)")
        cur.execute(
            "SELECT game_id FROM games WHERE source = 'chessbase' ORDER BY random() LIMIT %s",
            (sample_size,),
        )
        game_ids = [row[0] for row in cur.fetchall()]
        cur.execute(
            "SELECT game_id, fen_after FROM moves "
            "WHERE game_id = ANY(%s) AND ply <= %s ORDER BY game_id, ply",
            (game_ids, max_ply),
        )
        rows = cur.fetchall()
    by_game: dict[str, list[str]] = {}
    for game_id, fen in rows:
        by_game.setdefault(game_id, []).append(fen)
    # Drop games too short to say anything, matching the player's games, and
    # games that start from a set-up position, since the player's games are
    # all standard chess. Rows are ordered by ply, so the first FEN is ply 1.
    return [
        fens for fens in by_game.values() if len(fens) >= MIN_PLIES and is_standard_start(fens[0])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Game report structure census spike")
    parser.add_argument("--site", choices=["lichess", "chesscom"], required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--time-control", default="blitz")
    parser.add_argument("--baseline-games", type=int, default=5000)
    args = parser.parse_args()

    games = fetch_games(args.site, args.username, args.games, args.time_control)
    user_fens = [fens_after_each_ply(g.sans)[:MAX_PLY] for g in games]
    user_white_fens = [
        fens for g, fens in zip(games, user_fens, strict=True) if g.user_color == chess.WHITE
    ]

    pool = get_connection_pool()
    try:
        master_fens = query_with_retry(pool, load_master_fens, args.baseline_games)
    finally:
        pool.closeall()

    user_counts = structure_counts(user_fens)
    master_counts = structure_counts(master_fens)
    print(f"{len(user_fens)} of your games, {len(master_fens)} master games\n")
    print("| Structure | Your games reaching it | Master games reaching it |")
    print("|---|---|---|")
    for name in STRUCTURES:
        print(
            f"| {name} | {format_rate(user_counts[name], len(user_fens))} | "
            f"{format_rate(master_counts[name], len(master_fens))} |"
        )

    user_reached, user_played = minority_attack_rate(user_white_fens)
    master_reached, master_played = minority_attack_rate(master_fens)
    print("\nMinority attack in Carlsbad structures (as White):")
    print(f"  You: {format_rate(user_played, user_reached)}")
    print(f"  Masters: {format_rate(master_played, master_reached)}")

    print(
        "\nStructures are board patterns, counted over all games of both colors "
        "on both sides. The minority attack compares White's play in Carlsbad "
        "structures."
    )
    print(
        "The master baseline is a random sample of standard-start master games, "
        "not matched by opening or era."
    )


if __name__ == "__main__":
    main()
