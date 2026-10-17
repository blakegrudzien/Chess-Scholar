"""Throwaway spike for Phase 0 of the game report roadmap: how long does
analyzing a player's games take?

Measures, on one engine thread:
- how many positions Lichess has already evaluated (reusable for free),
- a shallow pass over every position Lichess did not evaluate,
- deep-pass timings on a cold sample of the positions the shallow pass flags
  (swings are measured in win-percentage points, using Lichess's formula),
- deep-pass timings on whole games analyzed warm and in order, which is how
  a full deep run would actually go,
and extrapolates the total for the shallow-then-deep design against deep
analysis of everything.

Run from the repo root:
    .venv/bin/python -m scripts.spikes.compute_budget --site lichess --username NAME
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time

import chess
import chess.engine

from scripts.spikes.fetch_games import MATE_CP, fens_after_each_ply, fetch_games
from src.engine.stockfish_eval import get_engine_path


def win_percent(cp: int) -> float:
    """White's winning chances in percent for a centipawn score, using the
    logistic curve Lichess publishes. A swing of 300 centipawns matters in an
    equal position and hardly at all in a won one; this scale reflects that."""
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def flag_swings(white_evals: list[float | None], threshold: float) -> list[int]:
    """Indexes of positions whose evaluation moved at least `threshold`
    (in whatever unit the values use) from the previous known evaluation."""
    flagged = []
    previous = None
    for index, value in enumerate(white_evals):
        if value is None:
            continue
        if previous is not None and abs(value - previous) >= threshold:
            flagged.append(index)
        previous = value
    return flagged


def missing_eval_indexes(white_evals: list[int | None] | None, plies: int) -> list[int]:
    """Indexes of plies that have no usable evaluation: every ply when there
    is no analysis, otherwise gaps (None) and plies past the end of the analysis."""
    if white_evals is None:
        return list(range(plies))
    return [
        index for index in range(plies) if index >= len(white_evals) or white_evals[index] is None
    ]


def _white_cp(
    engine: chess.engine.SimpleEngine, fen: str, depth: int, game: object | None = None
) -> int:
    # python-chess sends ucinewgame whenever the `game` object changes, which
    # clears the transposition table. Passing a new object per position makes
    # each analysis cold; passing one object per game keeps the hash warm.
    info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=depth), game=game)
    return info["score"].white().score(mate_score=MATE_CP)


def main() -> None:
    parser = argparse.ArgumentParser(description="Game report compute budget spike")
    parser.add_argument("--site", choices=["lichess", "chesscom"], required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--time-control", default="blitz")
    parser.add_argument("--shallow-depth", type=int, default=10)
    parser.add_argument("--deep-depths", type=int, nargs="+", default=[16, 20])
    parser.add_argument(
        "--swing",
        type=float,
        default=10,
        help="flag a position when White's win percentage moves this many points",
    )
    parser.add_argument("--sample", type=int, default=150, help="flagged positions to time")
    parser.add_argument(
        "--warm-games", type=int, default=2, help="games to analyze warm and in order"
    )
    args = parser.parse_args()

    games = fetch_games(args.site, args.username, args.games, args.time_control)
    total_positions = sum(len(g.sans) for g in games)
    lichess_covered = [g for g in games if any(v is not None for v in (g.white_evals or []))]
    lichess_positions = sum(
        sum(1 for v in (g.white_evals or []) if v is not None) for g in lichess_covered
    )
    print("Engine: Threads=1, Hash=64 MB")
    print(f"{len(games)} games, {total_positions} positions")
    print(
        f"Already evaluated by Lichess: {len(lichess_covered)} games, {lichess_positions} positions"
    )

    engine = chess.engine.SimpleEngine.popen_uci(get_engine_path())
    engine.configure({"Threads": 1, "Hash": 64})
    try:
        game_fens: list[list[str]] = []
        flagged_fens: list[str] = []
        shallow_positions = 0
        shallow_started = time.perf_counter()
        for game in games:
            fens = fens_after_each_ply(game.sans)
            game_fens.append(fens)
            # Lichess evals are reused only where present. Every other ply gets
            # a shallow eval. The shallow pass runs sequentially with a warm
            # hash, as production would analyze a game.
            evals: list[int | None] = [
                game.white_evals[i] if game.white_evals and i < len(game.white_evals) else None
                for i in range(len(fens))
            ]
            for index in missing_eval_indexes(game.white_evals, len(fens)):
                evals[index] = _white_cp(engine, fens[index], args.shallow_depth)
                shallow_positions += 1
            win_evals = [None if v is None else win_percent(v) for v in evals]
            flagged_fens += [fens[i] for i in flag_swings(win_evals, args.swing)]
        shallow_seconds = time.perf_counter() - shallow_started
        print(
            f"Shallow pass (depth {args.shallow_depth}): {shallow_positions} positions "
            f"in {shallow_seconds:.1f}s"
        )
        print(f"Flagged for the deep pass: {len(flagged_fens)} positions")

        # Seeded sampling for a timing estimate, not a security context, so S311 does not apply.
        rng = random.Random(0)  # noqa: S311
        flagged_sample = rng.sample(flagged_fens, min(args.sample, len(flagged_fens)))
        warm_games = rng.sample(game_fens, min(args.warm_games, len(game_fens)))
        print()
        print(
            "| Deep depth | flagged s/pos mean (median, p90) | "
            "sequential s/pos mean (median, p90) | Deep everything | Shallow + deep flagged |"
        )
        print("|---|---|---|---|---|")
        for depth in args.deep_depths:
            flagged_times = _timings(engine, [flagged_sample], depth, fresh=True)
            sequential_times = _timings(engine, warm_games, depth, fresh=False)
            flagged_mean = _mean(flagged_times)
            everything = total_positions * _mean(sequential_times)
            planned = shallow_seconds + len(flagged_fens) * flagged_mean
            print(
                f"| {depth} | {_describe(flagged_times)} | {_describe(sequential_times)} | "
                f"{everything / 60:.1f} min | {planned / 60:.1f} min |"
            )
        print()
        print("Deep everything analyzes every position and does not reuse Lichess evaluations.")
    finally:
        engine.quit()


def _timings(
    engine: chess.engine.SimpleEngine, fen_runs: list[list[str]], depth: int, fresh: bool
) -> list[float]:
    """Seconds per position. Each inner list is analyzed in order. With
    fresh=False the engine keeps its hash across positions, as in a real
    sequential run; with fresh=True every position starts cold."""
    timings = []
    for fens in fen_runs:
        run = object()
        for fen in fens:
            started = time.perf_counter()
            _white_cp(engine, fen, depth, game=object() if fresh else run)
            timings.append(time.perf_counter() - started)
    return timings


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _describe(values: list[float]) -> str:
    if len(values) < 2:
        return f"{_mean(values):.3f}"
    p90 = statistics.quantiles(values, n=10)[-1]
    return f"{statistics.mean(values):.3f} ({statistics.median(values):.3f}, {p90:.3f})"


if __name__ == "__main__":
    main()
