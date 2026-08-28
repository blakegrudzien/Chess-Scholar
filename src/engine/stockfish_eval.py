"""Layer 3 -- Stockfish ground-truth evaluation via python-chess's UCI
integration. Used to keep LLM commentary grounded (e.g. checking whether a
human annotator's claim like "brilliant move" matches the engine's verdict,
or scoring moves for annotated PGN export) instead of letting the agent
judge tactical soundness on its own.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import chess
import chess.engine
from dotenv import load_dotenv

DEFAULT_DEPTH = 18

# Centipawn-loss thresholds for classify_move, from the mover's perspective.
BLUNDER_THRESHOLD = 200
MISTAKE_THRESHOLD = 100
INACCURACY_THRESHOLD = 50

# Finite stand-in for a forced mate so mate and centipawn scores compare on
# one scale; larger than any real middlegame centipawn evaluation.
MATE_SCORE_CP = 10000


@dataclass
class PositionEval:
    fen: str
    score_cp: int | None  # from the side-to-move's perspective; None if mate_in is set
    mate_in: int | None  # moves to mate, signed from the side-to-move's perspective
    best_move_san: str
    pv_san: list[str]


def get_engine_path() -> str:
    # override=True: .env must win over anything already in os.environ.
    # Streamlit auto-loads .streamlit/secrets.toml into the environment
    # before app.py runs; that file exists only as a staging copy for
    # pasting into Streamlit Cloud's dashboard, but Streamlit doesn't know
    # that and injects it locally too. Without override=True, load_dotenv()
    # leaves that value in place instead of using .env's local one -- e.g.
    # STOCKFISH_PATH ends up as the deployed container's Linux path
    # (/usr/games/stockfish) even on a Mac. No effect in the actual
    # deployment, where .env doesn't exist and there's nothing to override.
    load_dotenv(override=True)
    return os.environ["STOCKFISH_PATH"]


def evaluate_position(
    engine: chess.engine.SimpleEngine, board: chess.Board, depth: int = DEFAULT_DEPTH
) -> PositionEval:
    """Evaluate `board` from the side-to-move's perspective."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].pov(board.turn)

    pv_board = board.copy()
    pv_san = []
    for move in info.get("pv", []):
        pv_san.append(pv_board.san(move))
        pv_board.push(move)
    best_move_san = pv_san[0] if pv_san else ""

    if score.is_mate():
        return PositionEval(
            fen=board.fen(),
            score_cp=None,
            mate_in=score.mate(),
            best_move_san=best_move_san,
            pv_san=pv_san,
        )
    return PositionEval(
        fen=board.fen(),
        score_cp=score.score(),
        mate_in=None,
        best_move_san=best_move_san,
        pv_san=pv_san,
    )


def evaluate_game(
    engine: chess.engine.SimpleEngine, move_sans: list[str], depth: int = DEFAULT_DEPTH
) -> list[PositionEval]:
    """Evaluate the position after each move in `move_sans`, in order."""
    board = chess.Board()
    evals = []
    for san in move_sans:
        board.push_san(san)
        evals.append(evaluate_position(engine, board, depth=depth))
    return evals


def _to_comparable_cp(position_eval: PositionEval) -> int:
    if position_eval.mate_in is not None:
        return MATE_SCORE_CP if position_eval.mate_in > 0 else -MATE_SCORE_CP
    return position_eval.score_cp


def classify_move(eval_before: PositionEval, eval_after: PositionEval) -> str:
    """Classify a move by centipawn loss, from the mover's perspective.

    `eval_before` must be evaluated before the move (mover to move), and
    `eval_after` after it (opponent to move) -- eval_after is negated since
    PositionEval scores are always from the side-to-move's own perspective.
    """
    before_cp = _to_comparable_cp(eval_before)
    after_cp = -_to_comparable_cp(eval_after)
    loss = before_cp - after_cp

    if loss >= BLUNDER_THRESHOLD:
        return "blunder"
    if loss >= MISTAKE_THRESHOLD:
        return "mistake"
    if loss >= INACCURACY_THRESHOLD:
        return "inaccuracy"
    return "sound"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FEN position with Stockfish.")
    parser.add_argument("--fen", required=True, help="Position to evaluate")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Search depth")
    args = parser.parse_args()

    board = chess.Board(args.fen)
    with chess.engine.SimpleEngine.popen_uci(get_engine_path()) as engine:
        result = evaluate_position(engine, board, depth=args.depth)

    if result.mate_in is not None:
        print(f"Mate in {result.mate_in}")
    else:
        print(f"Score: {result.score_cp} centipawns")
    print(f"Best move: {result.best_move_san}")
    print(f"PV: {' '.join(result.pv_san)}")


if __name__ == "__main__":
    _main()
