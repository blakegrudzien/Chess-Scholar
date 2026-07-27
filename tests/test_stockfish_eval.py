import shutil

import chess
import chess.engine
import pytest

from src.engine.stockfish_eval import (
    PositionEval,
    classify_move,
    evaluate_game,
    evaluate_position,
)

STOCKFISH_PATH = shutil.which("stockfish")
pytestmark = pytest.mark.skipif(STOCKFISH_PATH is None, reason="requires a local Stockfish binary")


@pytest.fixture(scope="module")
def engine():
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as eng:
        yield eng


def test_evaluate_starting_position_is_roughly_balanced(engine):
    result = evaluate_position(engine, chess.Board(), depth=12)
    assert result.mate_in is None
    assert result.score_cp is not None
    assert -100 < result.score_cp < 100  # near-equal, small white edge expected
    assert result.best_move_san
    assert result.pv_san


def test_evaluate_position_finds_back_rank_mate_in_one(engine):
    # White king g1, pawns f2/g2/h2; white rook e1; black king g8, pawns
    # f7/g7/h7 boxed in. Re8# is mate: no escape square, nothing captures e8.
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    result = evaluate_position(engine, board, depth=10)
    assert result.mate_in == 1
    assert result.best_move_san == "Re8#"


def test_evaluate_game_returns_one_eval_per_move(engine):
    moves = ["e4", "e5", "Nf3", "Nc6"]
    evals = evaluate_game(engine, moves, depth=8)
    assert len(evals) == len(moves)
    assert all(isinstance(e, PositionEval) for e in evals)
    # Position after each move should reflect that move being applied.
    assert evals[0].fen.startswith("rnbqkbnr/pppppppp/8/8/4P3")


def _eval(*, score_cp=None, mate_in=None):
    return PositionEval(fen="", score_cp=score_cp, mate_in=mate_in, best_move_san="", pv_san=[])


def test_classify_move_sound_when_eval_barely_changes():
    before = _eval(score_cp=20)
    after = _eval(score_cp=-10)  # mover's POV after negation: +10 -- 10cp loss
    assert classify_move(before, after) == "sound"


def test_classify_move_inaccuracy_at_threshold():
    before = _eval(score_cp=0)
    after = _eval(score_cp=50)  # mover's POV: -50 -- exactly 50cp loss
    assert classify_move(before, after) == "inaccuracy"


def test_classify_move_mistake_at_threshold():
    before = _eval(score_cp=0)
    after = _eval(score_cp=100)  # mover's POV: -100 -- exactly 100cp loss
    assert classify_move(before, after) == "mistake"


def test_classify_move_blunder_at_threshold():
    before = _eval(score_cp=0)
    after = _eval(score_cp=200)  # mover's POV: -200 -- exactly 200cp loss
    assert classify_move(before, after) == "blunder"


def test_classify_move_blunder_into_forced_mate():
    before = _eval(score_cp=50)
    after = _eval(mate_in=3)  # mover's POV after negation: getting mated
    assert classify_move(before, after) == "blunder"


def test_classify_move_sound_when_delivering_mate():
    before = _eval(score_cp=500)
    after = _eval(mate_in=-1)  # mover's POV after negation: mate_in=1 for mover
    assert classify_move(before, after) == "sound"
