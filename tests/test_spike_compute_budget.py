import pytest

from scripts.spikes.compute_budget import flag_swings, missing_eval_indexes, win_percent
from scripts.spikes.fetch_games import MATE_CP


def test_flags_positions_whose_evaluation_swings_past_the_threshold():
    evals = [20, 30, -150, -140, 400]
    assert flag_swings(evals, threshold=100) == [2, 4]


def test_mate_counts_as_a_swing_and_missing_values_are_skipped():
    evals = [20, None, 50, MATE_CP]
    assert flag_swings(evals, threshold=100) == [3]


def test_missing_eval_indexes_covers_every_ply_when_there_is_no_analysis():
    assert missing_eval_indexes(None, plies=3) == [0, 1, 2]


def test_missing_eval_indexes_includes_gaps_and_plies_past_the_analysis():
    assert missing_eval_indexes([10, None, 20], plies=5) == [1, 3, 4]


def test_win_percent_is_even_at_zero_and_saturates_at_mate():
    assert win_percent(0) == 50
    assert win_percent(MATE_CP) > 99
    assert win_percent(-MATE_CP) < 1


def test_win_percent_is_symmetric():
    assert win_percent(300) + win_percent(-300) == pytest.approx(100)


def test_swing_between_won_positions_is_not_flagged_but_a_real_swing_is():
    won_to_mate = [win_percent(900), win_percent(MATE_CP)]
    even_to_losing = [win_percent(0), win_percent(-300)]
    assert flag_swings(won_to_mate, threshold=10) == []
    assert flag_swings(even_to_losing, threshold=10) == [1]
