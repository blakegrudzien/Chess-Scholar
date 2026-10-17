import chess
import pytest

from scripts.spikes.pawn_keys import (
    Interval,
    family_of,
    find_intervals,
    game_keys,
    owner_result,
    position_keys,
)

# Knight moves only: the pawns never move, so every position has the same keys.
KNIGHT_SHUFFLE = [
    "Nf3",
    "Nf6",
    "Nc3",
    "Nc6",
    "Ng1",
    "Ng8",
    "Nb1",
    "Nb8",
    "Nf3",
    "Nf6",
    "Ng1",
    "Ng8",
]


def test_keys_of_the_starting_position():
    keys = position_keys(chess.Board())
    assert keys.white == "W:c2 d2 e2 f2"
    assert keys.black == "B:c7 d7 e7 f7"
    assert keys.both == "W:c2 d2 e2 f2 / B:c7 d7 e7 f7"
    assert keys.full == "W:a2 b2 c2 d2 e2 f2 g2 h2 / B:a7 b7 c7 d7 e7 f7 g7 h7"
    assert "|" not in keys.both + keys.full


def test_side_without_central_pawns_gets_a_dash():
    board = chess.Board("4k3/pp4pp/8/8/8/8/PP4PP/4K3 w - - 0 1")
    keys = position_keys(board)
    assert keys.white == "W:-"
    assert keys.both == "W:- / B:-"


def test_family_of_keeps_only_central_files():
    full = "W:a2 b2 d4 g2 h2 / B:a7 b7 e6 g7 h7"
    assert family_of(full) == "W:d4 / B:e6"
    assert family_of("W:a2 / B:h7") == "W:- / B:-"


def test_game_keys_has_one_key_per_ply_for_each_kind():
    keys = game_keys(["e4", "e5"])
    assert set(keys) == {"white", "black", "both", "full"}
    assert keys["white"] == ["W:c2 d2 e4 f2", "W:c2 d2 e4 f2"]
    assert keys["black"] == ["B:c7 d7 e7 f7", "B:c7 d7 e5 f7"]


def test_game_keys_rejects_an_illegal_move():
    with pytest.raises(ValueError):
        game_keys(["e4", "e4"])


def test_interval_needs_at_least_the_minimum_length():
    sans = [f"m{i}" for i in range(13)]
    keys = ["A"] * 10 + ["B"] * 3
    assert find_intervals(keys, sans) == [Interval("A", 0, 9, "W m10", "B")]
    assert find_intervals(["A"] * 9 + ["B"] * 4, sans) == []


def test_exit_move_records_which_side_moved():
    sans = [f"m{i}" for i in range(12)]
    keys = ["A"] + ["B"] * 10 + ["C"]
    # Index 11 is ply 12, a Black move.
    assert find_intervals(keys, sans) == [Interval("B", 1, 10, "B m11", "C")]


@pytest.mark.parametrize("suffix", ["+", "#"])
def test_exit_label_ignores_check_and_mate_suffixes(suffix):
    sans = [f"m{i}" for i in range(10)] + [f"m10{suffix}", "m11"]
    keys = ["A"] * 10 + ["B"] * 2
    # The same capture with and without a check mark must count as one exit.
    assert find_intervals(keys, sans) == [Interval("A", 0, 9, "W m10", "B")]


def test_structure_on_the_board_at_the_end_has_no_exit():
    sans = [f"m{i}" for i in range(16)]
    keys = ["A"] * 4 + ["B"] * 12
    assert find_intervals(keys, sans) == [Interval("B", 4, 15, None, None)]


def test_knight_shuffle_is_one_interval_ended_by_the_pawn_move():
    sans = KNIGHT_SHUFFLE + ["e4"]
    keys = game_keys(sans)["white"]
    assert find_intervals(keys, sans) == [Interval("W:c2 d2 e2 f2", 0, 11, "W e4", "W:c2 d2 e4 f2")]


def test_owner_result():
    assert owner_result("1-0", "white") == "win"
    assert owner_result("1-0", "both") == "win"
    assert owner_result("1-0", "full") == "win"
    assert owner_result("1-0", "black") == "loss"
    assert owner_result("0-1", "black") == "win"
    assert owner_result("1/2-1/2", "black") == "draw"
    assert owner_result("*", "white") is None
