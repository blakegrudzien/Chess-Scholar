from scripts.spikes.structure_census import (
    format_rate,
    is_standard_start,
    minority_attack_rate,
    structure_counts,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CARLSBAD = "rnbqkb1r/pp3ppp/2p2n2/3p2B1/3P4/2N5/PP2PPPP/R2QKBNR w KQkq - 0 6"
CARLSBAD_B5 = "rnbqkb1r/pp3ppp/2p2n2/1P1p2B1/3P4/2N5/P3PPPP/R2QKBNR b KQkq - 0 10"


def test_structure_counts_counts_each_game_once():
    games = [[START, CARLSBAD, CARLSBAD], [START, START]]
    counts = structure_counts(games)
    assert counts["carlsbad"] == 1
    assert counts["maroczy"] == 0


def test_minority_attack_rate():
    games = [[CARLSBAD, CARLSBAD_B5], [CARLSBAD, CARLSBAD], [START]]
    assert minority_attack_rate(games) == (2, 1)


def test_format_rate_handles_zero_games():
    assert format_rate(0, 0) == "0 of 0"
    assert format_rate(1, 4) == "1 of 4 (25%)"


def test_standard_start_accepts_the_position_after_one_e4():
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert is_standard_start(after_e4)


def test_standard_start_rejects_a_middlegame_position():
    assert not is_standard_start(CARLSBAD)


def test_standard_start_rejects_a_handicap_position_at_move_one():
    missing_queen = "rnb1kbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert not is_standard_start(missing_queen)
