from scripts.spikes.structures import (
    STRUCTURES,
    first_ply_with,
    is_carlsbad,
    is_iqp_black,
    is_iqp_white,
    is_maroczy,
    minority_attack_played,
    pawn_squares,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Queen's Gambit Declined, Exchange Variation, after 5...c6.
CARLSBAD = "rnbqkb1r/pp3ppp/2p2n2/3p2B1/3P4/2N5/PP2PPPP/R2QKBNR w KQkq - 0 6"
# The same structure after White's b-pawn reaches b5.
CARLSBAD_B5 = "rnbqkb1r/pp3ppp/2p2n2/1P1p2B1/3P4/2N5/P3PPPP/R2QKBNR b KQkq - 0 10"
# White has an isolated d4 pawn; Black has no d-pawn.
IQP_WHITE = "r1bq1rk1/pp2bppp/2n1pn2/8/3P4/2NB1N2/PP3PPP/R1BQ1RK1 w - - 0 10"
# Accelerated Dragon with White's c4 and e4 pawns: the Maroczy bind.
MAROCZY = "r1bqkb1r/pp1ppp1p/2n2np1/8/2P1P3/2N5/PP1N1PPP/R1BQKB1R b KQkq - 0 6"


def test_pawn_squares_of_the_starting_position():
    white, black = pawn_squares(START)
    assert white == frozenset(f"{f}2" for f in "abcdefgh")
    assert black == frozenset(f"{f}7" for f in "abcdefgh")


def test_carlsbad():
    assert is_carlsbad(*pawn_squares(CARLSBAD))
    assert not is_carlsbad(*pawn_squares(START))


def test_isolated_queen_pawn():
    assert is_iqp_white(*pawn_squares(IQP_WHITE))
    assert not is_iqp_black(*pawn_squares(IQP_WHITE))
    assert not is_iqp_white(*pawn_squares(START))


def test_maroczy_bind():
    assert is_maroczy(*pawn_squares(MAROCZY))
    assert not is_maroczy(*pawn_squares(START))


def test_every_structure_is_registered():
    assert set(STRUCTURES) == {"carlsbad", "iqp_white", "iqp_black", "maroczy"}


def test_first_ply_with():
    assert first_ply_with([START, START, CARLSBAD, CARLSBAD], is_carlsbad) == 2
    assert first_ply_with([START, START], is_carlsbad) is None


def test_minority_attack_played_within_the_window():
    assert minority_attack_played([CARLSBAD, CARLSBAD, CARLSBAD_B5], start=0)
    assert not minority_attack_played([CARLSBAD, CARLSBAD, CARLSBAD], start=0)


def test_minority_attack_outside_the_window_does_not_count():
    fens = [CARLSBAD] * 5 + [CARLSBAD_B5]
    assert not minority_attack_played(fens, start=0, window=3)
