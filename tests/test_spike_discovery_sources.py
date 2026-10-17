from scripts.spikes.discovery_sources import (
    ClubStreamStats,
    group_master_rows,
    iter_pgn_games,
    movetext_to_sans,
    qualifies,
    read_club_games,
)

AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
MIDGAME = "rnbqkb1r/pp3ppp/2p2n2/3p2B1/3P4/2N5/PP2PPPP/R2QKBNR w KQkq - 0 6"
TWELVE = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5"


def _headers(**overrides):
    headers = {
        "Event": "Rated Rapid game",
        "WhiteElo": "1800",
        "BlackElo": "1750",
        "Result": "1-0",
        "Termination": "Normal",
    }
    headers.update(overrides)
    return headers


def _pgn(headers, movetext):
    lines = [f'[{name} "{value}"]' for name, value in headers.items()]
    return lines + ["", movetext, ""]


def test_movetext_to_sans_strips_numbers_comments_annotations_and_result():
    movetext = "1. e4 { [%clk 0:10:00] } 1... e5?! { [%eval 0.3] } 2. Nf3! Nc6 $1 1-0"
    assert movetext_to_sans(movetext) == ["e4", "e5", "Nf3", "Nc6"]


def test_iter_pgn_games_splits_games():
    lines = _pgn(_headers(), TWELVE) + _pgn(_headers(Result="0-1"), "1. d4 d5 0-1")
    games = list(iter_pgn_games(lines))
    assert len(games) == 2
    assert games[0][0]["Event"] == "Rated Rapid game"
    assert games[1][0]["Result"] == "0-1"
    assert games[1][1] == "1. d4 d5 0-1"


def test_qualifies_accepts_rapid_classical_and_tournaments():
    assert qualifies(_headers())
    assert qualifies(_headers(Event="Rated Classical game"))
    assert qualifies(_headers(Event="Rated Rapid tournament https://lichess.org/tournament/x"))


def test_qualifies_rejects_other_speeds_variants_and_bad_terminations():
    assert not qualifies(_headers(Event="Rated Blitz game"))
    assert not qualifies(_headers(Event="Casual Rapid game"))
    assert not qualifies(_headers(Variant="Chess960"))
    assert not qualifies(_headers(FEN="8/8/8/8/8/8/8/K6k w - - 0 1"))
    assert not qualifies(_headers(Termination="Abandoned"))
    assert not qualifies(_headers(Termination="Rules infraction"))


def test_qualifies_checks_rating_band_and_gap():
    assert not qualifies(_headers(WhiteElo="2201"))
    assert not qualifies(_headers(BlackElo="1499"))
    assert not qualifies(_headers(WhiteElo="1800", BlackElo="1699"))
    assert qualifies(_headers(WhiteElo="1800", BlackElo="1700"))
    assert not qualifies(_headers(WhiteElo="?"))
    headers = _headers()
    del headers["BlackElo"]
    assert not qualifies(headers)


def test_read_club_games_filters_counts_and_stops_at_the_limit():
    lines = (
        _pgn(_headers(Event="Rated Blitz game"), TWELVE)
        + _pgn(_headers(), "1. e4 e5 1-0")  # too short
        + _pgn(_headers(), TWELVE)
        + _pgn(_headers(Result="0-1"), TWELVE)
        + _pgn(_headers(), TWELVE)
    )
    stats = ClubStreamStats()
    games = list(read_club_games(lines, limit=2, stats=stats))
    assert [g.result for g in games] == ["1-0", "0-1"]
    assert games[0].sans[:2] == ["e4", "e5"]
    assert (games[0].white_elo, games[0].black_elo) == (1800, 1750)
    assert stats.seen == 4
    assert stats.qualified == 2


def test_group_master_rows_builds_standard_start_games_only():
    sans = movetext_to_sans(TWELVE)
    rows = [("g1", "1-0", san, AFTER_E4 if i == 0 else None) for i, san in enumerate(sans)]
    rows += [("g2", "0-1", san, MIDGAME if i == 0 else None) for i, san in enumerate(sans)]
    rows += [("g3", "1/2-1/2", "e4", AFTER_E4)]  # too short
    games = list(group_master_rows(rows))
    assert len(games) == 1
    assert games[0].result == "1-0"
    assert games[0].sans == sans
