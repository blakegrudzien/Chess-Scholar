import io
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.spikes import structure_discovery
from scripts.spikes.discovery_sources import SourceGame
from scripts.spikes.pawn_keys import Interval
from scripts.spikes.structure_discovery import Tally, main, render_report

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
START_BOTH = "W:c2 d2 e2 f2 / B:c7 d7 e7 f7"
START_FULL = "W:a2 b2 c2 d2 e2 f2 g2 h2 / B:a7 b7 c7 d7 e7 f7 g7 h7"
MASTERS_WARNING = (
    "Warning: the master tally is incomplete, so master columns and coverage are not final."
)


class FakePool:
    """Stands in for the database pool and records what main() hands back."""

    def __init__(self):
        self.returned = []
        self.closed = False

    def getconn(self):
        return "conn"

    def putconn(self, conn):
        self.returned.append(conn)

    def closeall(self):
        self.closed = True


def _run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["structure_discovery", *args])
    main()


def _club_pgn(games: int) -> str:
    """A Lichess-style PGN with `games` rapid games that pass every filter."""
    movetext = " ".join(
        f"{ply // 2 + 1}. {san}" if ply % 2 == 0 else san for ply, san in enumerate(KNIGHT_SHUFFLE)
    )
    game = (
        '[Event "Rated Rapid game"]\n[Result "1-0"]\n[WhiteElo "1800"]\n[BlackElo "1800"]\n'
        f"\n{movetext} 1-0\n\n"
    )
    return game * games


def _row(report, key):
    """The both-sides table row for one key."""
    return next(line for line in report.splitlines() if line.startswith(f"| {key} |"))


def test_add_game_counts_a_structure_once_per_game_with_its_exit_and_result():
    tally = Tally()
    assert tally.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    assert tally.games == 1
    stats = tally.stats["both"][START_BOTH]
    assert (stats.games, stats.intervals, stats.plies) == (1, 1, 12)
    assert stats.results["win"] == 1
    assert stats.exits["W e4"] == 1
    assert tally.stats["black"]["B:c7 d7 e7 f7"].results["loss"] == 1


def test_add_game_skips_illegal_games():
    tally = Tally()
    assert not tally.add_game(["e4", "e4"], "1-0")
    assert (tally.games, tally.skipped) == (0, 1)


def test_add_game_counts_any_failure_as_skipped(monkeypatch):
    def broken_keys(sans):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(structure_discovery, "game_keys", broken_keys)
    tally = Tally()
    assert not tally.add_game(KNIGHT_SHUFFLE, "1-0")
    assert (tally.games, tally.skipped) == (0, 1)


def test_a_key_counts_once_per_game_even_when_it_has_two_runs(monkeypatch):
    runs = [
        Interval("W:c2 d2 e2 f2", 0, 11, "W e4", "W:c2 d2 e4 f2"),
        Interval("W:c2 d2 e2 f2", 14, 25, None, None),
    ]
    monkeypatch.setattr(structure_discovery, "find_intervals", lambda keys, sans: runs)
    tally = Tally()
    assert tally.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    stats = tally.stats["white"]["W:c2 d2 e2 f2"]
    assert (stats.games, stats.intervals) == (1, 2)
    assert stats.results["win"] == 1


def test_to_json_is_plain_data():
    tally = Tally()
    tally.add_game(KNIGHT_SHUFFLE + ["e4"], "1/2-1/2")
    data = tally.to_json()
    assert data["games"] == 1
    assert data["stats"]["both"][START_BOTH]["results"] == {"draw": 1}


def test_report_flags_keys_missing_from_master_games():
    club = Tally()
    club.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    masters = Tally()
    report = render_report(club, masters, notes=["note line"])
    assert "note line" in report
    assert START_BOTH in report
    assert "thin" in report
    assert "|---" in report


def test_coverage_is_thin_only_below_the_master_threshold():
    club, masters = Tally(), Tally()
    club.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    masters.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    thin = render_report(club, Tally(), notes=[])
    assert "| thin |" in _row(thin, START_BOTH)
    enough = render_report(club, masters, notes=[], min_master_games=1)
    assert "thin" not in _row(enough, START_BOTH)


def test_incomplete_master_tally_is_flagged_instead_of_thin():
    club = Tally()
    club.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    report = render_report(club, Tally(), notes=[], masters_complete=False)
    # Right under the title, so it is read before any verdict.
    assert report.splitlines()[:3] == ["# Pawn structure discovery", "", MASTERS_WARNING]
    row = _row(report, START_BOTH)
    assert "| incomplete |" in row
    assert "thin" not in row


def test_complete_master_tally_has_no_warning():
    club = Tally()
    club.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    report = render_report(club, Tally(), notes=[])
    assert "Warning:" not in report
    assert "incomplete" not in _row(report, START_BOTH)


def test_wdl_shares_leave_out_unknown_results():
    tally = Tally()
    for result in ("1-0", "1/2-1/2", "*"):
        tally.add_game(KNIGHT_SHUFFLE + ["e4"], result)
    report = render_report(tally, Tally(), notes=[])
    assert "| 50% / 50% / 0% |" in _row(report, START_BOTH)


def test_a_single_run_ended_by_a_move_shows_it_at_100_percent():
    tally = Tally()
    tally.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    report = render_report(tally, Tally(), notes=[])
    assert "| W e4 (100%) |" in _row(report, START_BOTH)


def test_exits_show_each_move_and_the_game_end_as_shares_of_runs():
    tally = Tally()
    tally.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    tally.add_game(KNIGHT_SHUFFLE, "1-0")
    report = render_report(tally, Tally(), notes=[])
    assert "| Club exits (% of runs) | Master exits (% of runs) |" in report
    assert "| W e4 (50%), game end (50%) |" in _row(report, START_BOTH)
    assert "Exits are shares of runs" in report


def test_exits_show_game_end_when_no_run_ends_in_a_move():
    tally = Tally()
    tally.add_game(KNIGHT_SHUFFLE, "1-0")
    report = render_report(tally, Tally(), notes=[])
    assert "| game end (100%) |" in _row(report, START_BOTH)


def test_full_board_section_lists_the_full_keys_under_their_family():
    tally = Tally()
    tally.add_game(KNIGHT_SHUFFLE + ["e4"], "1-0")
    report = render_report(tally, Tally(), notes=[])
    assert f"### {START_BOTH}" in report
    assert f"| {START_FULL} | 1 | 0 |" in report


def test_main_connects_before_reading_the_club_stream(tmp_path, monkeypatch):
    def no_database():
        raise RuntimeError("DATABASE_URL is not set")

    monkeypatch.setattr(structure_discovery, "get_connection_pool", no_database)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        _run_main(
            monkeypatch,
            "--club-pgn",
            str(tmp_path / "missing.pgn"),
            "--report",
            str(tmp_path / "report.md"),
        )


def test_main_keeps_the_run_when_the_master_read_fails(tmp_path, monkeypatch):
    club_pgn = tmp_path / "club.pgn"
    club_pgn.write_text("", encoding="utf-8")
    report_path = tmp_path / "report.md"
    json_path = tmp_path / "out" / "tally.json"
    pool = FakePool()

    def master_games_then_error(conn):
        yield SourceGame(result="1-0", sans=KNIGHT_SHUFFLE + ["e4"])
        raise RuntimeError("boom")

    monkeypatch.setattr(structure_discovery, "get_connection_pool", lambda: pool)
    monkeypatch.setattr(structure_discovery, "iter_master_games", master_games_then_error)
    _run_main(
        monkeypatch,
        "--club-pgn",
        str(club_pgn),
        "--report",
        str(report_path),
        "--json",
        str(json_path),
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Master tally incomplete after 1 games: RuntimeError: boom" in report
    assert MASTERS_WARNING in report
    # The full-sample description is only true for a finished tally.
    assert "Master sample: every standard-start" not in report
    assert json.loads(json_path.read_text(encoding="utf-8"))["masters"]["games"] == 1
    assert pool.returned and pool.closed


def test_main_describes_the_master_sample_only_when_the_tally_finished(tmp_path, monkeypatch):
    club_pgn = tmp_path / "club.pgn"
    club_pgn.write_text("", encoding="utf-8")
    report_path = tmp_path / "report.md"

    def one_master_game(conn):
        yield SourceGame(result="1-0", sans=KNIGHT_SHUFFLE + ["e4"])

    monkeypatch.setattr(structure_discovery, "get_connection_pool", lambda: FakePool())
    monkeypatch.setattr(structure_discovery, "iter_master_games", one_master_game)
    _run_main(monkeypatch, "--club-pgn", str(club_pgn), "--report", str(report_path))

    report = report_path.read_text(encoding="utf-8")
    assert "Master sample: every standard-start ChessBase game of at least 10 plies" in report
    assert "Warning:" not in report


def test_club_note_says_when_the_stream_ended_before_the_limit(tmp_path, monkeypatch, capsys):
    club_pgn = tmp_path / "club.pgn"
    club_pgn.write_text(_club_pgn(3), encoding="utf-8")
    report_path = tmp_path / "report.md"
    _run_main(
        monkeypatch,
        "--club-pgn",
        str(club_pgn),
        "--club-games",
        "5",
        "--report",
        str(report_path),
        "--skip-masters",
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Club sample: 3 qualifying games" in report
    assert "the stream ended before the 5-game limit" in report
    assert "stopped at" not in report
    assert "the stream ended before the 5-game limit" in capsys.readouterr().err


def test_club_note_says_when_the_limit_was_reached(tmp_path, monkeypatch, capsys):
    club_pgn = tmp_path / "club.pgn"
    club_pgn.write_text(_club_pgn(3), encoding="utf-8")
    report_path = tmp_path / "report.md"
    _run_main(
        monkeypatch,
        "--club-pgn",
        str(club_pgn),
        "--club-games",
        "2",
        "--report",
        str(report_path),
        "--skip-masters",
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Club sample: 2 qualifying games" in report
    assert "stopped at the 2-game limit" in report
    assert "ended before" not in report
    assert "stopped at the 2-game limit" in capsys.readouterr().err


def test_main_closes_the_club_stdin_stream_before_the_master_pass(tmp_path, monkeypatch):
    buffer = io.BytesIO(_club_pgn(1).encode("utf-8"))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=buffer))
    closed_when_masters_started = []

    def masters(conn):
        closed_when_masters_started.append(buffer.closed)
        return iter(())

    monkeypatch.setattr(structure_discovery, "get_connection_pool", lambda: FakePool())
    monkeypatch.setattr(structure_discovery, "iter_master_games", masters)
    _run_main(monkeypatch, "--club-pgn", "-", "--report", str(tmp_path / "report.md"))
    # Closing the stream lets the curl and zstd processes upstream exit
    # instead of waiting through the long master pass.
    assert closed_when_masters_started == [True]


def test_main_skip_masters_notes_empty_master_columns(tmp_path, monkeypatch):
    club_pgn = tmp_path / "club.pgn"
    club_pgn.write_text("", encoding="utf-8")
    report_path = tmp_path / "report.md"

    def no_database():
        raise AssertionError("the database must not be opened when masters are skipped")

    monkeypatch.setattr(structure_discovery, "get_connection_pool", no_database)
    _run_main(
        monkeypatch,
        "--club-pgn",
        str(club_pgn),
        "--report",
        str(report_path),
        "--skip-masters",
    )
    report = report_path.read_text(encoding="utf-8")
    assert "Master games were skipped in this run, so master columns are empty." in report


def test_main_reads_a_club_file_with_bad_bytes(tmp_path, monkeypatch):
    club_pgn = tmp_path / "club.pgn"
    club_pgn.write_bytes(b'[Event "Rated Rapid game \xff"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
    report_path = tmp_path / "report.md"

    _run_main(
        monkeypatch,
        "--club-pgn",
        str(club_pgn),
        "--report",
        str(report_path),
        "--skip-masters",
    )
    assert "Club sample: 0 qualifying games" in report_path.read_text(encoding="utf-8")


def test_report_handles_empty_tallies():
    report = render_report(Tally(), Tally(), notes=[])
    assert "No club games" in report
