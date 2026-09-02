import json
from unittest.mock import patch

import httpx
import pytest

from scripts.collect_study_candidates import collect_candidates
from src.recommendation.lichess_scraper import ScrapedStudyCard


def _card(study_id: str) -> ScrapedStudyCard:
    return ScrapedStudyCard(
        study_id=study_id,
        title=f"Study {study_id}",
        author="someauthor",
        likes=10,
        updated_at="2024-01-01T00:00:00.000Z",
        chapter_titles=["Chapter 1"],
        member_usernames=["someauthor"],
    )


def _fake_listing(cards_by_sort: dict[str, list[ScrapedStudyCard]]):
    def iter_studies_by_sort(sort, *, pacer=None, **kwargs):
        yield from cards_by_sort.get(sort, [])

    return iter_studies_by_sort


def test_collect_candidates_writes_one_jsonl_line_per_study(tmp_path):
    output_path = tmp_path / "candidates.jsonl"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing(
                {"popular": [_card("p1"), _card("p2")], "newest": [_card("n1")]}
            ),
        ),
        patch(
            "scripts.collect_study_candidates.LichessClient.fetch_study_pgn",
            side_effect=lambda study_id: f"PGN for {study_id}",
        ),
    ):
        collect_candidates({"popular": 2, "newest": 1}, output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [r["study_id"] for r in records] == ["p1", "p2", "n1"]
    assert [r["source_sort"] for r in records] == ["popular", "popular", "newest"]
    assert records[0]["pgn"] == "PGN for p1"
    assert records[0]["title"] == "Study p1"


def test_collect_candidates_stops_at_requested_count_even_with_more_available(tmp_path):
    output_path = tmp_path / "candidates.jsonl"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing({"popular": [_card(f"p{i}") for i in range(5)]}),
        ),
        patch(
            "scripts.collect_study_candidates.LichessClient.fetch_study_pgn",
            side_effect=lambda study_id: "pgn",
        ),
    ):
        collect_candidates({"popular": 2}, output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [r["study_id"] for r in records] == ["p0", "p1"]


def test_collect_candidates_deduplicates_within_one_run(tmp_path):
    # The same study could plausibly appear in both the popular and newest
    # listings; each study_id should still be written at most once.
    shared_card = _card("dup1")
    output_path = tmp_path / "candidates.jsonl"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing({"popular": [shared_card], "newest": [shared_card]}),
        ),
        patch(
            "scripts.collect_study_candidates.LichessClient.fetch_study_pgn",
            side_effect=lambda study_id: "pgn",
        ),
    ):
        collect_candidates({"popular": 1, "newest": 1}, output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 1


def test_collect_candidates_skips_a_study_whose_content_fetch_fails(tmp_path):
    cards = [_card("ok1"), _card("broken"), _card("ok2")]
    output_path = tmp_path / "candidates.jsonl"

    def flaky_fetch(study_id):
        if study_id == "broken":
            raise httpx.HTTPStatusError("boom", request=None, response=None)
        return "pgn"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing({"popular": cards}),
        ),
        patch(
            "scripts.collect_study_candidates.LichessClient.fetch_study_pgn",
            side_effect=flaky_fetch,
        ),
    ):
        collect_candidates({"popular": 3}, output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [r["study_id"] for r in records] == ["ok1", "ok2"]


def test_collect_candidates_preserves_existing_records_across_runs(tmp_path):
    # Simulates the real incremental-collection use case: a second run
    # targeting a different sort must not lose or re-fetch what a first
    # run already collected (and what may already have labels attached).
    output_path = tmp_path / "candidates.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "study_id": "old1",
                "title": "Old Study",
                "likes": 5000,
                "author": "x",
                "updated_at": "2024-01-01T00:00:00.000Z",
                "chapter_titles": [],
                "member_usernames": [],
                "source_sort": "popular",
                "pgn": "OLD PGN",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    fetch_calls = []

    def fetch(study_id):
        fetch_calls.append(study_id)
        return f"PGN for {study_id}"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing({"hot": [_card("h1")]}),
        ),
        patch("scripts.collect_study_candidates.LichessClient.fetch_study_pgn", side_effect=fetch),
    ):
        collect_candidates({"hot": 1}, output_path)

    records = {
        r["study_id"]: r
        for r in (json.loads(line) for line in output_path.read_text().splitlines())
    }
    assert set(records) == {"old1", "h1"}
    assert records["old1"]["pgn"] == "OLD PGN"  # untouched, not re-fetched
    assert fetch_calls == ["h1"]  # only the new study triggered a content fetch


def test_collect_candidates_persists_partial_progress_when_it_crashes(tmp_path):
    """Real, reproduced failure: an earlier version only wrote output_path
    once, at the very end -- an unexpected exception partway through the
    run (a listing sort hitting a page past Lichess's depth limit; see
    iter_studies_by_sort's own fix for that specific case) discarded an
    entire run's worth of already-fetched, rate-limited content. Whatever
    made it into `existing` before the crash must survive it.
    """
    cards = [_card("ok1"), _card("ok2"), _card("boom")]
    output_path = tmp_path / "candidates.jsonl"

    def fetch(study_id):
        if study_id == "boom":
            raise RuntimeError("something this code doesn't yet know how to handle")
        return f"PGN for {study_id}"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing({"popular": cards}),
        ),
        patch("scripts.collect_study_candidates.LichessClient.fetch_study_pgn", side_effect=fetch),
        pytest.raises(RuntimeError),
    ):
        collect_candidates({"popular": 3}, output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [r["study_id"] for r in records] == ["ok1", "ok2"]


def test_collect_candidates_does_not_refetch_an_already_known_study_id(tmp_path):
    # A study appearing in a newly-requested sort that was already
    # collected by an earlier run should be left alone, not re-fetched.
    output_path = tmp_path / "candidates.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "study_id": "seen1",
                "title": "Seen",
                "likes": 500,
                "author": "x",
                "updated_at": "2024-01-01T00:00:00.000Z",
                "chapter_titles": [],
                "member_usernames": [],
                "source_sort": "popular",
                "pgn": "ORIGINAL",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing({"hot": [_card("seen1")]}),
        ),
        patch(
            "scripts.collect_study_candidates.LichessClient.fetch_study_pgn",
            side_effect=AssertionError("should not be called for an already-known study_id"),
        ),
    ):
        collect_candidates({"hot": 1}, output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["pgn"] == "ORIGINAL"
