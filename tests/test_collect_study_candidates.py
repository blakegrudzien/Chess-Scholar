import json
from unittest.mock import patch

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


def _fake_listing(popular_cards, newest_cards):
    def iter_studies_by_sort(sort, *, pacer=None, **kwargs):
        yield from (popular_cards if sort == "popular" else newest_cards)

    return iter_studies_by_sort


def test_collect_candidates_writes_one_jsonl_line_per_study(tmp_path):
    popular_cards = [_card("p1"), _card("p2")]
    newest_cards = [_card("n1")]
    output_path = tmp_path / "candidates.jsonl"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing(popular_cards, newest_cards),
        ),
        patch(
            "scripts.collect_study_candidates.LichessClient.fetch_study_pgn",
            side_effect=lambda study_id: f"PGN for {study_id}",
        ),
    ):
        collect_candidates(popular_count=2, newest_count=1, output_path=output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [r["study_id"] for r in records] == ["p1", "p2", "n1"]
    assert [r["source_sort"] for r in records] == ["popular", "popular", "newest"]
    assert records[0]["pgn"] == "PGN for p1"
    assert records[0]["title"] == "Study p1"


def test_collect_candidates_stops_at_requested_count_even_with_more_available(tmp_path):
    popular_cards = [_card(f"p{i}") for i in range(5)]
    output_path = tmp_path / "candidates.jsonl"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing(popular_cards, []),
        ),
        patch(
            "scripts.collect_study_candidates.LichessClient.fetch_study_pgn",
            side_effect=lambda study_id: "pgn",
        ),
    ):
        collect_candidates(popular_count=2, newest_count=0, output_path=output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [r["study_id"] for r in records] == ["p0", "p1"]


def test_collect_candidates_deduplicates_by_study_id(tmp_path):
    # The same study could plausibly appear in both the popular and newest
    # listings; each study_id should still be written at most once.
    shared_card = _card("dup1")
    output_path = tmp_path / "candidates.jsonl"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing([shared_card], [shared_card]),
        ),
        patch(
            "scripts.collect_study_candidates.LichessClient.fetch_study_pgn",
            side_effect=lambda study_id: "pgn",
        ),
    ):
        collect_candidates(popular_count=1, newest_count=1, output_path=output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 1


def test_collect_candidates_skips_a_study_whose_content_fetch_fails(tmp_path):
    import httpx

    cards = [_card("ok1"), _card("broken"), _card("ok2")]
    output_path = tmp_path / "candidates.jsonl"

    def flaky_fetch(study_id):
        if study_id == "broken":
            raise httpx.HTTPStatusError("boom", request=None, response=None)
        return "pgn"

    with (
        patch(
            "scripts.collect_study_candidates.iter_studies_by_sort",
            side_effect=_fake_listing(cards, []),
        ),
        patch(
            "scripts.collect_study_candidates.LichessClient.fetch_study_pgn",
            side_effect=flaky_fetch,
        ),
    ):
        collect_candidates(popular_count=3, newest_count=0, output_path=output_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [r["study_id"] for r in records] == ["ok1", "ok2"]
