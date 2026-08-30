"""Collect a raw pool of Lichess studies for the recommendation feature's
quality-classifier labeling step (Step 2 of the recommendation roadmap).

Pulls from two Lichess listing sorts to get contrast in the pool the user
will hand-label from:

  - "popular": Lichess's own most-liked studies, a positive-skewed sample
  - "newest": freshly created, unvetted studies, a negative-skewed sample

Both come from lichess_scraper (there is no API for "most popular studies
site-wide"), while the full content of each discovered study is fetched
through the documented lichess_client API. Results are written to a local,
gitignored JSONL file -- one candidate per line -- so the labeling app
(Step 3) can review them offline without hitting Lichess again.

Usage (run as a module from the repo root, not as a bare script path --
"python scripts/x.py" puts scripts/ on sys.path instead of the repo root,
which breaks the "from src...." imports below):
    .venv/bin/python -m scripts.collect_study_candidates
    .venv/bin/python -m scripts.collect_study_candidates --popular-count 50 --newest-count 50
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import httpx

from src.recommendation.lichess_client import LichessClient, RequestPacer
from src.recommendation.lichess_scraper import ScrapedStudyCard, iter_studies_by_sort

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("data/processed/study_candidates.jsonl")
PROGRESS_LOG_INTERVAL = 10


def _collect_cards(sort: str, count: int, pacer: RequestPacer) -> list[ScrapedStudyCard]:
    """Pull exactly `count` cards from one listing sort, or fewer if the
    listing runs out first. iter_studies_by_sort paginates lazily, so this
    stops issuing requests the moment enough cards are collected instead of
    always walking the whole listing.
    """
    cards: list[ScrapedStudyCard] = []
    for card in iter_studies_by_sort(sort, pacer=pacer):
        cards.append(card)
        if len(cards) >= count:
            break
    logger.info("Collected %d/%d cards from %r listing", len(cards), count, sort)
    return cards


def collect_candidates(
    popular_count: int,
    newest_count: int,
    output_path: Path,
) -> None:
    pacer = RequestPacer()
    candidates: list[tuple[ScrapedStudyCard, str]] = []
    for sort, count in (("popular", popular_count), ("newest", newest_count)):
        for card in _collect_cards(sort, count, pacer):
            candidates.append((card, sort))

    seen_ids: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as out, LichessClient(pacer=pacer) as client:
        for card, source_sort in candidates:
            if card.study_id in seen_ids:
                continue
            seen_ids.add(card.study_id)

            try:
                pgn = client.fetch_study_pgn(card.study_id)
            except httpx.HTTPError as exc:
                logger.warning("Skipping study %s, failed to fetch content: %s", card.study_id, exc)
                skipped += 1
                continue

            record = {**asdict(card), "source_sort": source_sort, "pgn": pgn}
            out.write(json.dumps(record) + "\n")
            out.flush()
            written += 1
            if written % PROGRESS_LOG_INTERVAL == 0:
                logger.info("Fetched content for %d studies so far", written)

    logger.info("Wrote %d candidates to %s (%d skipped)", written, output_path, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--popular-count", type=int, default=100)
    parser.add_argument("--newest-count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect_candidates(args.popular_count, args.newest_count, args.output)


if __name__ == "__main__":
    main()
