"""Collect a raw pool of Lichess studies for the recommendation feature's
quality-classifier labeling step (Step 2 of the recommendation roadmap).

Pulls from Lichess listing sorts to get contrast in the pool the user will
hand-label from:

  - "popular": Lichess's own most-liked studies, a positive-skewed sample
  - "newest": freshly created, unvetted studies, a negative-skewed sample
  - "hot": recently active studies -- verified live to span roughly 9 to a
    few thousand likes, filling the gap "popular" and "newest" leave empty
    (the first two sorts alone produced a training set with a hard gap:
    zero candidates between 5 and 1000 likes, since "popular" only surfaces
    the extreme top and "newest" only the extreme bottom)

All three come from lichess_scraper (there is no API for "list studies by
popularity"), while the full content of each discovered study is fetched
through the documented lichess_client API. Results are written to a local,
gitignored JSONL file -- one candidate per line -- so the labeling app
(Step 3) can review them offline without hitting Lichess again.

Re-running this script is additive, not destructive: existing candidates in
the output file (and, by extension, any labels already recorded against
them in study_labels.jsonl) are preserved. New candidates are merged in;
already-known study_ids are skipped before ever issuing a content-fetch
request for them, both to save requests and because their labels must stay
matched to the exact record they were labeled against.

Usage (run as a module from the repo root, not as a bare script path --
"python scripts/x.py" puts scripts/ on sys.path instead of the repo root,
which breaks the "from src...." imports below):
    .venv/bin/python -m scripts.collect_study_candidates --popular-count 100 --newest-count 100
    .venv/bin/python -m scripts.collect_study_candidates --hot-count 80
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


def _load_existing_records(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    records = {}
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record["study_id"]] = record
    return records


def collect_candidates(sort_counts: dict[str, int], output_path: Path) -> None:
    existing = _load_existing_records(output_path)
    logger.info("Found %d existing candidates in %s", len(existing), output_path)

    pacer = RequestPacer()
    candidates: list[tuple[ScrapedStudyCard, str]] = []
    for sort, count in sort_counts.items():
        if count <= 0:
            continue
        for card in _collect_cards(sort, count, pacer):
            candidates.append((card, sort))

    written = 0
    skipped = 0
    already_known = 0

    with LichessClient(pacer=pacer) as client:
        for card, source_sort in candidates:
            if card.study_id in existing:
                already_known += 1
                continue

            try:
                pgn = client.fetch_study_pgn(card.study_id)
            except httpx.HTTPError as exc:
                logger.warning("Skipping study %s, failed to fetch content: %s", card.study_id, exc)
                skipped += 1
                continue

            existing[card.study_id] = {**asdict(card), "source_sort": source_sort, "pgn": pgn}
            written += 1
            if written % PROGRESS_LOG_INTERVAL == 0:
                logger.info("Fetched content for %d new studies so far", written)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in existing.values():
            out.write(json.dumps(record) + "\n")

    logger.info(
        "Wrote %d total candidates to %s (%d newly added, %d already known, %d skipped)",
        len(existing),
        output_path,
        written,
        already_known,
        skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--popular-count", type=int, default=0)
    parser.add_argument("--newest-count", type=int, default=0)
    parser.add_argument("--hot-count", type=int, default=0)
    parser.add_argument("--updated-count", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    sort_counts = {
        "popular": args.popular_count,
        "newest": args.newest_count,
        "hot": args.hot_count,
        "updated": args.updated_count,
    }
    if not any(count > 0 for count in sort_counts.values()):
        parser.error(
            "at least one of --popular-count/--newest-count/--hot-count/--updated-count "
            "must be greater than 0"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect_candidates(sort_counts, args.output)


if __name__ == "__main__":
    main()
