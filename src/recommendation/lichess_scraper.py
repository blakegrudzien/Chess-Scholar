"""Scraper for Lichess's study listing pages (lichess.org/study/all/{sort})
and study pages.

The API can list one user's studies or fetch one by id, but can't list the
most popular studies site-wide or give chapter ids, so this reads the
server-rendered HTML instead. It has no documented contract and can break
when Lichess changes its markup, which is why it lives apart from
lichess_client: finding no cards most likely means the markup changed.

Each listing page renders up to STUDIES_PER_PAGE `div.study.paginated`
cards without JavaScript, and `?page=N` pages through them (checked
2026-08-30).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from src.recommendation.lichess_client import RequestPacer, default_http_client

logger = logging.getLogger(__name__)

# The sort keys Lichess's listing navigation links to ("newest", not "new").
VALID_SORTS = frozenset({"hot", "newest", "popular", "updated"})

# Observed on every fetched page during verification. A page returning fewer
# than this many cards is the last page of the listing.
STUDIES_PER_PAGE = 16


@dataclass(frozen=True)
class ScrapedStudyCard:
    """One study as listed on a lichess.org/study/all/{sort} page.

    Read from HTML class names, so nothing here is guaranteed. `likes` is
    the number Lichess shows next to the bookmark icon.
    """

    study_id: str
    title: str
    author: str
    likes: int
    updated_at: str
    chapter_titles: list[str] = field(default_factory=list)
    member_usernames: list[str] = field(default_factory=list)


def _parse_card(card) -> ScrapedStudyCard | None:
    """Parse one `div.study.paginated` card, or return None if its markup is
    unexpected, so one odd card doesn't fail the page."""
    link = card.select_one("a.overlay")
    name_el = card.select_one("h2.study-name")
    meta = card.select_one(".top span")
    time_el = card.select_one("time")
    if link is None or name_el is None or meta is None or time_el is None:
        return None
    href = link.get("href", "")
    if not href.startswith("/study/"):
        return None

    # Rendered as "<likes> • <author> • <relative time>". The time text is
    # filled in by JavaScript, so the timestamp comes from the datetime
    # attribute; only likes and author are read from the text.
    parts = [p.strip() for p in meta.get_text(" ", strip=True).split("•") if p.strip()]
    likes = int(parts[0]) if parts and parts[0].isdigit() else 0
    author = parts[1] if len(parts) > 1 else ""

    return ScrapedStudyCard(
        study_id=href.removeprefix("/study/"),
        title=name_el.get_text(strip=True),
        author=author,
        likes=likes,
        updated_at=time_el.get("datetime", ""),
        chapter_titles=[li.get_text(strip=True) for li in card.select("ol.chapters li")],
        member_usernames=[li.get_text(strip=True) for li in card.select("ol.members li")],
    )


def iter_studies_by_sort(
    sort: str,
    *,
    max_pages: int | None = None,
    http_client: httpx.Client | None = None,
    pacer: RequestPacer | None = None,
) -> Iterator[ScrapedStudyCard]:
    """Yield studies from a listing, page by page, until a page comes back
    short, max_pages is reached, or Lichess returns 400 past its depth
    limit. Pass the caller's RequestPacer so listing and other requests
    share one rate limit."""
    if sort not in VALID_SORTS:
        raise ValueError(f"sort must be one of {sorted(VALID_SORTS)}, got {sort!r}")

    owns_client = http_client is None
    client = http_client or default_http_client()
    pacer = pacer or RequestPacer()
    try:
        page = 1
        while max_pages is None or page <= max_pages:
            pacer.wait()
            response = client.get(f"/study/all/{sort}", params={"page": page})
            if response.status_code == 400:
                # The "popular" listing returns 400 past its maximum page
                # depth instead of an empty page; that is the end of it.
                logger.info("Reached the end of the %r listing at page %d", sort, page)
                return
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            cards = soup.select("div.study.paginated")
            if not cards:
                return
            for card in cards:
                parsed = _parse_card(card)
                if parsed is None:
                    logger.warning(
                        "Skipping a study card with unexpected markup on %s page %d",
                        sort,
                        page,
                    )
                    continue
                yield parsed
            if len(cards) < STUDIES_PER_PAGE:
                return
            page += 1
    finally:
        if owns_client:
            client.close()


@dataclass(frozen=True)
class StudyChapter:
    chapter_id: str
    name: str


def fetch_study_chapters(
    study_id: str,
    *,
    http_client: httpx.Client | None = None,
    pacer: RequestPacer | None = None,
) -> list[StudyChapter]:
    """Fetch the chapter list (id and name) for one study.

    The API and the PGN export don't include chapter ids, so this reads the
    study page's `<script type="application/json" id="page-init-data">`
    block, which holds study.chapters with their ids.

    Returns an empty list for unexpected markup or a private or deleted
    study, rather than raising.
    """
    owns_client = http_client is None
    client = http_client or default_http_client()
    pacer = pacer or RequestPacer()
    try:
        pacer.wait()
        response = client.get(f"/study/{study_id}")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        script = soup.find("script", id="page-init-data")
        if script is None or not script.string:
            logger.warning("No page-init-data block found for study %s", study_id)
            return []
        try:
            data = json.loads(script.string)
            chapters = data["study"]["chapters"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Unexpected page-init-data shape for study %s: %s", study_id, exc)
            return []
        return [StudyChapter(chapter_id=c["id"], name=c["name"]) for c in chapters]
    finally:
        if owns_client:
            client.close()
