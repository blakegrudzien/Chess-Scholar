"""Scraper for Lichess's study *listing* pages -- lichess.org/study/all/{sort}.

Unlike lichess_client.py, this has no documented contract: it parses
server-rendered HTML that Lichess could restructure at any time without
notice. It exists only because there is no API for what it does. Lichess's
own OpenAPI spec has no "list/search studies by popularity" endpoint, only
per-user listing (lichess_client.iter_studies_by_username) and per-id fetch
(lichess_client.fetch_study_pgn) -- neither of which can answer "what are
the most-liked studies on the whole site right now."

Kept in its own module, not folded into lichess_client, so the reliability
boundary is visible at the import line: code that imports lichess_client is
talking to a stable, documented API; code that imports this module is
parsing a web page and should treat a failure to find any cards as "the
site changed its markup," not "our request was malformed."

Verified live against https://lichess.org/study/all/{popular,newest} on
2026-08-30: each page renders up to STUDIES_PER_PAGE `div.study.paginated`
cards server-side (confirmed via a plain curl, no JavaScript involved), and
`?page=N` pages through them even though the live site itself uses infinite
scroll for this same listing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from src.recommendation.lichess_client import LICHESS_BASE_URL, RequestPacer

logger = logging.getLogger(__name__)

# The sort keys Lichess's own listing-page navigation actually links to,
# confirmed by reading the rendered nav rather than guessing ("new" 404s;
# the real key is "newest").
VALID_SORTS = frozenset({"hot", "newest", "popular", "updated"})

# Observed on every fetched page during verification. A page returning fewer
# than this many cards is the last page of the listing.
STUDIES_PER_PAGE = 16


@dataclass(frozen=True)
class ScrapedStudyCard:
    """One study as listed on a lichess.org/study/all/{sort} page.

    Unlike the API's study objects, no field here carries a documented
    guarantee -- they're read positionally off HTML class names that Lichess
    could change. `likes` is whatever number Lichess displays next to the
    bookmark icon; this scraper does not interpret what that metric means
    beyond reading it.
    """

    study_id: str
    title: str
    author: str
    likes: int
    updated_at: str
    chapter_titles: list[str] = field(default_factory=list)
    member_usernames: list[str] = field(default_factory=list)


def _parse_card(card) -> ScrapedStudyCard | None:
    """Parse one `div.study.paginated` card, or return None if its markup
    doesn't match what verification found -- callers log and skip rather
    than fail the whole page over one unexpected card.
    """
    link = card.select_one("a.overlay")
    name_el = card.select_one("h2.study-name")
    meta = card.select_one(".top span")
    time_el = card.select_one("time")
    if link is None or name_el is None or meta is None or time_el is None:
        return None
    href = link.get("href", "")
    if not href.startswith("/study/"):
        return None

    # Rendered as "<likes> • <author> • <relative time>". The <time>
    # element carries the real timestamp in its datetime attribute (the
    # visible relative text, e.g. "3 hours ago", is filled in client-side by
    # JavaScript and so is empty in the server-rendered HTML this scraper
    # reads) -- only the like count and author need pulling from the text.
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
    """Yield studies from a lichess.org/study/all/{sort} listing, page by
    page, stopping when a page returns fewer than STUDIES_PER_PAGE cards
    (the natural end of the listing) or when max_pages is reached,
    whichever comes first.

    Accepts an external RequestPacer for the same reason LichessClient
    does: a caller issuing both listing and content-fetch requests in the
    same run should have them share one rate limit against the same host,
    not enforce two independent ones.
    """
    if sort not in VALID_SORTS:
        raise ValueError(f"sort must be one of {sorted(VALID_SORTS)}, got {sort!r}")

    owns_client = http_client is None
    client = http_client or httpx.Client(base_url=LICHESS_BASE_URL, timeout=30.0)
    pacer = pacer or RequestPacer()
    try:
        page = 1
        while max_pages is None or page <= max_pages:
            pacer.wait()
            response = client.get(f"/study/all/{sort}", params={"page": page})
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

    Neither Lichess's documented API nor the PGN export
    (lichess_client.fetch_study_pgn) exposes chapter ids -- the PGN's
    per-chapter [Event] header carries only the chapter name. This instead
    reads the plain study page (lichess.org/study/{study_id}), which
    server-renders a `<script type="application/json" id="page-init-data">`
    block containing the same data the page's own JavaScript bootstraps
    from, including a full study.chapters list with real ids. No
    authentication or websocket connection needed -- confirmed live against
    a real study, whose 10 chapters (Introduction, Standart Scheme, Dragon
    Accelerated, ...) all appeared with distinct 8-character ids in that
    one block.

    Returns an empty list if the page doesn't have the expected data block
    (unexpected markup, private/deleted study) rather than raising --
    a caller resolving a chapter for an embed URL should treat "unknown"
    the same as "couldn't determine one," not crash the whole request.
    """
    owns_client = http_client is None
    client = http_client or httpx.Client(base_url=LICHESS_BASE_URL, timeout=30.0)
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
