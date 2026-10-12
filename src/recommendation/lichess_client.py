"""A small client for Lichess's documented Studies API, used by the study
recommendations.

It fetches a study by id and lists a user's studies. The API has no topic
search, so topic matching runs over this project's own indexed copy of
selected studies.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from types import TracebackType
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

LICHESS_BASE_URL = "https://lichess.org"

# Every request to Lichess (API and scraper) identifies the project and how
# to reach it, instead of httpx's anonymous default User-Agent.
USER_AGENT = "chess-scholar/0.1.0 (+https://github.com/blakegrudzien/Chess-Scholar)"

# Lichess asks for one request at a time and a one-minute wait after a 429.
REQUEST_PACING_SECONDS = 1.0
RATE_LIMIT_BACKOFF_SECONDS = 60
MAX_RATE_LIMIT_RETRIES = 3


def default_http_client() -> httpx.Client:
    """The httpx.Client every Lichess request uses, so the User-Agent, base
    URL and timeout are set in one place."""
    return httpx.Client(base_url=LICHESS_BASE_URL, timeout=30.0, headers={"User-Agent": USER_AGENT})


class RequestPacer:
    """Keeps REQUEST_PACING_SECONDS between requests to lichess.org, whoever
    makes them. Shared by the API client and the scraper."""

    def __init__(self) -> None:
        self._last_request_at: float | None = None

    def wait(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            remaining = REQUEST_PACING_SECONDS - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()


class LichessClient:
    """A connection to Lichess's API, reused across calls so request pacing
    (see _pace) applies across the whole session, not just within one call.

    Use as a context manager so the underlying HTTP connection is always
    closed, the same RAII-style pattern EnginePool and the DB pool use
    elsewhere in this project for their own external resources.
    """

    def __init__(
        self,
        http_client: httpx.Client | None = None,
        pacer: RequestPacer | None = None,
    ) -> None:
        self._http = http_client or default_http_client()
        # Accepts an external pacer so a caller making both API and scraper
        # requests in the same run (see collect_study_candidates.py) can
        # share one rate limit across both instead of each object enforcing
        # its own, independent 1-request-per-second ceiling.
        self._pacer = pacer or RequestPacer()

    def __enter__(self) -> LichessClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def fetch_study_pgn(self, study_id: str) -> str:
        """Fetch an entire study (every chapter) as PGN, comments included.

        Retries a fixed number of times on 429, waiting the backoff Lichess
        itself documents each time, then gives up loudly rather than
        retrying forever against a service that's telling us to stop.
        """
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            self._pacer.wait()
            response = self._http.get(f"/api/study/{study_id}.pgn")
            if response.status_code == 429:
                if attempt == MAX_RATE_LIMIT_RETRIES - 1:
                    response.raise_for_status()
                logger.warning("Rate limited fetching study %s, backing off", study_id)
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            response.raise_for_status()
            return response.text
        # Unreachable: every attempt returns or raises. Stated so a type
        # checker doesn't see an implicit None return.
        raise AssertionError("unreachable: every fetch_study_pgn attempt returns or raises")

    def iter_studies_by_username(self, username: str, limit: int | None = None) -> Iterator[dict]:
        """Yield {id, name, createdAt, updatedAt} for a user's public
        studies, most recently updated first.

        The endpoint streams newline-delimited JSON and can be slow and huge
        for an active account, so it's read line by line and `limit` stops
        early.
        """
        self._pacer.wait()
        count = 0
        with self._http.stream("GET", f"/api/study/by/{username}") as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                yield json.loads(line)
                count += 1
                if limit is not None and count >= limit:
                    return


def game_embed_url(game_id: str, *, theme: str = "auto", bg: str = "auto") -> str:
    """A read-only embed URL for one game (Lichess's "Embed in your
    website" format)."""
    return f"{LICHESS_BASE_URL}/embed/{quote(game_id, safe='')}?theme={theme}&bg={bg}"


def study_chapter_embed_url(study_id: str, chapter_id: str, *, bg: str = "dark") -> str:
    """A read-only embed URL for one study chapter. The chapter is required:
    Lichess's embed format has no whole-study form.

    bg defaults to "dark" rather than Lichess's "auto", which follows the
    viewer's system theme and shows a bright white board on most screens,
    out of place in this app's wood-toned design.
    """
    return (
        f"{LICHESS_BASE_URL}/study/embed/"
        f"{quote(study_id, safe='')}/{quote(chapter_id, safe='')}?bg={bg}"
    )
