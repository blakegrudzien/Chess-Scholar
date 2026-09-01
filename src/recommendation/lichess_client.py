"""Thin wrapper over Lichess's real, documented Studies API, for the study
recommendation feature (Layer 2's sibling: point at good external resources
instead of ingesting and serving their text).

Deliberately narrow: this only fetches a study's content by id and lists a
known user's studies. It does not attempt to search Lichess by topic --
verified against the actual OpenAPI spec, not assumed, no such endpoint
exists. Topic-based discovery happens later, over our own indexed copy of a
curated set of studies (collected once via this client, scored, embedded,
and cached), not by querying Lichess live per question.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from types import TracebackType

import httpx

logger = logging.getLogger(__name__)

LICHESS_BASE_URL = "https://lichess.org"

# Lichess documents no fixed request-per-minute limit, just "one request at
# a time" and a minute-long backoff on 429 -- these constants encode that
# guidance directly rather than guessing at a number of our own.
REQUEST_PACING_SECONDS = 1.0
RATE_LIMIT_BACKOFF_SECONDS = 60
MAX_RATE_LIMIT_RETRIES = 3


class RequestPacer:
    """Enforces REQUEST_PACING_SECONDS between consecutive requests to
    lichess.org, regardless of which caller issues them.

    Pulled out of LichessClient so lichess_scraper.py (a separate module by
    design -- see its docstring) can share the exact same courtesy behavior
    against the same host without duplicating the algorithm. Composition
    over inheritance: the scraper and the API client have no "is-a"
    relationship, they just both need this one piece of behavior, so each
    holds an instance rather than sharing a base class built for it.
    """

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
        self._http = http_client or httpx.Client(base_url=LICHESS_BASE_URL, timeout=30.0)
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
        # Every iteration above returns or raises (the last attempt's 429
        # branch always raises instead of looping again), so this is
        # unreachable -- stated explicitly for the same reason as the
        # equivalent marker in chess_agent._query: it tells a type checker,
        # and a future reader who changes MAX_RATE_LIMIT_RETRIES, that
        # falling out of the loop is a bug, not a valid path.
        raise AssertionError("unreachable: every fetch_study_pgn attempt returns or raises")

    def iter_studies_by_username(self, username: str, limit: int | None = None) -> Iterator[dict]:
        """Yield {id, name, createdAt, updatedAt} for a user's public
        studies, most recently updated first.

        This is a streaming newline-delimited-JSON endpoint (confirmed via
        a live request, not assumed from the docs), and in practice it can
        be slow and return a very large number of results for an active
        account -- a request against Lichess's own official account timed
        out after 15 seconds having already streamed dozens of studies with
        no end in sight. Reading it line by line, and supporting a limit,
        means a caller asking for "the first handful" isn't forced to wait
        for or buffer someone's entire study history first.
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
    """Build a read-only, embeddable URL for a single game -- Lichess's own
    "Embed in your website" feature, confirmed format.
    """
    return f"{LICHESS_BASE_URL}/embed/{game_id}?theme={theme}&bg={bg}"


def study_chapter_embed_url(study_id: str, chapter_id: str, *, bg: str = "dark") -> str:
    """Build a read-only, embeddable URL for one chapter of a study.

    chapter_id is required, not optional: Lichess's study embed format is
    /study/embed/{studyId}/{chapterId} -- there's no confirmed "whole
    study, pick a default chapter" form, so this doesn't pretend one exists
    by defaulting chapter_id to something unverified. Whatever calls this
    is responsible for having already picked a chapter.

    bg defaults to "dark", not Lichess's own "auto" -- confirmed live that
    auto sets data-theme="system" on the embedded page, meaning it follows
    whichever viewer's own OS light/dark preference, not this app's own
    walnut/parchment identity. Most viewers are on light mode by default,
    which renders Lichess's stark white theme directly against a dark
    wood-toned app background. A fixed dark theme looks consistent for
    every viewer regardless of their own system setting, matching the
    deliberate single-theme choice made for the rest of the app's design.
    """
    return f"{LICHESS_BASE_URL}/study/embed/{study_id}/{chapter_id}?bg={bg}"
