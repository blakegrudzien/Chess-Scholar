import json
from unittest.mock import patch

import httpx
import pytest

from src.recommendation.lichess_client import (
    LichessClient,
    game_embed_url,
    study_chapter_embed_url,
)


def _client_with_handler(handler) -> LichessClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="https://lichess.org")
    return LichessClient(http_client=http_client)


def test_fetch_study_pgn_returns_response_text():
    requested_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(200, text='[Event "Test"]\n1. e4 e5 *')

    with _client_with_handler(handler) as client:
        pgn = client.fetch_study_pgn("abc123")

    assert pgn == '[Event "Test"]\n1. e4 e5 *'
    assert requested_paths == ["/api/study/abc123.pgn"]


def test_fetch_study_pgn_retries_on_429_then_succeeds():
    responses = iter([httpx.Response(429), httpx.Response(200, text="ok")])

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    with (
        patch("src.recommendation.lichess_client.RATE_LIMIT_BACKOFF_SECONDS", 0),
        patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0),
        _client_with_handler(handler) as client,
    ):
        pgn = client.fetch_study_pgn("abc123")

    assert pgn == "ok"


def test_fetch_study_pgn_gives_up_after_max_retries():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(429)

    with (
        patch("src.recommendation.lichess_client.RATE_LIMIT_BACKOFF_SECONDS", 0),
        patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0),
        patch("src.recommendation.lichess_client.MAX_RATE_LIMIT_RETRIES", 2),
        _client_with_handler(handler) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        client.fetch_study_pgn("abc123")

    assert call_count == 2


def test_iter_studies_by_username_parses_ndjson_lines():
    studies = [
        {"id": "aaa111", "name": "First study", "createdAt": 1, "updatedAt": 2},
        {"id": "bbb222", "name": "Second study", "createdAt": 3, "updatedAt": 4},
    ]
    body = "\n".join(json.dumps(s) for s in studies)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/study/by/someuser"
        return httpx.Response(200, text=body)

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as client:
            result = list(client.iter_studies_by_username("someuser"))

    assert result == studies


def test_iter_studies_by_username_respects_limit():
    studies = [
        {"id": str(i), "name": f"Study {i}", "createdAt": i, "updatedAt": i} for i in range(10)
    ]
    body = "\n".join(json.dumps(s) for s in studies)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as client:
            result = list(client.iter_studies_by_username("someuser", limit=3))

    assert result == studies[:3]


def test_pace_sleeps_only_when_a_request_follows_closely_behind_the_last():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    # _pace() calls time.monotonic() once on the very first call (nothing to
    # compare against yet) and twice on every call after (once to compute
    # elapsed, once to record the new last-request time) -- three values
    # covers both calls: 100.0 for the first, 100.2 (0.2s later) for the
    # second call's elapsed check, then whatever for its own record-keeping.
    fake_clock = iter([100.0, 100.2, 101.0])
    with (
        patch("src.recommendation.lichess_client.time.monotonic", lambda: next(fake_clock)),
        patch("src.recommendation.lichess_client.time.sleep") as mock_sleep,
        _client_with_handler(handler) as client,
    ):
        client.fetch_study_pgn("first")
        client.fetch_study_pgn("second")

    mock_sleep.assert_called_once()
    (slept_seconds,) = mock_sleep.call_args.args
    assert slept_seconds == pytest.approx(0.8, abs=0.01)  # 1.0s pacing minus 0.2s elapsed


def test_game_embed_url_format():
    url = game_embed_url("abcd1234")
    assert url == "https://lichess.org/embed/abcd1234?theme=auto&bg=auto"


def test_study_chapter_embed_url_format():
    url = study_chapter_embed_url("1sGpLrkI", "JinVss1N", bg="dark")
    assert url == "https://lichess.org/study/embed/1sGpLrkI/JinVss1N?bg=dark"
