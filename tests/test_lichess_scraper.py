import json
from unittest.mock import patch

import httpx
import pytest

from src.recommendation.lichess_scraper import (
    STUDIES_PER_PAGE,
    ScrapedStudyCard,
    StudyChapter,
    fetch_study_chapters,
    iter_studies_by_sort,
)


def _card_html(
    study_id: str,
    title: str = "Some Study",
    likes: int = 5,
    author: str = "someauthor",
    updated_at: str = "2024-01-01T00:00:00.000Z",
    chapters: list[str] | None = None,
    members: list[str] | None = None,
) -> str:
    """Build one study card matching the real markup structure found on
    lichess.org/study/all/{sort}, verified live before this scraper was
    written -- not a guess at the format.
    """
    chapters = chapters if chapters is not None else ["Chapter 1"]
    members = members if members is not None else [author]
    chapters_html = "".join(f'<li class="text">{c}</li>' for c in chapters)
    members_html = "".join(f'<li class="text">{m}</li>' for m in members)
    return f"""
    <div class="study paginated">
      <a class="overlay" href="/study/{study_id}" title="{title}"></a>
      <div class="top">
        <div class="study__icon"><icon data-icon=""></icon></div>
        <div>
          <h2 class="study-name">{title}</h2>
          <span><icon data-icon=""></icon> {likes} • {author} •
            <time class="timeago" datetime="{updated_at}"></time></span>
        </div>
      </div>
      <div class="body">
        <ol class="chapters">{chapters_html}</ol>
        <ol class="members">{members_html}</ol>
      </div>
    </div>
    """


def _page_html(cards: list[str]) -> str:
    return f'<div class="studies infinite-scroll">{"".join(cards)}</div>'


def _client_with_handler(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://lichess.org")


def test_iter_studies_by_sort_parses_card_fields():
    page = _page_html(
        [
            _card_html(
                "abc123",
                title="Caro-Kann Basics",
                likes=42,
                author="coach_kim",
                updated_at="2021-02-03T16:32:17.607Z",
                chapters=["Intro", "Advance Variation"],
                members=["coach_kim", "helper"],
            )
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/study/all/popular"
        return httpx.Response(200, text=page)

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as http_client:
            results = list(iter_studies_by_sort("popular", http_client=http_client))

    assert results == [
        ScrapedStudyCard(
            study_id="abc123",
            title="Caro-Kann Basics",
            author="coach_kim",
            likes=42,
            updated_at="2021-02-03T16:32:17.607Z",
            chapter_titles=["Intro", "Advance Variation"],
            member_usernames=["coach_kim", "helper"],
        )
    ]


def test_iter_studies_by_sort_paginates_until_a_short_page():
    full_page = _page_html([_card_html(f"id{i}") for i in range(STUDIES_PER_PAGE)])
    short_page = _page_html([_card_html("last")])
    requested_pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page")
        requested_pages.append(page)
        return httpx.Response(200, text=full_page if page == "1" else short_page)

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as http_client:
            results = list(iter_studies_by_sort("newest", http_client=http_client))

    assert requested_pages == ["1", "2"]
    assert len(results) == STUDIES_PER_PAGE + 1
    assert results[-1].study_id == "last"


def test_iter_studies_by_sort_treats_a_400_response_as_the_end_of_the_listing():
    """Real, reproduced failure: requesting a large --popular-count walked
    past the "popular" listing's actual depth, and Lichess responded 400
    Bad Request to that page instead of an empty 200 the way every other
    sort/page naturally ends -- this used to propagate as an unhandled
    HTTPStatusError and abort the whole collection run.
    """
    full_page = _page_html([_card_html(f"id{i}") for i in range(STUDIES_PER_PAGE)])
    requested_pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page")
        requested_pages.append(page)
        if page == "3":
            return httpx.Response(400, text="Bad Request")
        return httpx.Response(200, text=full_page)

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as http_client:
            results = list(iter_studies_by_sort("popular", http_client=http_client))

    assert requested_pages == ["1", "2", "3"]
    assert len(results) == STUDIES_PER_PAGE * 2


def test_iter_studies_by_sort_stops_at_max_pages():
    full_page = _page_html([_card_html(f"id{i}") for i in range(STUDIES_PER_PAGE)])
    requested_pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_pages.append(request.url.params.get("page"))
        return httpx.Response(200, text=full_page)

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as http_client:
            results = list(iter_studies_by_sort("hot", max_pages=1, http_client=http_client))

    assert requested_pages == ["1"]
    assert len(results) == STUDIES_PER_PAGE


def test_iter_studies_by_sort_rejects_invalid_sort():
    with pytest.raises(ValueError, match="sort must be one of"):
        list(iter_studies_by_sort("trending"))


def test_iter_studies_by_sort_skips_a_malformed_card_without_failing_the_page():
    good_card = _card_html("good1", title="Fine")
    malformed_card = '<div class="study paginated"><a class="overlay" href="/study/bad1"></a></div>'
    page = _page_html([good_card, malformed_card])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=page)

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as http_client:
            results = list(iter_studies_by_sort("updated", http_client=http_client))

    assert [r.study_id for r in results] == ["good1"]


def _study_page_html(chapters: list[dict]) -> str:
    """A minimal page matching the real structure verified live: a
    `<script type="application/json" id="page-init-data">` block holding
    study.chapters, surrounded by markup this scraper never reads.
    """
    payload = json.dumps({"study": {"id": "abc123", "chapters": chapters}})
    script = f'<script type="application/json" id="page-init-data">{payload}</script>'
    return f"<html><body>{script}</body></html>"


def test_fetch_study_chapters_parses_id_and_name():
    page = _study_page_html(
        [
            {"id": "gIOq5Mk2", "name": "Introduction", "orientation": "black"},
            {"id": "BYjz8j5O", "name": "Main Line", "orientation": "black"},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/study/abc123"
        return httpx.Response(200, text=page)

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as http_client:
            chapters = fetch_study_chapters("abc123", http_client=http_client)

    assert chapters == [
        StudyChapter(chapter_id="gIOq5Mk2", name="Introduction"),
        StudyChapter(chapter_id="BYjz8j5O", name="Main Line"),
    ]


def test_fetch_study_chapters_returns_empty_list_when_data_block_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>no data here</body></html>")

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as http_client:
            chapters = fetch_study_chapters("abc123", http_client=http_client)

    assert chapters == []


def test_fetch_study_chapters_returns_empty_list_on_malformed_json():
    page = '<script type="application/json" id="page-init-data">{not valid json</script>'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=page)

    with patch("src.recommendation.lichess_client.REQUEST_PACING_SECONDS", 0):
        with _client_with_handler(handler) as http_client:
            chapters = fetch_study_chapters("abc123", http_client=http_client)

    assert chapters == []
