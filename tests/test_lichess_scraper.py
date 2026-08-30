from unittest.mock import patch

import httpx
import pytest

from src.recommendation.lichess_scraper import (
    STUDIES_PER_PAGE,
    ScrapedStudyCard,
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
