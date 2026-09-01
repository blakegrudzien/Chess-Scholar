from unittest.mock import MagicMock, patch

from src.recommendation.lichess_scraper import StudyChapter
from src.recommendation.pipeline import (
    ChessbaseGameRecommendation,
    LichessStudyRecommendation,
    _SessionState,
    build_tools,
    recommend_resources,
)
from src.recommendation.study_search import StudyResult
from src.search.structured_search import NarrativeGameCandidate


def _build_tools():
    db_pool = MagicMock()
    conn = MagicMock()
    db_pool.getconn.return_value = conn
    voyage_client = MagicMock()
    state = _SessionState()
    tools = {t.name: t for t in build_tools(db_pool, voyage_client, state)}
    return tools, db_pool, conn, voyage_client, state


def test_search_lichess_studies_formats_results_and_populates_state():
    tools, db_pool, conn, voyage_client, state = _build_tools()
    results = [
        StudyResult(
            study_id="abc123",
            title="Sicilian Dragon",
            likes=10881,
            quality_probability=0.977,
            distance=0.1,
        )
    ]
    with patch("src.recommendation.pipeline.search_studies", return_value=results) as mock_fn:
        text = tools["search_lichess_studies"]("sicilian dragon")

    mock_fn.assert_called_once_with(conn, voyage_client, "sicilian dragon", limit=5)
    db_pool.putconn.assert_called_once_with(conn)
    assert "abc123" in text
    assert "Sicilian Dragon" in text
    assert "10881 likes" in text
    assert state.studies_by_id == {"abc123": "Sicilian Dragon"}


def test_search_lichess_studies_reports_no_matches():
    tools, *_ = _build_tools()
    with patch("src.recommendation.pipeline.search_studies", return_value=[]):
        text = tools["search_lichess_studies"]("obscure query")
    assert "No matching studies" in text


def test_get_lichess_study_chapters_formats_and_populates_state():
    tools, *_, state = _build_tools()
    chapters = [
        StudyChapter(chapter_id="gIOq5Mk2", name="Introduction"),
        StudyChapter(chapter_id="BYjz8j5O", name="Main Line"),
    ]
    with patch(
        "src.recommendation.pipeline.fetch_study_chapters", return_value=chapters
    ) as mock_fn:
        text = tools["get_lichess_study_chapters"]("abc123")

    mock_fn.assert_called_once_with("abc123")
    assert "gIOq5Mk2" in text
    assert "Introduction" in text
    assert state.chapters_by_study_id == {
        "abc123": {"gIOq5Mk2": "Introduction", "BYjz8j5O": "Main Line"}
    }


def test_get_lichess_study_chapters_reports_none_found():
    tools, *_ = _build_tools()
    with patch("src.recommendation.pipeline.fetch_study_chapters", return_value=[]):
        text = tools["get_lichess_study_chapters"]("abc123")
    assert "No chapters found" in text


def test_find_chessbase_game_formats_and_populates_state():
    tools, db_pool, conn, _, state = _build_tools()
    candidate = NarrativeGameCandidate(
        game_id="g1",
        white="Carlsen",
        black="Nepomniachtchi",
        event="World Championship",
        year=2021,
        eco_code="C88",
        result="1-0",
        annotation_chunk_count=29,
    )
    with patch(
        "src.recommendation.pipeline.select_narrative_game", return_value=candidate
    ) as mock_fn:
        text = tools["find_chessbase_game"](["C88", "C89"])

    mock_fn.assert_called_once_with(conn, ["C88", "C89"])
    db_pool.putconn.assert_called_once_with(conn)
    assert "g1" in text
    assert "Carlsen vs Nepomniachtchi" in text
    assert "29 comments" in text
    assert state.games_by_id == {"g1": candidate}


def test_find_chessbase_game_reports_no_match():
    tools, *_ = _build_tools()
    with patch("src.recommendation.pipeline.select_narrative_game", return_value=None):
        text = tools["find_chessbase_game"](["Z99"])
    assert "No chessbase game found" in text


def test_recommend_lichess_study_records_a_recommendation():
    tools, *_, state = _build_tools()
    state.studies_by_id["abc123"] = "Sicilian Dragon"
    state.chapters_by_study_id["abc123"] = {"gIOq5Mk2": "Introduction"}

    result = tools["recommend_lichess_study"]("abc123", "gIOq5Mk2", "Covers the main ideas.")

    assert result == "Recorded."
    assert state.recommendations == [
        LichessStudyRecommendation(
            study_id="abc123",
            study_title="Sicilian Dragon",
            chapter_id="gIOq5Mk2",
            chapter_name="Introduction",
            embed_url="https://lichess.org/study/embed/abc123/gIOq5Mk2?bg=dark",
            blurb="Covers the main ideas.",
        )
    ]


def test_recommend_lichess_study_rejects_a_study_not_looked_up_first():
    tools, *_, state = _build_tools()
    result = tools["recommend_lichess_study"]("never_searched", "some_chapter", "blurb")
    assert "look it up" in result
    assert state.recommendations == []


def test_recommend_chessbase_game_records_a_recommendation():
    tools, db_pool, conn, _, state = _build_tools()
    candidate = NarrativeGameCandidate(
        game_id="g1",
        white="Carlsen",
        black="Nepomniachtchi",
        event="World Championship",
        year=2021,
        eco_code="C88",
        result="1-0",
        annotation_chunk_count=29,
    )
    state.games_by_id["g1"] = candidate

    with patch(
        "src.recommendation.pipeline.game_moves_as_pgn", return_value="1. e4 e5 *"
    ) as mock_fn:
        result = tools["recommend_chessbase_game"]("g1", "A classic Ruy Lopez battle.")

    mock_fn.assert_called_once_with(conn, "g1")
    assert result == "Recorded."
    assert state.recommendations == [
        ChessbaseGameRecommendation(
            game_id="g1",
            white="Carlsen",
            black="Nepomniachtchi",
            event="World Championship",
            pgn="1. e4 e5 *",
            blurb="A classic Ruy Lopez battle.",
        )
    ]


def test_recommend_chessbase_game_rejects_a_game_not_looked_up_first():
    tools, *_, state = _build_tools()
    result = tools["recommend_chessbase_game"]("never_looked_up", "blurb")
    assert "look it up" in result
    assert state.recommendations == []


def test_recommend_resources_returns_empty_list_when_nothing_recommended():
    with patch("src.recommendation.pipeline.anthropic.Anthropic") as mock_anthropic:
        client = mock_anthropic.return_value
        client.beta.messages.tool_runner.return_value = []
        result = recommend_resources(
            "what's the best way to learn chess?", MagicMock(), MagicMock()
        )

    assert result == []


def test_recommend_resources_returns_whatever_the_model_recommended_during_the_loop():
    # tool_runner drives tool invocation internally in real usage; this
    # simulates that by having the mocked runner call one of the tools it
    # was actually handed, the same way the real SDK would when the model
    # decides to call it.
    client = MagicMock()

    def fake_tool_runner(*, tools, **kwargs):
        search_tool = next(t for t in tools if t.name == "search_lichess_studies")
        with patch(
            "src.recommendation.pipeline.search_studies",
            return_value=[
                StudyResult(
                    study_id="abc123",
                    title="Sicilian Dragon",
                    likes=10881,
                    quality_probability=0.977,
                    distance=0.1,
                )
            ],
        ):
            search_tool("sicilian dragon")

        chapters_tool = next(t for t in tools if t.name == "get_lichess_study_chapters")
        with patch(
            "src.recommendation.pipeline.fetch_study_chapters",
            return_value=[StudyChapter(chapter_id="gIOq5Mk2", name="Introduction")],
        ):
            chapters_tool("abc123")

        recommend_tool = next(t for t in tools if t.name == "recommend_lichess_study")
        recommend_tool("abc123", "gIOq5Mk2", "A strong, well-annotated Dragon repertoire.")
        return []

    client.beta.messages.tool_runner.side_effect = fake_tool_runner

    result = recommend_resources(
        "what's a good sicilian dragon resource?", MagicMock(), MagicMock(), client
    )

    assert result == [
        LichessStudyRecommendation(
            study_id="abc123",
            study_title="Sicilian Dragon",
            chapter_id="gIOq5Mk2",
            chapter_name="Introduction",
            embed_url="https://lichess.org/study/embed/abc123/gIOq5Mk2?bg=dark",
            blurb="A strong, well-annotated Dragon repertoire.",
        )
    ]
