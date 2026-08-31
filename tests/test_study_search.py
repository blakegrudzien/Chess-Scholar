from unittest.mock import MagicMock, patch

from src.recommendation.study_search import search_studies


def test_search_studies_embeds_query_and_ranks_by_distance():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        ("abc123", "Sicilian Dragon", 10881, 0.977, 0.12),
        ("def456", "Queen's Gambit", 8393, 0.95, 0.34),
    ]
    conn.cursor.return_value.__enter__.return_value = cursor

    client = MagicMock()
    client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2, 0.3]])

    with patch("src.recommendation.study_search.register_vector") as mock_register:
        results = search_studies(conn, client, "sicilian dragon ideas", model="voyage-4", limit=5)

    mock_register.assert_called_once_with(conn)
    client.embed.assert_called_once_with(
        ["sicilian dragon ideas"], model="voyage-4", input_type="query"
    )

    assert len(results) == 2
    assert results[0].study_id == "abc123"
    assert results[0].title == "Sicilian Dragon"
    assert results[0].likes == 10881
    assert results[0].quality_probability == 0.977
    assert results[0].distance == 0.12
    assert results[1].study_id == "def456"

    sql, params = cursor.execute.call_args.args
    assert "ORDER BY embedding <=> %s::vector" in sql
    assert "FROM lichess_study_cache" in sql
    assert params == ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 5)


def test_search_studies_returns_empty_list_when_no_matches():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    conn.cursor.return_value.__enter__.return_value = cursor

    client = MagicMock()
    client.embed.return_value = MagicMock(embeddings=[[0.0]])

    with patch("src.recommendation.study_search.register_vector"):
        results = search_studies(conn, client, "nonexistent concept")

    assert results == []
