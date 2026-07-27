from unittest.mock import MagicMock, patch

from src.rag.vector_search import search_chunks


def test_search_chunks_embeds_query_and_ranks_by_distance():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        (
            "Bxb4!?: Accepting the gambit.",
            "game_annotation",
            "abc123",
            None,
            None,
            2021,
            "C51",
            0.12,
        ),
        ("A classic Slav plan.", "book", None, "My Book", "Jane Doe", 2019, "D12", 0.34),
    ]
    conn.cursor.return_value.__enter__.return_value = cursor

    client = MagicMock()
    client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2, 0.3]])

    with patch("src.rag.vector_search.register_vector") as mock_register:
        results = search_chunks(conn, client, "gambit acceptance", model="voyage-4", limit=5)

    mock_register.assert_called_once_with(conn)
    client.embed.assert_called_once_with(
        ["gambit acceptance"], model="voyage-4", input_type="query"
    )

    assert len(results) == 2
    assert results[0].text == "Bxb4!?: Accepting the gambit."
    assert results[0].source_type == "game_annotation"
    assert results[0].game_id == "abc123"
    assert results[0].distance == 0.12
    assert results[1].source_title == "My Book"
    assert results[1].distance == 0.34

    sql, params = cursor.execute.call_args.args
    assert "ORDER BY embedding <=> %s::vector" in sql
    assert "WHERE embedding IS NOT NULL" in sql
    assert params == ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 5)


def test_search_chunks_returns_empty_list_when_no_matches():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    conn.cursor.return_value.__enter__.return_value = cursor

    client = MagicMock()
    client.embed.return_value = MagicMock(embeddings=[[0.0]])

    with patch("src.rag.vector_search.register_vector"):
        results = search_chunks(conn, client, "nonexistent concept")

    assert results == []
