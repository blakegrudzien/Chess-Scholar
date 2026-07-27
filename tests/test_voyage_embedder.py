from unittest.mock import MagicMock, patch

import psycopg2
import pytest
import voyageai.error

from src.embeddings.voyage_embedder import embed_pending_chunks


def _mock_connect_with_batches(batches: list[list[tuple[int, str]]]):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.side_effect = [*batches, []]
    conn.cursor.return_value.__enter__.return_value = cursor
    connect = MagicMock(return_value=conn)
    return connect, conn, cursor


def test_embeds_single_batch_and_stops():
    connect, conn, cursor = _mock_connect_with_batches([[(1, "text one"), (2, "text two")]])
    client = MagicMock()
    client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2], [0.3, 0.4]])

    with (
        patch("src.embeddings.voyage_embedder.register_vector") as mock_register,
        patch("src.embeddings.voyage_embedder.execute_values") as mock_execute_values,
    ):
        embedded = embed_pending_chunks(connect, client, model="voyage-4")

    assert embedded == 2
    client.embed.assert_called_once_with(
        ["text one", "text two"], model="voyage-4", input_type="document"
    )
    conn.commit.assert_called_once()

    # One connection for the whole job -- no per-batch reconnect on the happy path.
    assert connect.call_count == 1
    assert mock_register.call_count == 1
    conn.close.assert_called_once()

    # The write is a single batched execute_values call, not one round-trip per row
    # (128 individual UPDATEs per batch was the actual bottleneck in production --
    # ~6s/batch of pure network round-trip time).
    mock_execute_values.assert_called_once()
    call = mock_execute_values.call_args
    sql, rows = call.args[1], call.args[2]
    assert "UPDATE chunks" in sql
    assert "FROM (VALUES %s)" in sql
    assert rows == [(1, [0.1, 0.2]), (2, [0.3, 0.4])]
    assert call.kwargs["page_size"] == 2

    # Only the two SELECTs (real batch + final empty check) go through cur.execute directly.
    assert cursor.execute.call_count == 2
    assert (
        "SELECT chunk_id, text FROM chunks WHERE embedding IS NULL"
        in cursor.execute.call_args_list[0].args[0]
    )


def test_embeds_across_multiple_batches():
    batch1 = [(i, f"text {i}") for i in range(3)]
    batch2 = [(i, f"text {i}") for i in range(3, 5)]
    connect, conn, cursor = _mock_connect_with_batches([batch1, batch2])

    client = MagicMock()
    client.embed.side_effect = [
        MagicMock(embeddings=[[0.0]] * 3),
        MagicMock(embeddings=[[0.0]] * 2),
    ]

    with (
        patch("src.embeddings.voyage_embedder.register_vector"),
        patch("src.embeddings.voyage_embedder.execute_values") as mock_execute_values,
    ):
        embedded = embed_pending_chunks(connect, client)

    assert embedded == 5
    assert client.embed.call_count == 2
    assert mock_execute_values.call_count == 2
    assert conn.commit.call_count == 2
    assert connect.call_count == 1  # single connection reused across batches
    conn.close.assert_called_once()


def test_returns_zero_when_nothing_pending():
    connect, conn, cursor = _mock_connect_with_batches([])
    client = MagicMock()

    with (
        patch("src.embeddings.voyage_embedder.register_vector"),
        patch("src.embeddings.voyage_embedder.execute_values") as mock_execute_values,
    ):
        embedded = embed_pending_chunks(connect, client)

    assert embedded == 0
    client.embed.assert_not_called()
    mock_execute_values.assert_not_called()
    conn.commit.assert_not_called()
    conn.close.assert_called_once()


def test_connection_closed_even_if_batch_processing_fails():
    connect, conn, cursor = _mock_connect_with_batches([[(1, "text one")]])
    client = MagicMock()
    client.embed.side_effect = RuntimeError("non-connection error, should not be retried")

    with (
        patch("src.embeddings.voyage_embedder.register_vector"),
        patch("src.embeddings.voyage_embedder.execute_values"),
        pytest.raises(RuntimeError),
    ):
        embed_pending_chunks(connect, client)

    conn.close.assert_called_once()
    conn.commit.assert_not_called()
    connect.assert_called_once()  # RuntimeError isn't a connection error -- no reconnect attempt


def test_reconnects_and_retries_once_on_operational_error():
    # Same batch is still pending after the failed attempt, since nothing
    # was committed -- the retry re-fetches and re-embeds it successfully.
    batch = [(1, "text one"), (2, "text two")]

    conn1 = MagicMock()
    cursor1 = MagicMock()
    cursor1.fetchall.return_value = batch
    conn1.cursor.return_value.__enter__.return_value = cursor1

    conn2 = MagicMock()
    cursor2 = MagicMock()
    cursor2.fetchall.side_effect = [batch, []]
    conn2.cursor.return_value.__enter__.return_value = cursor2

    connect = MagicMock(side_effect=[conn1, conn2])
    client = MagicMock()
    client.embed.return_value = MagicMock(embeddings=[[0.1], [0.2]])

    with (
        patch("src.embeddings.voyage_embedder.register_vector"),
        patch(
            "src.embeddings.voyage_embedder.execute_values",
            side_effect=[
                psycopg2.OperationalError("server closed the connection unexpectedly"),
                None,
            ],
        ),
    ):
        embedded = embed_pending_chunks(connect, client)

    assert embedded == 2
    assert connect.call_count == 2
    conn1.close.assert_called_once()
    conn1.commit.assert_not_called()
    conn2.close.assert_called_once()
    conn2.commit.assert_called_once()


def test_gives_up_after_too_many_consecutive_failures():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = [(1, "text one")]
    conn.cursor.return_value.__enter__.return_value = cursor

    connect = MagicMock(return_value=conn)
    client = MagicMock()
    client.embed.return_value = MagicMock(embeddings=[[0.1]])

    with (
        patch("src.embeddings.voyage_embedder.register_vector"),
        patch(
            "src.embeddings.voyage_embedder.execute_values",
            side_effect=psycopg2.OperationalError("connection keeps dying"),
        ),
        pytest.raises(psycopg2.OperationalError),
    ):
        embed_pending_chunks(connect, client, max_consecutive_failures=3)

    assert connect.call_count == 4  # initial + 3 retries, then give up


def test_retries_on_voyage_timeout_without_reconnecting_db():
    batch = [(1, "text one")]
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.side_effect = [batch, batch, []]
    conn.cursor.return_value.__enter__.return_value = cursor
    connect = MagicMock(return_value=conn)

    client = MagicMock()
    client.embed.side_effect = [
        voyageai.error.Timeout("Request timed out"),
        MagicMock(embeddings=[[0.1]]),
    ]

    with (
        patch("src.embeddings.voyage_embedder.register_vector"),
        patch("src.embeddings.voyage_embedder.execute_values") as mock_execute_values,
        patch("src.embeddings.voyage_embedder.time.sleep") as mock_sleep,
    ):
        embedded = embed_pending_chunks(connect, client)

    assert embedded == 1
    assert connect.call_count == 1  # Voyage errors don't trigger a DB reconnect
    conn.close.assert_called_once()
    mock_sleep.assert_called_once_with(5)
    mock_execute_values.assert_called_once()


def test_gives_up_after_too_many_voyage_errors():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = [(1, "text one")]
    conn.cursor.return_value.__enter__.return_value = cursor
    connect = MagicMock(return_value=conn)

    client = MagicMock()
    client.embed.side_effect = voyageai.error.ServiceUnavailableError("down")

    with (
        patch("src.embeddings.voyage_embedder.register_vector"),
        patch("src.embeddings.voyage_embedder.execute_values"),
        patch("src.embeddings.voyage_embedder.time.sleep"),
        pytest.raises(voyageai.error.ServiceUnavailableError),
    ):
        embed_pending_chunks(connect, client, max_consecutive_failures=2)

    assert connect.call_count == 1  # no DB reconnect needed for Voyage errors
