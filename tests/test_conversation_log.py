import logging
from unittest.mock import MagicMock

import psycopg2

from src.ui.conversation_log import log_conversation, log_conversation_best_effort


def test_log_conversation_inserts_question_fen_and_answer():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value

    log_conversation(
        conn,
        "How should White meet the Sicilian?",
        "8/8/8/8/8/8/8/4K2k w - - 0 1",
        "Play 3.d4.",
    )

    cursor.execute.assert_called_once()
    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO conversation_log" in sql
    assert params == (
        "How should White meet the Sicilian?",
        "8/8/8/8/8/8/8/4K2k w - - 0 1",
        "Play 3.d4.",
    )
    conn.commit.assert_called_once()


def test_log_conversation_allows_a_null_fen_context():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value

    log_conversation(conn, "What is the Sicilian?", None, "A defense starting 1.e4 c5.")

    params = cursor.execute.call_args.args[1]
    assert params[1] is None


def test_log_conversation_best_effort_inserts_via_the_pool():
    db_pool = MagicMock()
    conn = MagicMock()
    db_pool.getconn.return_value = conn
    cursor = conn.cursor.return_value.__enter__.return_value

    log_conversation_best_effort(db_pool, "A question", None, "An answer")

    cursor.execute.assert_called_once()
    db_pool.putconn.assert_called_once_with(conn)


def test_log_conversation_best_effort_swallows_a_failure_instead_of_raising(caplog):
    """Regression test for the actual reason this function exists: by the
    time it's called (see chat.py's call site), the user's real answer is
    already on screen. A DB hiccup logging that answer for later eval
    review must never turn a successful answer into a crashed page -- the
    broad except is deliberate here, not an oversight, and is documented as
    such in the function's own docstring.
    """
    db_pool = MagicMock()
    db_pool.getconn.side_effect = psycopg2.OperationalError("connection refused")

    with caplog.at_level(logging.ERROR):
        log_conversation_best_effort(db_pool, "A question", None, "An answer")  # must not raise

    assert "Failed to log conversation" in caplog.text


def test_log_conversation_best_effort_swallows_a_missing_table_too(caplog):
    """A distinct failure mode from the connection error above: the table
    genuinely doesn't exist yet (a fresh deploy before the schema migration
    has run). Not a retryable connection problem -- query_with_retry
    re-raises it immediately -- but still must not reach the caller.
    """
    db_pool = MagicMock()
    conn = MagicMock()
    db_pool.getconn.return_value = conn
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = psycopg2.errors.UndefinedTable("relation does not exist")

    with caplog.at_level(logging.ERROR):
        log_conversation_best_effort(db_pool, "A question", None, "An answer")  # must not raise

    assert "Failed to log conversation" in caplog.text
