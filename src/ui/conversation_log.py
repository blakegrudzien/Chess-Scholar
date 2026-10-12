"""Saves each question and answer to the conversation_log table, so real
usage can be reviewed later; chat history otherwise lasts only as long as
the browser session.
"""

from __future__ import annotations

import logging

import psycopg2.extensions
import psycopg2.pool

from src.ingestion.db_loader import query_with_retry

logger = logging.getLogger(__name__)


def log_conversation(
    conn: psycopg2.extensions.connection, question: str, fen_context: str | None, answer: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO conversation_log (question, fen_context, answer) VALUES (%s, %s, %s)",
            (question, fen_context, answer),
        )
    conn.commit()


def log_conversation_best_effort(
    db_pool: psycopg2.pool.ThreadedConnectionPool,
    question: str,
    fen_context: str | None,
    answer: str,
) -> None:
    """log_conversation that never raises. It runs after the answer is
    already on screen, and losing one log row is better than turning a
    good answer into an error page, hence the broad except.
    """
    try:
        query_with_retry(db_pool, log_conversation, question, fen_context, answer)
    except Exception:
        logger.exception("Failed to log conversation for later eval review")
