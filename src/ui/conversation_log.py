"""Logs every real question asked in the deployed app to conversation_log,
so real usage survives past one browser session for later eval work (see
CLAUDE.md's testing/eval notes) -- st.session_state.chat_history is
in-memory per session only, and the structured logs elsewhere in this
project (chess_agent.py's per-turn timing, chat.py's logger.exception calls
on a failed API call) capture operational signal, not the actual question
and answer text a later human review would need.
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
    """log_conversation, but a failure here must never take the chat flow
    down with it -- the caller (chat.py) always calls this after the user's
    real answer is already on screen, so losing one row of eval data to a
    momentary DB hiccup is a fair trade against turning a successful answer
    into a crashed page. The broad except is deliberate, the same reasoning
    engine_pool._quit_all documents for its own wide catch: this function's
    entire job is "log if at all possible," not "log or raise," so unlike
    the rest of this project's usual narrow, specific exception handling,
    there is no failure mode here where letting the exception propagate
    would be the more correct behavior. Caught and logged here, not left to
    the caller to remember to wrap, so "best effort" is actually guaranteed
    by this function rather than by every call site's own discipline.
    """
    try:
        query_with_retry(db_pool, log_conversation, question, fen_context, answer)
    except Exception:
        logger.exception("Failed to log conversation for later eval review")
