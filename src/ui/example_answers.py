"""Saved answers for the app's example questions, replayed at a live pace
without calling the API.

The answers come from the real agent (scripts/generate_example_answers.py)
and are stored in the example_answers table rather than the repo, since an
answer can paraphrase the licensed annotations it drew on. Rerun the script
after changing the prompt or the corpus.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import psycopg2.extensions
import streamlit as st

from src.ingestion.db_loader import query_with_retry
from src.recommendation.pipeline import (
    ChessbaseGameRecommendation,
    LichessStudyRecommendation,
    Recommendation,
)
from src.ui.resources import get_db_pool

logger = logging.getLogger(__name__)

# One per layer, so the four together show the routing.
EXAMPLE_QUESTIONS = [
    "How should White meet the Sicilian Defense?",  # Layers 1 + 2: stats and plans
    "Evaluate 1. e4 e5 2. Qh5 for White",  # Layer 3: engine
    "What's the plan behind an isolated queen pawn?",  # Layer 2: concepts
    "Where does White's knight usually end up in the Najdorf?",  # Layer 1: statistics
]

_RECOMMENDATION_TYPES = {
    "lichess_study": LichessStudyRecommendation,
    "chessbase_game": ChessbaseGameRecommendation,
}


# Saved answers replay at this fraction of their original time: faster than
# a live answer, but still paced like one.
REPLAY_SPEED = 0.65

# How many pieces the saved answer streams in.
_REPLAY_ANSWER_PIECES = 40


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    touched_fens: list[tuple[str, str | None]]
    recommendations: list[Recommendation]
    # {"status_lines": [{"seconds", "text"}], "answer_start_seconds",
    # "total_seconds"} from the original generation; None for older rows.
    timeline: dict | None = None


def _recommendation_to_json(rec: Recommendation) -> dict:
    kind = next(k for k, cls in _RECOMMENDATION_TYPES.items() if isinstance(rec, cls))
    return {"kind": kind, **asdict(rec)}


def _recommendation_from_json(data: dict) -> Recommendation:
    fields = dict(data)
    return _RECOMMENDATION_TYPES[fields.pop("kind")](**fields)


def save_cached_answer(
    conn: psycopg2.extensions.connection,
    question: str,
    answer: str,
    touched_fens: list[tuple[str, str | None]],
    recommendations: list[Recommendation],
    timeline: dict,
) -> None:
    """Store (or replace) the saved answer for `question`."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO example_answers
                (question, answer, touched_fens, recommendations, timeline)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (question) DO UPDATE SET
                answer = EXCLUDED.answer,
                touched_fens = EXCLUDED.touched_fens,
                recommendations = EXCLUDED.recommendations,
                timeline = EXCLUDED.timeline,
                generated_at = now()
            """,
            (
                question,
                answer,
                json.dumps(touched_fens),
                json.dumps([_recommendation_to_json(r) for r in recommendations]),
                json.dumps(timeline),
            ),
        )
    conn.commit()


def load_cached_answers(conn: psycopg2.extensions.connection) -> dict[str, CachedAnswer]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT question, answer, touched_fens, recommendations, timeline FROM example_answers"
        )
        rows = cur.fetchall()
    return {
        question: CachedAnswer(
            answer=answer,
            touched_fens=[(fen, label) for fen, label in touched_fens],
            recommendations=[_recommendation_from_json(r) for r in recommendations],
            timeline=timeline,
        )
        for question, answer, touched_fens, recommendations, timeline in rows
    }


# Read once and reused for ten minutes. A read that fails raises, so the
# failure isn't cached and the next click tries again.
@st.cache_data(ttl=600, show_spinner=False)
def _load_all_cached() -> dict[str, CachedAnswer]:
    return query_with_retry(get_db_pool(), load_cached_answers)


def cached_example_answer(question: str, *, fen_context: str | None) -> CachedAnswer | None:
    """The saved answer for an example question asked without a board
    position, or None, in which case the question goes to the agent. A
    cache that can't be read, or a row with no timeline to replay, counts
    as a miss."""
    if fen_context is not None or question not in EXAMPLE_QUESTIONS:
        return None
    try:
        cached = _load_all_cached().get(question)
    except Exception:
        # Any failure here only costs the shortcut; the agent still answers.
        logger.warning("Couldn't read cached example answers", exc_info=True)
        return None
    return cached if cached is not None and cached.timeline is not None else None


def replay_cached_answer(
    cached: CachedAnswer,
    on_step: Callable[[str], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
    on_position: Callable[..., None] | None = None,
    *,
    speed: float = REPLAY_SPEED,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Stand in for the agent: emit a saved answer through the same
    callbacks, on its original timeline scaled by `speed`. Each status line
    arrives when it did originally, then the answer streams in until the
    original end time."""
    timeline = cached.timeline or {}
    started = clock()

    def wait_until(original_seconds: float) -> None:
        remaining = started + original_seconds * speed - clock()
        if remaining > 0:
            sleep(remaining)

    for line in timeline.get("status_lines", []):
        wait_until(line["seconds"])
        if on_chunk is not None:
            on_chunk(line["text"])
        if on_step is not None:
            on_step(line["text"])

    answer_start = timeline.get("answer_start_seconds", 0.0)
    answer_end = max(timeline.get("total_seconds", answer_start), answer_start)
    pieces = _split_evenly(cached.answer, _REPLAY_ANSWER_PIECES)
    for i, piece in enumerate(pieces):
        wait_until(answer_start + (answer_end - answer_start) * i / len(pieces))
        if on_chunk is not None:
            on_chunk(piece)
    wait_until(answer_end)

    if on_position is not None:
        for fen, label in cached.touched_fens:
            on_position(fen, label=label)
    return cached.answer


def _split_evenly(text: str, pieces: int) -> list[str]:
    """`text` cut at word boundaries into about `pieces` parts that rejoin
    exactly."""
    words = re.findall(r"\s*\S+\s*", text) or [text]
    size = max(1, -(-len(words) // pieces))
    return ["".join(words[i : i + size]) for i in range(0, len(words), size)]
