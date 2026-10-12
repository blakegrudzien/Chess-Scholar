"""Generate the saved answers for the app's example questions: run each one
through the real chat agent and resource lookup, and store the result in
the example_answers table (see src/ui/example_answers.py).

Rerun after changing the system prompt, the tools or the corpus, since the
saved answers don't change on their own.

Run as a module from the repo root:
    .venv/bin/python -m scripts.generate_example_answers
"""

from __future__ import annotations

import logging
import time

import anthropic

from src.agent.chess_agent import ask
from src.embeddings.voyage_embedder import get_voyage_client
from src.engine.engine_pool import EnginePool
from src.engine.stockfish_eval import get_engine_path
from src.ingestion.db_loader import get_connection_pool, query_with_retry
from src.recommendation.lichess_client import RequestPacer, default_http_client
from src.recommendation.pipeline import recommend_resources
from src.ui.chat import _lookup_context
from src.ui.example_answers import EXAMPLE_QUESTIONS, save_cached_answer

logger = logging.getLogger(__name__)


def _answer_with_positions(question: str, db_pool, engine_pool, voyage, client):
    """The agent's answer, the positions its tools touched (recorded as the
    app records them: each new position once, in order) and its timeline:
    when each status line appeared, when the answer began streaming, and the
    total time, for the app to replay at a live pace."""
    touched_fens: list[tuple[str, str | None]] = []
    status_lines: list[dict] = []
    chunk_times: list[float] = []
    started = time.monotonic()

    def on_position(fen: str, *, label: str | None = None) -> None:
        if not touched_fens or touched_fens[-1][0] != fen:
            touched_fens.append((fen, label))

    def on_step(text: str) -> None:
        status_lines.append({"seconds": round(time.monotonic() - started, 2), "text": text})

    def on_chunk(_delta: str) -> None:
        chunk_times.append(time.monotonic() - started)

    answer = ask(
        question,
        db_pool,
        engine_pool,
        voyage,
        client=client,
        on_position=on_position,
        on_step=on_step,
        on_chunk=on_chunk,
    )
    total = time.monotonic() - started
    last_step = status_lines[-1]["seconds"] if status_lines else 0.0
    answer_start = next((t for t in chunk_times if t > last_step), total)
    timeline = {
        "status_lines": status_lines,
        "answer_start_seconds": round(answer_start, 2),
        "total_seconds": round(total, 2),
    }
    return answer, touched_fens, timeline


def main() -> None:
    db_pool = get_connection_pool()
    engine_pool = EnginePool(get_engine_path(), size=2)
    voyage = get_voyage_client()
    client = anthropic.Anthropic()
    http_client, pacer = default_http_client(), RequestPacer()
    try:
        for question in EXAMPLE_QUESTIONS:
            answer, touched_fens, timeline = _answer_with_positions(
                question, db_pool, engine_pool, voyage, client
            )
            history = [("user", question, None, []), ("assistant", answer, None, touched_fens)]
            recommendations = recommend_resources(
                _lookup_context(history),
                db_pool,
                voyage,
                client=client,
                http_client=http_client,
                pacer=pacer,
            )
            query_with_retry(
                db_pool,
                save_cached_answer,
                question,
                answer,
                touched_fens,
                recommendations,
                timeline,
            )
            logger.info(
                "Saved %r: %.0fs, %d chars, %d diagram(s), %d recommendation(s)",
                question,
                timeline["total_seconds"],
                len(answer),
                len(touched_fens),
                len(recommendations),
            )
    finally:
        engine_pool.close()
        db_pool.closeall()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
