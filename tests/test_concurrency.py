"""Concurrency regression tests for the Stockfish engine pool
(src/engine/engine_pool.py).

Before this pool existed, app.py handed every Streamlit session the SAME
cached chess.engine.SimpleEngine (@st.cache_resource singleton). python-chess
serializes concurrent analyse() calls on one engine process (only one UCI
command can be in flight at a time), so a second user's request didn't even
start until the first user's full search finished. These tests drive two/
three simulated users through the real ask() entry point at the same time,
sharing one real EnginePool exactly as app.py now does -- only the Anthropic
model call is faked (deterministic, no API cost), scripted to make the same
tool call the real model makes for a position-evaluation question.

Confirms two things about the fix:
1. Concurrent users up to the pool size get genuine parallelism (each
   finishes close to solo-call time, not ~Nx it).
2. A user beyond the pool size gets an immediate "too many concurrent users"
   answer instead of hanging or erroring badly.
"""

from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import chess
import chess.engine
import pytest

from src.agent.chess_agent import ask
from src.engine.engine_pool import EnginePool

STOCKFISH_PATH = shutil.which("stockfish")
pytestmark = pytest.mark.skipif(STOCKFISH_PATH is None, reason="requires a local Stockfish binary")

POOL_SIZE = 2

# How much slower than a solo baseline a concurrent user's answer is allowed
# to be. >1x because some queuing is unavoidable; comfortably below 2x, which
# is what full serialization on a shared engine looks like.
SLA_MULTIPLIER = 1.6

# Same position for every simulated user: the property under test is whether
# N equal-cost requests run in parallel, not whether different positions cost
# different amounts of search time (they do -- verified independently, e.g.
# the position after 1.e4 takes ~2.4x as long to search as the start
# position at depth 16, which would otherwise swamp the timing comparison).
EVAL_FEN = chess.Board().fen()


@pytest.fixture
def engine_pool():
    pool = EnginePool(STOCKFISH_PATH, size=POOL_SIZE)
    yield pool
    pool.close()


def _fake_client(fen: str) -> MagicMock:
    """Stands in for the Anthropic client: skips the real model call (no API
    cost, deterministic) but performs the same tool call the real model makes
    for a position-evaluation question -- so this still exercises the real
    evaluate_chess_position tool and the real engine pool underneath it.
    """
    client = MagicMock()

    def fake_tool_runner(*, tools, **kwargs):
        by_name = {t.name: t for t in tools}
        result_text = by_name["evaluate_chess_position"](fen)
        message = MagicMock()
        message.content = [MagicMock(type="text", text=result_text)]
        return [message]

    client.beta.messages.tool_runner.side_effect = fake_tool_runner
    return client


def _ask_for_position(
    fen: str, engine_pool: EnginePool, barrier: threading.Barrier | None = None
) -> tuple[float, str]:
    """Simulate one user asking one question. Returns (seconds, answer)."""
    db_pool = MagicMock()
    voyage_client = MagicMock()
    client = _fake_client(fen)
    question = f"Evaluate this chess position and tell me the best move: {fen}."

    if barrier is not None:
        barrier.wait()  # line every simulated user up so they hit the pool at once
    start = time.monotonic()
    answer = ask(question, db_pool, engine_pool, voyage_client, client=client)
    return time.monotonic() - start, answer


def test_users_up_to_pool_size_get_answers_within_sla(engine_pool):
    # Measured on a separate, dedicated engine -- not one from engine_pool --
    # so this reflects a genuinely cold, uncontended solo call. Reusing a
    # pooled engine here would warm its transposition table before the
    # concurrent run, unfairly advantaging whichever request lands on it.
    baseline_pool = EnginePool(STOCKFISH_PATH, size=1)
    try:
        baseline, _ = _ask_for_position(EVAL_FEN, baseline_pool)
    finally:
        baseline_pool.close()
    sla_seconds = baseline * SLA_MULTIPLIER

    barrier = threading.Barrier(POOL_SIZE)
    with ThreadPoolExecutor(max_workers=POOL_SIZE) as pool:
        futures = [
            pool.submit(_ask_for_position, EVAL_FEN, engine_pool, barrier) for _ in range(POOL_SIZE)
        ]
        results = [f.result(timeout=30) for f in futures]

    durations = [d for d, _ in results]
    failures = [d for d in durations if d >= sla_seconds]
    assert not failures, (
        f"{len(failures)}/{len(durations)} concurrent user(s) waited longer than "
        f"{sla_seconds:.2f}s ({SLA_MULTIPLIER}x the {baseline:.2f}s solo baseline): "
        f"{[f'{d:.2f}s' for d in durations]}. A pool of size {POOL_SIZE} should let this "
        "many concurrent users run in true parallel."
    )


def test_user_beyond_pool_size_gets_busy_message_not_a_hang(engine_pool):
    num_users = POOL_SIZE + 1
    barrier = threading.Barrier(num_users)
    with ThreadPoolExecutor(max_workers=num_users) as pool:
        futures = [
            pool.submit(_ask_for_position, EVAL_FEN, engine_pool, barrier)
            for _ in range(num_users)
        ]
        results = [f.result(timeout=30) for f in futures]

    answers = [answer for _, answer in results]
    busy_answers = [a for a in answers if "busy" in a.lower()]
    assert len(busy_answers) >= 1, (
        f"expected at least one of {num_users} concurrent users (pool size {POOL_SIZE}) to "
        f"get a 'too many concurrent users' response instead of hanging; got: {answers}"
    )
