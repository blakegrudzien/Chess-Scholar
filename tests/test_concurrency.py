"""Concurrency tests for the Stockfish engine pool (src/engine/engine_pool.py).

A single shared chess.engine.SimpleEngine can't serve concurrent requests in
parallel: python-chess serializes concurrent analyse() calls on one engine
process, since only one UCI command can be in flight at a time. These tests
verify the pool instead delivers real parallelism, by driving multiple
simulated users through the real ask() entry point at the same time, all
sharing one EnginePool the same way app.py does. Only the Anthropic model
call is faked (deterministic, no API cost) -- the fake is scripted to make
the same tool call the real model makes for a position-evaluation question,
so the real evaluate_chess_position tool and the real pool are what's
actually exercised.

Two properties are checked:
1. Requests up to the pool size run in genuine parallel: each finishes close
   to a solo-call baseline, not ~Nx it.
2. A request beyond the pool size gets an immediate "too many concurrent
   users" response rather than hanging or erroring.
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

# Every simulated user searches the same position. Search cost varies widely
# by position (e.g. the position after 1.e4 takes roughly 2.4x as long to
# search as the start position at depth 16), so comparing timings across
# different positions would swamp the signal these tests are actually after:
# whether equal-cost requests run in parallel.
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
        turn = MagicMock()
        turn.text_stream = iter([result_text])
        message = MagicMock()
        message.content = [MagicMock(type="text", text=result_text)]
        turn.get_final_message.return_value = message
        return [turn]

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
        barrier.wait()  # synchronize simulated users so they hit the pool at the same instant
    start = time.monotonic()
    answer = ask(question, db_pool, engine_pool, voyage_client, client=client)
    return time.monotonic() - start, answer


def test_users_up_to_pool_size_get_answers_within_sla(engine_pool):
    # The baseline runs on its own dedicated engine, not one borrowed from
    # engine_pool, so it reflects a cold, uncontended solo call. A pooled
    # engine would carry a warmed-up transposition table into the baseline,
    # giving an unearned speed advantage to whichever concurrent request
    # later happens to reuse that same engine.
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
            pool.submit(_ask_for_position, EVAL_FEN, engine_pool, barrier) for _ in range(num_users)
        ]
        results = [f.result(timeout=30) for f in futures]

    answers = [answer for _, answer in results]
    busy_answers = [a for a in answers if "busy" in a.lower()]
    assert len(busy_answers) >= 1, (
        f"expected at least one of {num_users} concurrent users (pool size {POOL_SIZE}) to "
        f"get a 'too many concurrent users' response instead of hanging; got: {answers}"
    )
