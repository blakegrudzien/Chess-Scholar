from unittest.mock import MagicMock, patch

import chess.engine
import pytest

from src.engine.engine_pool import EngineBusyError, EnginePool


def _mock_engines(count: int) -> list[MagicMock]:
    return [MagicMock(name=f"engine{i}") for i in range(count)]


def test_checkout_yields_a_pooled_engine_and_returns_it_after():
    engines = _mock_engines(2)
    with patch("chess.engine.SimpleEngine.popen_uci", side_effect=engines):
        pool = EnginePool("stockfish", size=2)

    with pool.checkout() as engine:
        assert engine in engines

    # Returned to the pool: checking out twice more (size=2) should not
    # raise EngineBusyError, and a third concurrent checkout should.
    with pool.checkout(), pool.checkout():
        with pytest.raises(EngineBusyError):
            with pool.checkout():
                pass


def test_checkout_discards_a_terminated_engine_and_respawns_a_replacement():
    """Regression test: the old code unconditionally returned whatever
    engine checkout() handed out back to the pool in a bare `finally`,
    even one whose subprocess had just died mid-use (EngineTerminatedError
    -- crashed, killed, an internal fault). The next checkout() would then
    hand out that same broken engine and fail identically. A dead engine
    must be discarded, not pooled, and replaced so self.size (quoted
    directly in EngineBusyError's message) stays true.
    """
    dead_engine = MagicMock(name="dead")
    replacement_engine = MagicMock(name="replacement")
    with patch("chess.engine.SimpleEngine.popen_uci", side_effect=[dead_engine]):
        pool = EnginePool("stockfish", size=1)

    with (
        patch("chess.engine.SimpleEngine.popen_uci", return_value=replacement_engine),
        pytest.raises(chess.engine.EngineTerminatedError),
    ):
        with pool.checkout() as engine:
            assert engine is dead_engine
            raise chess.engine.EngineTerminatedError("process exited")

    # The replacement, not the dead engine, is what the next caller gets --
    # and the pool is back at full capacity (size=1), not exhausted.
    with pool.checkout() as engine:
        assert engine is replacement_engine


def test_checkout_still_returns_a_healthy_engine_after_an_unrelated_error():
    """A caller's own bug (or any exception unrelated to engine health)
    must not be treated as if the engine itself died -- it's still a
    perfectly good, live subprocess and belongs back in the pool.
    """
    engine = MagicMock(name="engine")
    with patch("chess.engine.SimpleEngine.popen_uci", side_effect=[engine]):
        pool = EnginePool("stockfish", size=1)

    with pytest.raises(ValueError, match="bad argument"):
        with pool.checkout() as checked_out:
            raise ValueError("bad argument")

    with pool.checkout() as checked_out:
        assert checked_out is engine  # the same object, not a respawned one


def test_close_quits_every_engine():
    engines = _mock_engines(3)
    with patch("chess.engine.SimpleEngine.popen_uci", side_effect=engines):
        pool = EnginePool("stockfish", size=3)

    pool.close()

    for engine in engines:
        engine.quit.assert_called_once()


def test_close_continues_past_one_engine_failing_to_quit():
    """One engine failing to shut down cleanly must not strand the rest --
    close() exists to release every engine, not to stop at the first error.
    """
    engines = _mock_engines(3)
    engines[1].quit.side_effect = RuntimeError("already dead")
    with patch("chess.engine.SimpleEngine.popen_uci", side_effect=engines):
        pool = EnginePool("stockfish", size=3)

    pool.close()  # must not raise

    for engine in engines:
        engine.quit.assert_called_once()


def test_init_cleans_up_already_spawned_engines_when_a_later_one_fails():
    """If engine 3 of 3 fails to spawn, engines 1 and 2 are already live
    subprocesses -- __init__ never finishes, so nothing else will ever get
    a chance to close them unless it happens here.
    """
    engines = _mock_engines(2)
    with (
        patch(
            "chess.engine.SimpleEngine.popen_uci",
            side_effect=[*engines, RuntimeError("failed to spawn")],
        ),
        pytest.raises(RuntimeError, match="failed to spawn"),
    ):
        EnginePool("stockfish", size=3)

    for engine in engines:
        engine.quit.assert_called_once()


def test_init_cleanup_survives_a_bad_quit_during_partial_failure():
    """Combines both failure modes: engine 3 fails to spawn, and engine 1
    also fails to quit during cleanup. Engine 2 must still be released.
    """
    engines = _mock_engines(2)
    engines[0].quit.side_effect = RuntimeError("already dead")
    with (
        patch(
            "chess.engine.SimpleEngine.popen_uci",
            side_effect=[*engines, RuntimeError("failed to spawn")],
        ),
        pytest.raises(RuntimeError, match="failed to spawn"),
    ):
        EnginePool("stockfish", size=3)

    for engine in engines:
        engine.quit.assert_called_once()
