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
    """An engine whose process died (EngineTerminatedError) is not returned
    to the pool; a replacement keeps the pool at full size."""
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

    # The next caller gets the replacement, and the pool is full again.
    with pool.checkout() as engine:
        assert engine is replacement_engine


def test_checkout_surfaces_original_error_when_respawn_after_crash_also_fails(caplog):
    """If the replacement also fails to start, the original
    EngineTerminatedError still reaches the caller, and the lost capacity
    is logged as an error."""
    dead_engine = MagicMock(name="dead")
    with patch("chess.engine.SimpleEngine.popen_uci", side_effect=[dead_engine]):
        pool = EnginePool("stockfish", size=1)

    with (
        patch(
            "chess.engine.SimpleEngine.popen_uci",
            side_effect=RuntimeError("stockfish binary not found"),
        ),
        pytest.raises(chess.engine.EngineTerminatedError),
    ):
        with pool.checkout() as engine:
            assert engine is dead_engine
            raise chess.engine.EngineTerminatedError("process exited")

    assert any(record.levelname == "ERROR" for record in caplog.records)
    with pytest.raises(EngineBusyError):
        with pool.checkout():
            pass


def test_checkout_still_returns_a_healthy_engine_after_an_unrelated_error():
    """An exception from the caller's own code returns the engine to the
    pool, since the engine is fine."""
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
    """close() releases every engine even if one fails to quit."""
    engines = _mock_engines(3)
    engines[1].quit.side_effect = RuntimeError("already dead")
    with patch("chess.engine.SimpleEngine.popen_uci", side_effect=engines):
        pool = EnginePool("stockfish", size=3)

    pool.close()  # must not raise

    for engine in engines:
        engine.quit.assert_called_once()


def test_init_cleans_up_already_spawned_engines_when_a_later_one_fails():
    """If engine 3 of 3 fails to start, engines 1 and 2 are shut down,
    since nothing else could close them."""
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
