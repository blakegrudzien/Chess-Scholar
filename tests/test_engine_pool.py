from unittest.mock import MagicMock, patch

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
