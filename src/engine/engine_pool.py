"""A small, fixed-size pool of Stockfish engine subprocesses, so that
concurrent evaluation requests get genuine parallelism instead of queuing
behind one shared engine process. See tests/test_concurrency.py.

Checkout never blocks. Each engine is CPU-bound for the duration of a
search, so making a caller wait for one to free up would just reintroduce
serialization under a different name. When every engine is checked out,
checkout() raises EngineBusyError immediately, so the caller can return a
clear "try again shortly" response instead of hanging.
"""

from __future__ import annotations

import logging
import queue
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import chess.engine

logger = logging.getLogger(__name__)


class EngineBusyError(Exception):
    """Raised by checkout() when every pooled engine is already in use."""


def _quit_all(engines: Iterable[chess.engine.SimpleEngine]) -> None:
    """Quit every engine, carrying on past any that fail to shut down.

    The broad except is deliberate: the aim is to release as many engines
    as possible, and an already-dead engine shouldn't stop the rest. Each
    failure is logged as a warning.
    """
    for engine in engines:
        try:
            engine.quit()
        except Exception as exc:
            logger.warning("Engine failed to quit cleanly: %s", exc)


class EnginePool:
    def __init__(self, engine_path: str, size: int):
        self.size = size
        self._engine_path = engine_path
        self._available: queue.Queue[chess.engine.SimpleEngine] = queue.Queue(maxsize=size)
        spawned: list[chess.engine.SimpleEngine] = []
        try:
            for _ in range(size):
                engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                spawned.append(engine)
                self._available.put(engine)
        except Exception:
            # If engine N fails to spawn, engines 1..N-1 are already live
            # subprocesses this object never finishes constructing, so
            # nothing else will ever get a chance to close them. Release
            # them here before re-raising, rather than leaking them.
            _quit_all(spawned)
            raise

    @contextmanager
    def checkout(self) -> Iterator[chess.engine.SimpleEngine]:
        try:
            engine = self._available.get_nowait()
        except queue.Empty:
            raise EngineBusyError(
                f"All {self.size} engine(s) are busy handling other requests right now. "
                "Please try again in a moment."
            ) from None
        try:
            yield engine
        except chess.engine.EngineTerminatedError:
            # The engine process itself died, so it can't go back in the
            # pool. A replacement keeps the pool at full size.
            logger.warning("Engine process terminated during use; respawning a replacement.")
            try:
                self._available.put(chess.engine.SimpleEngine.popen_uci(self._engine_path))
            except Exception:
                # The pool stays one engine short until the app restarts,
                # so this is logged as an error.
                logger.error(
                    "Failed to respawn a replacement engine after a crash; "
                    "pool capacity permanently reduced by one (of %d) until the "
                    "process restarts.",
                    self.size,
                )
            raise
        except Exception:
            # The caller's error, not the engine's: return the engine
            # before re-raising.
            self._available.put(engine)
            raise
        else:
            self._available.put(engine)

    def close(self) -> None:
        engines: list[chess.engine.SimpleEngine] = []
        while not self._available.empty():
            engines.append(self._available.get_nowait())
        _quit_all(engines)
