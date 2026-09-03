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
    """Quit each engine, continuing even if one fails to shut down cleanly.

    Deliberately broader than the rest of this codebase's error handling:
    everywhere else, catching Exception and moving on would hide a real
    bug. Here it is the correct behavior, not a shortcut, because the goal
    of this function is "release as many engines as possible," and one
    engine already being dead is exactly the kind of failure this needs to
    survive to reach the rest. It's still worth knowing about, though, so
    it's logged rather than silently discarded -- this module has no
    __main__ entry point to call logging.basicConfig from, but a WARNING
    still reaches stderr by default: Python's logging module falls back to
    a "handler of last resort" for WARNING and above when nothing else has
    configured one, specifically so warnings are never completely silent.
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
            # The subprocess itself died mid-use (crashed, was killed, hit
            # an internal fault) -- unlike every other exception a caller's
            # code might raise inside the `with` block, this one means the
            # engine object is unusable now and forever. The old,
            # unconditional `finally: self._available.put(engine)` handed
            # this exact same broken engine to the next checkout(), which
            # would fail identically on first use -- no test exercised
            # this path, since nothing in this pool has ever actually
            # simulated a mid-use crash. Respawn a replacement so the
            # pool's advertised capacity (self.size, quoted directly in
            # EngineBusyError's own message above) stays true rather than
            # silently shrinking by one every time this happens.
            logger.warning("Engine process terminated during use; respawning a replacement.")
            self._available.put(chess.engine.SimpleEngine.popen_uci(self._engine_path))
            raise
        except Exception:
            # Not an engine-health problem (a bad argument, a bug in the
            # caller's own code) -- the process itself is still healthy,
            # so return it normally before letting the caller's error
            # propagate, the same distinction db_loader.query_with_retry
            # draws between a dead connection and any other failure.
            self._available.put(engine)
            raise
        else:
            self._available.put(engine)

    def close(self) -> None:
        engines: list[chess.engine.SimpleEngine] = []
        while not self._available.empty():
            engines.append(self._available.get_nowait())
        _quit_all(engines)
