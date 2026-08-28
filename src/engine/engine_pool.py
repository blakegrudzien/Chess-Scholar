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

import queue
from collections.abc import Iterator
from contextlib import contextmanager

import chess.engine


class EngineBusyError(Exception):
    """Raised by checkout() when every pooled engine is already in use."""


class EnginePool:
    def __init__(self, engine_path: str, size: int):
        self.size = size
        self._available: queue.Queue[chess.engine.SimpleEngine] = queue.Queue(maxsize=size)
        for _ in range(size):
            self._available.put(chess.engine.SimpleEngine.popen_uci(engine_path))

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
        finally:
            self._available.put(engine)

    def close(self) -> None:
        while not self._available.empty():
            self._available.get_nowait().quit()
