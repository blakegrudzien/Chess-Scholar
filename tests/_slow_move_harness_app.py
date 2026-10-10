"""The real app with the server's move handling slowed by 1.5 seconds, for
tests/test_board_component.py. With the delay, a move that shows up on the
board right away can only have been drawn by the browser, not by a rerun."""

import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.ui.board_panel as board_panel  # noqa: E402

if not getattr(board_panel, "_slowed_for_tests", False):
    _attempt_move = board_panel._attempt_move

    def _slow_attempt_move(source: str, target: str) -> None:
        time.sleep(1.5)
        _attempt_move(source, target)

    board_panel._attempt_move = _slow_attempt_move
    board_panel._slowed_for_tests = True

runpy.run_path(str(ROOT / "src" / "app.py"), run_name="__main__")
