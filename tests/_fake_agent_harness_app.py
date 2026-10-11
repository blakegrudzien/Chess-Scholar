"""The real app with the chat agent and the resource lookup replaced by
fast, offline fakes, for tests/test_board_component.py. The lookup takes 4
seconds and finds nothing, long enough to interact with the page while it
runs."""

import re
import runpy
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.ui.chat as chat  # noqa: E402


def _slow_empty_lookup(*args, **kwargs):
    time.sleep(4)
    return []


def _fake_ask_agent(question, on_position=None, **kwargs):
    # Reports the board's position the way the real evaluate_chess_position
    # tool does, for questions asked about the board.
    match = re.search(r"Current board position: (.+)", question)
    if match and on_position is not None:
        on_position(match.group(1).strip())
    # The heading matches real answers, which use markdown section headings.
    return "## Summary\n\nA fake answer for the test."


chat.ask_agent = _fake_ask_agent
chat.recommend_resources = _slow_empty_lookup
chat.log_conversation_best_effort = lambda *args, **kwargs: None
for getter in (
    "get_db_pool",
    "get_voyage",
    "get_anthropic_client",
    "get_lichess_http_client",
    "get_lichess_pacer",
):
    setattr(chat, getter, MagicMock())

runpy.run_path(str(ROOT / "src" / "app.py"), run_name="__main__")
