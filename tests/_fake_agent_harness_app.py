"""The real app with the chat agent and the resource lookup replaced by
fast, offline fakes, for tests/test_board_component.py. The lookup takes 4
seconds and finds nothing, long enough to interact with the page while it
runs."""

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


chat.ask_agent = lambda question, **kwargs: "A fake answer for the test."
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
