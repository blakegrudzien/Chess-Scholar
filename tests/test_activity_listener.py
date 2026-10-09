"""Real-browser coverage for the keystroke activity listener
(src/ui/activity_listener). Its arming and disarming live in JavaScript,
so only a real browser can show that a keystroke reaches Python, does so
once per armed period, and reruns only its fragment without disturbing
the text being typed. Runs against a small harness app, and skips like
test_board_component.py when Playwright or Chromium isn't installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from tests.test_board_component import _launch_streamlit_app, _playwright_available

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.skipif(
    not _playwright_available(),
    reason="requires Playwright + Chromium (pip install -e '.[dev]' && "
    "playwright install chromium)",
)


@pytest.fixture(scope="module")
def harness_server() -> Iterator[str]:
    yield from _launch_streamlit_app("tests/_activity_listener_harness_app.py")


@pytest.fixture
def page(harness_server: str) -> Iterator[Page]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.goto(harness_server, wait_until="networkidle")
        pg.get_by_text("keystrokes recorded: 0").wait_for(timeout=15000)
        yield pg
        browser.close()


def test_first_keystroke_is_recorded_once_and_reruns_only_the_fragment(page) -> None:
    box = page.get_by_label("Question")
    box.click()
    box.press_sequentially("e4", delay=100)
    page.get_by_text("keystrokes recorded: 1").wait_for(timeout=10000)

    box.press_sequentially(" or d4?", delay=100)
    page.wait_for_timeout(1500)  # time for a wrongly re-armed listener to fire again

    assert page.get_by_text("keystrokes recorded: 1").is_visible()
    assert page.get_by_text("full runs: 1").is_visible()
    assert box.input_value() == "e4 or d4?"
