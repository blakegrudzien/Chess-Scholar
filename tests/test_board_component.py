"""Real-browser coverage for the draggable board (src/ui/board_component),
the one part of this app that genuinely cannot be verified by mocking --
it's a hand-rolled chessboard.js integration wired through Streamlit's
st.components.v2 bidirectional protocol, actual drag physics included.

Runs as part of the normal `pytest -q` / `pytest -v tests` command, same as
every other test file here -- not a separate suite or CI job. Mirrors the
project's own established pattern for "needs a real external resource, skip
cleanly if it's not present" (see STOCKFISH_PATH in test_stockfish_eval.py
and _postgres_available() in test_structured_search.py): if Playwright and a
Chromium build aren't installed, this file's tests are skipped with a clear
reason instead of failing the whole suite.

    pip install -e ".[dev]"
    playwright install chromium

No live Postgres/Anthropic/Voyage/Stockfish credentials are needed for these
tests specifically -- confirmed by reading src/ui/chat.py, upload.py, and
resources.py end to end: every external-resource getter is behind
@st.cache_resource and only ever reached from a chat/recommendation
button's on-click path, never from booting the app or dragging a piece.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Page


def _playwright_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _playwright_available(),
    reason="requires Playwright + Chromium (pip install -e '.[dev]' && "
    "playwright install chromium)",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    # Bind to port 0 to let the OS assign a free one, then release it --
    # avoids colliding with a dev server the developer might already have
    # running on Streamlit's default 8501. A small TOCTOU window exists
    # between closing this socket and Streamlit binding the same port, not
    # worth engineering around for a local, single-worker test suite.
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def app_server() -> Iterator[str]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            "src/app.py",
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # so teardown can kill the whole process
        # group, not just this one PID -- streamlit run can spawn a server
        # process distinct from the one launched here depending on install
        # configuration, and killing only the parent can leak an orphaned
        # server still bound to the port.
    )
    try:
        # /_stcore/health is Streamlit's own readiness endpoint (returns
        # 200 "ok" only once actually serving) -- polling "/" instead would
        # give a false positive, since Streamlit serves its static frontend
        # shell immediately regardless of whether the app script has
        # finished running.
        for _ in range(60):
            try:
                if httpx.get(f"{base_url}/_stcore/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("app server did not become healthy in time")
        yield base_url
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def page(app_server: str) -> Iterator[Page]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Function-scoped, unlike the module-scoped server: st.session_state
        # is scoped per browser session, so a fresh page per test gets a
        # fresh, isolated board with no explicit reset needed between tests.
        pg = browser.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(app_server, wait_until="networkidle")
        pg.wait_for_selector(".square-e2", timeout=15000)
        yield pg
        browser.close()


def _drag(page, from_square: str, to_square: str) -> None:
    source = page.locator(f".square-{from_square}")
    target = page.locator(f".square-{to_square}")
    s, t = source.bounding_box(), target.bounding_box()
    sx, sy = s["x"] + s["width"] / 2, s["y"] + s["height"] / 2
    tx, ty = t["x"] + t["width"] / 2, t["y"] + t["height"] / 2
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move((sx + tx) / 2, (sy + ty) / 2, steps=5)
    page.mouse.move(tx, ty, steps=5)
    page.mouse.up()


def _board_fen(page) -> str:
    return page.evaluate("document.querySelector('pre').textContent.trim()")


def test_dragging_a_legal_move_updates_the_fen(page) -> None:
    _drag(page, "e2", "e4")
    page.wait_for_function(
        "document.querySelector('pre').textContent.includes('4P3')", timeout=10000
    )
    assert _board_fen(page).startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")


def test_dragging_an_illegal_move_leaves_the_fen_unchanged(page) -> None:
    starting_fen = _board_fen(page)
    _drag(page, "e2", "e5")
    time.sleep(1)  # let a wrongly-accepted move have time to show up, if any
    assert _board_fen(page) == starting_fen
