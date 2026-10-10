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
tests specifically. Every external-resource getter is behind
@st.cache_resource and reached only from a chat/recommendation button's
on-click path, with one exception: booting the app starts a background
database ping (app.py's _keep_database_warm). Without a database it fails
on its own thread and only logs a warning, so the page and these tests are
unaffected.
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


def _launch_streamlit_app(script_relpath: str) -> Iterator[str]:
    """Launch `script_relpath` (relative to the repo root) as a real
    Streamlit server on a free port, yielding its base URL once healthy.
    Shared by app_server (the real app) and replay_app_server (a small
    harness -- see _replay_harness_app.py -- that seeds
    st.session_state.game_path directly, so the read-only/Prev/Next replay
    behavior can be tested without a real recommend_resources() call).
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            script_relpath,
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
            raise RuntimeError(f"{script_relpath} did not become healthy in time")
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


@pytest.fixture(scope="module")
def app_server() -> Iterator[str]:
    yield from _launch_streamlit_app("src/app.py")


@pytest.fixture(scope="module")
def replay_app_server() -> Iterator[str]:
    yield from _launch_streamlit_app("tests/_replay_harness_app.py")


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


@pytest.fixture
def replay_page(replay_app_server: str) -> Iterator[Page]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(replay_app_server, wait_until="networkidle")
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


# The FEN under the board is an `st.caption` with inline `` `code` ``
# (see _render_board_panel), not an st.code() block, so it has no <pre> of
# its own -- find the caption whose text starts with "FEN:" and read its
# nested <code> element instead of matching on tag alone, since multiple
# stCaptionContainer elements exist on the page (e.g. "Turn: White").
_FEN_CAPTION_JS = (
    "[...document.querySelectorAll('[data-testid=\"stCaptionContainer\"]')]"
    ".find(c => c.textContent.startsWith('FEN:'))?.querySelector('code')"
)


def _board_fen(page) -> str:
    return page.evaluate(f"{_FEN_CAPTION_JS}?.textContent.trim()")


def test_dragging_a_legal_move_updates_the_fen(page) -> None:
    _drag(page, "e2", "e4")
    page.wait_for_function(f"{_FEN_CAPTION_JS}?.textContent.includes('4P3')", timeout=10000)
    assert _board_fen(page).startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")


def test_dragging_an_illegal_move_leaves_the_fen_unchanged(page) -> None:
    starting_fen = _board_fen(page)
    _drag(page, "e2", "e5")
    time.sleep(1)  # let a wrongly-accepted move have time to show up, if any
    assert _board_fen(page) == starting_fen


def test_replay_board_is_read_only(replay_page) -> None:
    """draggable=False during replay must actually stop a drop from being
    acted on, not just look inert -- _attempt_move mutates the (invisible,
    behind-the-scenes) free-play board unconditionally, so if this ever
    regressed, a drag during replay would corrupt that board silently
    while the visibly-replayed piece just snaps back with no feedback at
    all. Checking the FEN caption doesn't change is the only way to catch
    that from outside; a passing drag with no visible effect could still
    mean silent corruption happened underneath.
    """
    starting_fen = _board_fen(replay_page)
    _drag(replay_page, "e2", "e4")
    time.sleep(1)
    assert _board_fen(replay_page) == starting_fen


def test_replay_next_and_previous_step_through_the_game(replay_page) -> None:
    starting_fen = _board_fen(replay_page)
    assert replay_page.get_by_role("button", name="Previous").is_disabled()

    replay_page.get_by_role("button", name="Next").click()
    replay_page.wait_for_function(
        f"{_FEN_CAPTION_JS}?.textContent.trim() !== {starting_fen!r}", timeout=5000
    )
    assert _board_fen(replay_page) != starting_fen
    assert not replay_page.get_by_role("button", name="Previous").is_disabled()

    replay_page.get_by_role("button", name="Previous").click()
    replay_page.wait_for_function(
        f"{_FEN_CAPTION_JS}?.textContent.trim() === {starting_fen!r}", timeout=5000
    )
    assert _board_fen(replay_page) == starting_fen
    assert replay_page.get_by_role("button", name="Previous").is_disabled()


@pytest.fixture
def phone_page(app_server: str) -> Iterator[Page]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(**p.devices["iPhone 13"])
        pg = context.new_page()
        pg.goto(app_server, wait_until="networkidle")
        pg.wait_for_selector(".square-e2", timeout=15000)
        yield pg
        browser.close()


def test_phone_layout_puts_the_board_first_and_folds_its_controls(phone_page) -> None:
    def top(selector: str) -> float:
        return phone_page.locator(selector).first.bounding_box()["y"]

    assert top(".st-key-board_panel") < top(".st-key-chat_panel")

    reset = phone_page.get_by_role("button", name="Reset board")
    assert reset.count() == 0 or not reset.is_visible()
    phone_page.get_by_text("Board controls").click()
    reset.wait_for(state="visible", timeout=5000)


def test_desktop_shows_board_controls_without_an_expander_header(page) -> None:
    assert page.get_by_role("button", name="Reset board").is_visible()
    assert not page.get_by_text("Board controls").first.is_visible()


def test_phone_chat_panel_leaves_no_blank_gap_above_the_chat_input(phone_page) -> None:
    """Streamlit sizes the panel's wrapper with flex: 0 0 560px, which kept
    reserving the desktop height below a shrunken phone panel."""
    panel = phone_page.locator(".st-key-chat_panel").bounding_box()
    chat_input = phone_page.locator('[data-testid="stChatInput"]').bounding_box()
    gap = chat_input["y"] - (panel["y"] + panel["height"])
    assert gap < 60, f"{gap:.0f}px of blank space between the chat panel and the input"


_SELECTED = "highlight1-32417"  # chessboard.js's own yellow drag highlight


def _piece_on(page, square: str) -> str | None:
    # The FEN caption sits in the collapsed "Board controls" expander on a
    # phone, so these tests read chessboard.js's own piece markup instead.
    piece = page.locator(f".square-{square} img[data-piece]")
    return piece.get_attribute("data-piece") if piece.count() else None


def _selected(page, square: str) -> bool:
    return _SELECTED in (page.locator(f".square-{square}").get_attribute("class") or "")


def test_phone_tap_to_move_moves_the_piece(phone_page) -> None:
    phone_page.locator(".square-e2").tap()
    phone_page.locator(".square-e4").tap()
    phone_page.locator(".square-e4 img[data-piece='wP']").wait_for(timeout=10000)
    assert _piece_on(phone_page, "e2") is None


def test_phone_tapping_the_selected_piece_deselects_it(phone_page) -> None:
    phone_page.locator(".square-e2").tap()
    assert _selected(phone_page, "e2")
    phone_page.locator(".square-e2").tap()
    assert not _selected(phone_page, "e2")


def test_phone_tapping_another_own_piece_switches_selection(phone_page) -> None:
    phone_page.locator(".square-e2").tap()
    phone_page.locator(".square-d2").tap()
    assert _selected(phone_page, "d2")
    assert not _selected(phone_page, "e2")


def test_phone_tapping_an_unreachable_square_is_rejected_in_the_browser(phone_page) -> None:
    phone_page.locator(".square-e2").tap()
    phone_page.locator(".square-e5").tap()
    phone_page.wait_for_timeout(1000)  # time for a wrongly sent move to come back

    assert not _selected(phone_page, "e2")
    assert _piece_on(phone_page, "e2") == "wP"
    assert _piece_on(phone_page, "e5") is None
    # Rejected by chess.js, never sent: no server-side warning appears.
    assert phone_page.get_by_text("That move isn't legal").count() == 0


@pytest.fixture(scope="module")
def slow_server() -> Iterator[str]:
    yield from _launch_streamlit_app("tests/_slow_move_harness_app.py")


@pytest.fixture
def slow_page(slow_server: str) -> Iterator[Page]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1400, "height": 1000})
        pg.goto(slow_server, wait_until="networkidle")
        pg.wait_for_selector(".square-e2", timeout=15000)
        yield pg
        browser.close()


@pytest.fixture
def slow_phone_page(slow_server: str) -> Iterator[Page]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_context(**p.devices["iPhone 13"]).new_page()
        pg.goto(slow_server, wait_until="networkidle")
        pg.wait_for_selector(".square-e2", timeout=15000)
        yield pg
        browser.close()


def test_a_tapped_move_is_drawn_before_the_server_answers(slow_phone_page) -> None:
    slow_phone_page.locator(".square-e2").tap()
    slow_phone_page.locator(".square-e4").tap()
    # The server takes 1.5s to process a move here, so seeing the pawn well
    # within that means the browser drew it.
    slow_phone_page.locator(".square-e4 img[data-piece='wP']").wait_for(timeout=500)


def test_castling_moves_the_rook_instantly(slow_phone_page) -> None:
    for from_square, to_square in (
        ("e2", "e4"),
        ("e7", "e5"),
        ("g1", "f3"),
        ("b8", "c6"),
        ("f1", "c4"),
        ("g8", "f6"),
    ):
        slow_phone_page.locator(f".square-{from_square}").tap()
        slow_phone_page.locator(f".square-{to_square}").tap()
    slow_phone_page.locator(".square-e1").tap()
    slow_phone_page.locator(".square-g1").tap()

    slow_phone_page.locator(".square-f1 img[data-piece='wR']").wait_for(timeout=500)
    assert _piece_on(slow_phone_page, "g1") == "wK"
    assert _piece_on(slow_phone_page, "h1") is None


def test_quick_consecutive_moves_all_reach_the_server(slow_page) -> None:
    """Each move is sent only after the previous one is confirmed, because
    Streamlit can fold two quick trigger values into a single rerun."""
    _drag(slow_page, "e2", "e4")
    _drag(slow_page, "e7", "e5")
    _drag(slow_page, "g1", "f3")
    after_three_moves = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b"
    slow_page.wait_for_function(
        f"{_FEN_CAPTION_JS}?.textContent.includes('{after_three_moves}')", timeout=15000
    )


def test_undo_redraws_the_board_from_the_server(page) -> None:
    _drag(page, "e2", "e4")
    page.wait_for_function(f"{_FEN_CAPTION_JS}?.textContent.includes('4P3')", timeout=10000)
    page.get_by_role("button", name="Undo last move").click()
    page.locator(".square-e2 img[data-piece='wP']").wait_for(timeout=10000)
    assert _piece_on(page, "e4") is None
