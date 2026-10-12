"""Real-browser tests (Playwright) for the board, the phone and desktop
layouts, the tour and the resource lookup: the parts that run in the
browser and can't be checked with mocks.

Skipped with a reason when Playwright or Chromium isn't installed:

    pip install -e ".[dev]"
    playwright install chromium

No database or API keys are needed. Loading the app starts a background
database ping, which without a database only logs a warning.
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
    # Port 0 gets a free port from the OS, avoiding a dev server on 8501.
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _launch_streamlit_app(script_relpath: str) -> Iterator[str]:
    """Run `script_relpath` (from the repo root) as a Streamlit server on a
    free port and yield its URL once it's healthy. Used for the real app and
    the test harness apps."""
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
        # A new process group, so teardown also kills any server process
        # streamlit spawned.
        start_new_session=True,
    )
    try:
        # The health endpoint answers only once the app is serving; "/"
        # answers immediately with the static frontend.
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


# The FEN is inline code in a caption starting "FEN:", among other captions.
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
    """A drag during replay must change nothing: a move there would alter
    the hidden free-play game. The FEN caption must stay the same."""
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


def test_board_fits_a_360px_wide_phone(app_server: str) -> None:
    """360px is the most common Android width; with 16px page margins that
    leaves 328px for the board."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(
            viewport={"width": 360, "height": 780}, is_mobile=True, has_touch=True
        ).new_page()
        page.goto(app_server, wait_until="networkidle")
        page.wait_for_selector(".square-h1", timeout=15000)
        h1 = page.locator(".square-h1").bounding_box()
        overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        browser.close()

    assert h1["x"] + h1["width"] <= 360 - 16, "the h-file runs into or past the page margin"
    assert overflow == 0


def test_title_has_no_self_link(phone_page) -> None:
    """st.title adds a "link to this heading" anchor that phones show
    permanently; on a one-page app it only links back to itself."""
    assert phone_page.locator("h1 a").count() == 0


def _board_width(page) -> float:
    a8, h8 = page.locator(".square-a8").bounding_box(), page.locator(".square-h8").bounding_box()
    return h8["x"] + h8["width"] - a8["x"]


def test_board_is_full_size_when_the_phone_has_room(phone_page) -> None:
    # An iPhone 13 has 358px for the board, more than the 340px maximum.
    # chessboard.js rounds squares down to whole pixels: 8 x 42 = 336.
    assert _board_width(phone_page) == 8 * (340 // 8)


def test_board_is_not_clipped_on_a_1280px_laptop(app_server: str) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(app_server, wait_until="networkidle")
        page.wait_for_selector(".square-h1", timeout=15000)
        h1 = page.locator(".square-h1").bounding_box()
        column = page.locator(".st-key-tutorial_board_target").bounding_box()
        browser.close()

    assert h1["x"] + h1["width"] <= column["x"] + column["width"] + 1


def test_board_refits_when_the_screen_narrows(phone_page) -> None:
    """Rotating a phone or resizing a window changes the available width
    after the board is drawn."""
    phone_page.set_viewport_size({"width": 360, "height": 780})
    phone_page.wait_for_function(
        "(() => { const a = document.querySelector('.square-a8').getBoundingClientRect();"
        " const h = document.querySelector('.square-h8').getBoundingClientRect();"
        " return h.right - a.left <= 328; })()",
        timeout=5000,
    )


@pytest.fixture(scope="module")
def fake_agent_server() -> Iterator[str]:
    yield from _launch_streamlit_app("tests/_fake_agent_harness_app.py")


def test_resource_lookup_runs_in_the_background_and_unlocks_the_button(
    fake_agent_server: str,
) -> None:
    """The lookup starts on its own after an answer, runs in the background
    so a board move can't cancel it, and unlocks the button when done."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(fake_agent_server, wait_until="networkidle")
        page.wait_for_selector(".square-e2", timeout=15000)
        page.get_by_role("button", name="Evaluate 1. e4 e5 2. Qh5 for White").click()

        finding = page.get_by_role("button", name="Finding related resources...")
        finding.wait_for(timeout=15000)
        assert finding.is_disabled()

        _drag(page, "e2", "e4")  # a rerun while the lookup is running

        done = page.get_by_role("button", name="No related resources found")
        done.wait_for(timeout=15000)
        browser.close()


_TOUR_OVERLAP_JS = """(() => {
  const s = document.querySelector('.tour-spotlight').getBoundingClientRect();
  const t = document.querySelector('.tour-tooltip').getBoundingClientRect();
  return Math.max(0, Math.min(s.bottom, t.bottom) - Math.max(s.top, t.top));
})()"""


def _walk_tour(page) -> list[tuple[str, str, float]]:
    """(title, text, px of spotlight covered by the tooltip) for each step."""
    page.get_by_role("button", name="How this works").click()
    page.locator(".tour-tooltip").wait_for(state="visible", timeout=10000)
    steps = []
    while page.locator(".tour-tooltip").count():
        page.wait_for_timeout(900)  # let the smooth scroll and reposition settle
        steps.append(
            (
                page.locator(".tour-title").inner_text(),
                page.locator(".tour-text").inner_text(),
                page.evaluate(_TOUR_OVERLAP_JS),
            )
        )
        page.locator(".tour-next").click()
    return steps


def test_phone_tour_follows_page_order_and_never_covers_its_target(phone_page) -> None:
    steps = _walk_tour(phone_page)
    assert [title for title, _, _ in steps] == [
        "Play on the board",
        "Ask about a position",
        "Analyze with Stockfish",
        "Find related resources",
        "Ask anything about chess",
        "Upload a game",
    ]
    assert all(overlap == 0 for _, _, overlap in steps), steps
    board_text = steps[0][1]
    assert board_text.startswith("Tap a piece")
    # The room added at the page end for the sheet goes away with the tour.
    assert phone_page.locator(".tour-sheet-spacer").count() == 0


def test_desktop_tour_keeps_its_order_and_says_drag(page) -> None:
    steps = _walk_tour(page)
    assert steps[0][0] == "Ask anything about chess"
    board_text = next(text for title, text, _ in steps if title == "Play on the board")
    assert board_text.startswith("Drag pieces")
    assert "python-chess on the server double-checks" in board_text


def test_evaluating_your_own_position_keeps_undo_available(fake_agent_server: str) -> None:
    """Evaluating the board's position leaves its move history, and Undo,
    intact."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(fake_agent_server, wait_until="networkidle")
        page.wait_for_selector(".square-e2", timeout=15000)
        _drag(page, "e2", "e4")
        page.wait_for_function(f"{_FEN_CAPTION_JS}?.textContent.includes('4P3')", timeout=10000)
        undo = page.get_by_role("button", name="Undo last move")
        assert undo.is_enabled()

        page.get_by_role("button", name="Evaluate this position with Stockfish").click()
        page.get_by_text("A fake answer for the test.").first.wait_for(timeout=15000)
        page.wait_for_timeout(1500)

        assert undo.is_enabled()
        browser.close()


def test_answer_headings_have_no_self_links(fake_agent_server: str) -> None:
    """Streamlit adds a "link to this heading" icon to every markdown
    heading; phones show it permanently, and on a one-page app it only
    links back to the same page."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(**p.devices["iPhone 13"]).new_page()
        page.goto(fake_agent_server, wait_until="networkidle")
        page.wait_for_selector(".square-e2", timeout=15000)
        page.get_by_role("button", name="Evaluate 1. e4 e5 2. Qh5 for White").click()
        page.get_by_role("heading", name="Summary").wait_for(timeout=15000)
        links = page.locator('[data-testid="stHeaderActionElements"]')
        assert all(not links.nth(i).is_visible() for i in range(links.count()))
        browser.close()
