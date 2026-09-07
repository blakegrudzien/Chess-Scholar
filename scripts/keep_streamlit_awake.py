"""Visits the deployed Chess Scholar app with a real headless browser so
Streamlit Community Cloud counts it as an actual visit, not just an HTTP hit.

A plain curl ping is not sufficient here, and reports success whether or
not the app is actually up: curl receives whatever the gateway is currently
serving -- the static "this app has gone to sleep" page included -- with a
normal 200/303, so its exit code says nothing about the state of the
underlying app container. Waking the app, and keeping it awake, requires
the full page load and WebSocket handshake that only a real browser
performs, which is what this script does instead.

Exits non-zero if the app never actually loads within the timeout below --
unlike curl's "any HTTP response is success", a real failure here means
something is actually wrong (the app crashed, corpus DB is unreachable, a
genuine outage), so it's worth this job going red instead of silently
reporting green forever.
"""

from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = "https://chess-scholar.streamlit.app/"
# Generous: a cold start (container boot + corpus-adjacent imports) has been
# observed taking upward of a minute in practice, not just a few seconds.
MAX_WAIT_SECONDS = 120


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(APP_URL, timeout=60_000)
        page.wait_for_timeout(3_000)

        text = page.evaluate("() => document.body.innerText")
        if "get this app back up" in text.lower() or "zzz" in text.lower():
            try:
                page.get_by_role("button", name="Yes, get this app back up!").click(timeout=10_000)
            except Exception as exc:
                # The button's exact text and presence are Streamlit Cloud's
                # own UI, not this repo's, so they are not a stable contract.
                # Fall through to the polling loop regardless: if this click
                # was needed and did not happen, the loop times out and fails
                # the job, which is the signal that matters.
                print(f"Wake-up button not clicked ({type(exc).__name__}); continuing anyway.")

        # The real app renders inside a nested iframe on Streamlit Cloud;
        # the top-level page is only hosting chrome. Waiting for real
        # content in that specific frame, rather than for any frame at all,
        # is what actually confirms the app came up.
        start = time.monotonic()
        deadline = start + MAX_WAIT_SECONDS
        while time.monotonic() < deadline:
            app_frame = next((f for f in page.frames if "/~/+/" in f.url), None)
            if app_frame is not None:
                try:
                    frame_text = app_frame.evaluate("() => document.body.innerText")
                except Exception:
                    frame_text = ""
                if "Chess Scholar" in frame_text:
                    elapsed = int(time.monotonic() - start)
                    print(f"App is awake and loaded (after ~{elapsed}s).")
                    browser.close()
                    return 0
            page.wait_for_timeout(3_000)

        print("App did not finish loading within the timeout -- treating as a failure.")
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
