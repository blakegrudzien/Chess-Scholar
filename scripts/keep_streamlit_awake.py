"""Visits the deployed Chess Scholar app with a real headless browser so
Streamlit Community Cloud counts it as an actual visit, not just an HTTP hit.

Replaces a plain curl ping (the original keep-alive.yml) that ran daily and
reported success for days straight while the app was, in fact, asleep --
confirmed live, not theoretical: curl gets back whatever the gateway is
currently serving (the static "this app has gone to sleep" page included)
with a normal 200/303, so its exit code says nothing about whether the
underlying app container is actually awake. Waking (and staying awake)
requires the real page load + WebSocket handshake only a browser produces,
which is what this script does instead.

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
            except Exception:
                # The button's exact text/presence isn't a stable contract
                # (Streamlit Cloud's own UI, not this repo's) -- fall
                # through to the polling loop below regardless; it will
                # simply time out and fail the job if this didn't help.
                pass

        # The real app renders inside a nested iframe on Streamlit Cloud
        # (confirmed live: the top-level page is just hosting chrome) --
        # waiting for real content in that specific frame, not just "some
        # frame changed", is what actually confirms the app came up.
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
