"""The progress bar shown while an answer generates, in place of a bare
"Thinking..." label.

Answers take roughly 20 to 70 seconds, past the ~10 second point where a
looping spinner starts to read as "is this frozen?", so the bar shows
steady progress instead. The app can't know in advance how many tool-calling
turns an answer will take, so the bar is paced by time: it fills to
FILL_FRACTION over FULL_BAR_SECONDS (the slowest answers seen so far), then
creeps toward CREEP_TARGET so a long answer never shows a full bar that is
still working. When the answer arrives, ask_with_status swaps in a short
animation from wherever the bar is to 100%.

The fill runs as a CSS animation in the browser rather than as Python
updates, so it keeps moving while Python is blocked waiting on the model or
a tool. Both the CSS keyframes and progress_fraction are built from the
constants below, so the browser's bar and Python's idea of where it is
can't drift apart.
"""

from __future__ import annotations

FULL_BAR_SECONDS = 70.0
FILL_FRACTION = 0.90
CREEP_SECONDS = 120.0
CREEP_TARGET = 0.98
FINISH_SECONDS = 0.3

_TOTAL_SECONDS = FULL_BAR_SECONDS + CREEP_SECONDS

# Shared by both bars. Fixed class names rather than Streamlit keys: st.html
# content is plain markup, and theme.css styles only Streamlit widgets.
_STYLE = """
.answer-progress {
    height: 6px;
    /* Darker than the assistant bubble (#DACBA8) the bar sits on, so the
       unfilled track stays visible, with the inset shadow the app uses
       for other carved surfaces. */
    background: rgba(43, 31, 23, 0.16);
    box-shadow: inset 0 1px 2px rgba(43, 31, 23, 0.25);
    border-radius: 2px;
    overflow: hidden;
    margin: 4px 0 8px;
}
.answer-progress-fill {
    height: 100%;
    background: #6B1E2B;
    border-radius: 2px;
}
"""


def progress_fraction(elapsed: float) -> float:
    """Where the running bar is after `elapsed` seconds, from 0 to
    CREEP_TARGET. Piecewise linear, matching running_bar_html's keyframes."""
    if elapsed <= 0:
        return 0.0
    if elapsed <= FULL_BAR_SECONDS:
        return FILL_FRACTION * elapsed / FULL_BAR_SECONDS
    creep = min((elapsed - FULL_BAR_SECONDS) / CREEP_SECONDS, 1.0)
    return FILL_FRACTION + (CREEP_TARGET - FILL_FRACTION) * creep


def running_bar_html() -> str:
    fill_point = 100 * FULL_BAR_SECONDS / _TOTAL_SECONDS
    return f"""<style>{_STYLE}
@keyframes answer-progress-run {{
    0% {{ width: 0%; }}
    {fill_point:.2f}% {{ width: {100 * FILL_FRACTION:.2f}%; }}
    100% {{ width: {100 * CREEP_TARGET:.2f}%; }}
}}
</style>
<div class="answer-progress" role="progressbar" aria-label="Generating answer">
    <div class="answer-progress-fill"
         style="animation: answer-progress-run {_TOTAL_SECONDS:g}s linear forwards"></div>
</div>"""


def finishing_bar_html(elapsed: float) -> str:
    start = 100 * progress_fraction(elapsed)
    return f"""<style>{_STYLE}
@keyframes answer-progress-finish {{
    from {{ width: {start:.2f}%; }}
    to {{ width: 100%; }}
}}
</style>
<div class="answer-progress" role="progressbar" aria-label="Answer ready">
    <div class="answer-progress-fill"
         style="animation: answer-progress-finish {FINISH_SECONDS:g}s ease-out forwards"></div>
</div>"""
