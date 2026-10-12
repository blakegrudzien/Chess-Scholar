"""Limits on what a visitor can send, so a public demo with no login can't
run up the API bill: a maximum question length, and a daily question budget
shared by every visitor (on top of chat.py's per-session rate limit).
"""

from __future__ import annotations

import threading
import time
from collections import deque

import streamlit as st

# About 200 words. Every question is resent on each agent turn and on each
# turn of the background resource lookup, so its length multiplies its cost.
MAX_QUESTION_CHARS = 1200

# Questions per rolling 24 hours across every session. The per-session rate
# limit resets when a visitor reloads; this one doesn't. It lives in server
# memory, so it also resets when the app restarts.
SHARED_DAILY_QUESTION_LIMIT = 500
SHARED_LIMIT_WINDOW_SECONDS = 24 * 60 * 60


def accept_question_length(text: str) -> bool:
    """Whether `text` is short enough to send, showing a warning if not.
    Checked on the server too, since the inputs' max_chars only binds the
    browser."""
    if len(text) <= MAX_QUESTION_CHARS:
        return True
    st.warning(
        f"That question is too long. Please keep it under about 200 words "
        f"({MAX_QUESTION_CHARS} characters)."
    )
    return False


class SharedQuestionLimit:
    """A rolling-window count of questions, shared across sessions."""

    def __init__(self, limit: int, window_seconds: float):
        self._limit = limit
        self._window_seconds = window_seconds
        self._asked_at: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self, now: float | None = None) -> bool:
        """Count one question and return True, or return False if the limit
        is already reached within the window."""
        now = time.monotonic() if now is None else now
        with self._lock:
            while self._asked_at and now - self._asked_at[0] >= self._window_seconds:
                self._asked_at.popleft()
            if len(self._asked_at) >= self._limit:
                return False
            self._asked_at.append(now)
            return True


@st.cache_resource
def get_shared_question_limit() -> SharedQuestionLimit:
    return SharedQuestionLimit(SHARED_DAILY_QUESTION_LIMIT, SHARED_LIMIT_WINDOW_SECONDS)
