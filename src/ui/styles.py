"""Page-wide visual identity that .streamlit/config.toml can't express on
its own, plus the noindex tag. Applied once, at import time from app.py,
before any content renders.

The stylesheet itself lives in theme.css rather than in a Python string.
That keeps it lintable and editable as CSS -- with real syntax
highlighting, and visible to any CSS tooling -- instead of being opaque
text that only Streamlit ever parses.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_THEME_CSS = (Path(__file__).parent / "theme.css").read_text(encoding="utf-8")


def apply_global_styles() -> None:
    _apply_noindex_tag()
    _apply_theme_css()


def _apply_noindex_tag() -> None:
    # Portfolio demo backed by a personal ChessBase export; keep it out of
    # search engine indexes rather than relying on the URL being merely
    # unlisted. st.markdown's unsafe_allow_html doesn't execute <script>
    # tags (React sets innerHTML), so this goes through st.html's explicit
    # script-execution opt-in instead, rendered inside a sandboxed iframe
    # nested one level inside Streamlit's own app frame -- window.top (not
    # window.parent) is needed to reach the real top document.
    st.html(
        """<script>
        var meta = window.top.document.createElement('meta');
        meta.name = 'robots';
        meta.content = 'noindex, nofollow';
        window.top.document.head.appendChild(meta);
        </script>""",
        unsafe_allow_javascript=True,
    )


def _apply_theme_css() -> None:
    st.html(f"<style>{_THEME_CSS}</style>")
