"""Page styling that .streamlit/config.toml can't express, plus a noindex
tag. The CSS lives in theme.css so it can be edited and linted as CSS.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_THEME_CSS = (Path(__file__).parent / "theme.css").read_text(encoding="utf-8")


def apply_global_styles() -> None:
    _apply_noindex_tag()
    _apply_theme_css()


def _apply_noindex_tag() -> None:
    # Keeps the demo out of search engines. st.html must opt in to run
    # scripts, and its frame sits one level inside the app's, hence
    # window.top.
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
