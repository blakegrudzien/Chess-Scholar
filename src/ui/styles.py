"""Page-wide visual identity that .streamlit/config.toml can't express on
its own, plus the noindex tag. Applied once, at import time from app.py,
before any tab content renders.
"""

from __future__ import annotations

import streamlit as st


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
    # Theme extras that .streamlit/config.toml can't express: config.toml
    # covers colors, all three font roles, heading weights, and base radius
    # natively (see that file's own comments), but a specific inset
    # treatment on the chat input and the recommendation cards' accent rail
    # both need real CSS.
    #
    # The chat-input selector targets Streamlit's own generated Emotion
    # classes for stChatInputTextArea's wrapper, confirmed against the live
    # rendered DOM (data-testid alone has no visible box, its ancestor
    # wrapper does) rather than guessed -- these are an internal,
    # unversioned implementation detail, not a public API, and may need
    # re-verifying after a Streamlit upgrade.
    st.html("""
<style>
.st-emotion-cache-1eewxfn.e1p9v2yr1 {
    background: #DACBA8;
    box-shadow:
        inset 0 2px 3px rgba(43, 31, 23, 0.30),
        inset 0 -1px 0 rgba(237, 225, 204, 0.35);
    border-radius: 2px;
}

.rec-card {
    background: #F1E8D4;
    border: 1px solid #A88F72;
    border-left: 4px solid #6B1E2B;
    border-radius: 2px;
    padding: 18px 20px;
    margin-bottom: 8px;
}
.rec-card .rec-kind {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #A87F3F;
    margin: 0 0 4px;
}
.rec-card h4 {
    font-family: Fraunces, Georgia, serif;
    font-style: italic;
    font-weight: 600;
    font-size: 19px;
    margin: 0 0 2px;
    color: #2B1F17;
}
.rec-card .rec-chapter {
    font-style: italic;
    color: #6B1E2B;
    font-size: 14px;
    margin: 0 0 10px;
}
.rec-card .rec-blurb {
    font-size: 14px;
    color: #4A3527;
    margin: 0;
    line-height: 1.55;
}

/* Chat bubbles: originally keyed on stChatMessageAvatarUser/Assistant, but
   that testid only exists for Streamlit's own built-in emoji/icon avatars
   (confirmed by reading the compiled frontend) -- once the avatar became a
   custom SVG image (see _CHAT_AVATARS), Streamlit renders a bare avatar
   image element with no testid at all, and these rules silently stopped
   matching anything. Do not write anything that looks like an HTML tag
   inside comments in this style block, angle brackets included -- the
   st.html() sanitizer silently drops the whole block whenever it finds
   one, even inside a CSS comment (confirmed by bisection: a single such
   comment reproduces the drop in total isolation, elsewhere in this file).
   stChatMessageContent's aria-label is set unconditionally to
   "Chat message from {role}" regardless of avatar type, so it stays
   correct even if the avatar mechanism changes again later. */
div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageContent"][aria-label="Chat message from user"]
) {
    background: #6B1E2B;
    border-radius: 2px;
}
div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageContent"][aria-label="Chat message from user"]
) [data-testid="stMarkdownContainer"] {
    color: #EDE1CC;
}
div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageContent"][aria-label="Chat message from assistant"]
) {
    /* #F1E8D4 against the page's own #E8DCC3 background read as almost the
       same tone -- reusing #DACBA8 (already the chat-input and inline-code
       "recessed surface" color elsewhere on the page) plus the same inset
       shadow the chat input uses gives every sunken/carved surface in the
       app one consistent color and depth language instead of a bespoke
       near-miss just for this bubble. */
    background: #DACBA8;
    border: 1px solid #A88F72;
    border-radius: 2px;
    box-shadow:
        inset 0 2px 3px rgba(43, 31, 23, 0.30),
        inset 0 -1px 0 rgba(237, 225, 204, 0.35);
}

/* Code blocks (st.code(), language=None everywhere it's used -- FEN, PGN,
   move notation): config.toml's codeTextColor/codeBackgroundColor don't
   reliably reach the "plaintext" language case Streamlit renders when no
   language is set, confirmed by these coming through as illegible
   near-black-on-black. !important is deliberate here: this overrides
   Streamlit's own internal theme CSS of unknown specificity from the
   outside, not a shortcut around this file's own rules. */
pre code.language-plaintext,
pre code.language-plaintext span {
    background: transparent !important;
    color: #EDE1CC !important;
}
pre:has(code.language-plaintext) {
    background: #2B1F17 !important;
}

/* Inline code (backtick text inside a paragraph, not a full st.code()
   block) was inheriting the same dark block treatment, showing as a
   jarring near-black patch floating inside otherwise normal paragraph
   text. Distinguished from block code via :not(pre code) and given its
   own lighter, subtler inline treatment instead. */
code:not(pre code) {
    background: #DACBA8 !important;
    color: #2B1F17 !important;
    padding: 0.1em 0.35em;
    border-radius: 2px;
    font-weight: 500;
}

/* The top-right "Running..." spinner (data-testid confirmed by reading
   Streamlit's compiled frontend directly) is a framework chrome element,
   not part of this app's own designed surface -- hidden rather than
   themed. */
div[data-testid="stStatusWidget"] {
    display: none;
}

/* The Lichess study embed (st.iframe) renders with square corners by
   default, breaking the 2px radius (config.toml's baseRadius) used
   everywhere else on the page -- overflow: hidden is needed alongside
   border-radius since a border-radius alone doesn't clip an iframe's own
   rendered content. */
div[data-testid="stIFrame"] {
    border-radius: 2px;
    overflow: hidden;
}

/* Quarter-sawn grain: three layered streak patterns at slightly different
   angles and widths, reading as sanded walnut planks rather than a flat
   tint, so the empty parchment behind the chat panel isn't bare. Layered
   as background-image on top of config.toml's backgroundColor, not a
   replacement for it. */
div[data-testid="stApp"] {
    background-image:
        repeating-linear-gradient(91deg,
            rgba(107, 74, 44, 0.05) 0px, rgba(107, 74, 44, 0.05) 1px,
            transparent 1px, transparent 7px),
        repeating-linear-gradient(89deg,
            rgba(43, 31, 23, 0.04) 0px, rgba(43, 31, 23, 0.04) 2px,
            transparent 2px, transparent 23px),
        repeating-linear-gradient(90.5deg,
            rgba(168, 143, 114, 0.06) 0px, rgba(168, 143, 114, 0.06) 1px,
            transparent 1px, transparent 41px);
}

/* layout="wide" is needed for the upload tab's file uploader and the
   board panel's 8-column position-editor grid, but it also stretches the
   chat/board row edge to edge on a wide window. Capping the panel that
   contains the chat input (rather than assuming DOM order among the tab
   panels, which didn't actually match :first-of-type in practice) keeps
   the upload tab at full width while giving the chat+board row a fixed
   total measure -- 1100px, split roughly 3:2 between chat and board by
   the st.columns call itself, not by CSS. No margin: auto -- the panel
   already starts at the same left edge as the page title and tab bar
   above it (same parent padding), so capping width alone keeps that edge
   aligned instead of centering the panel into a column visually
   disconnected from the header above it. */
div[data-testid="stTabPanel"]:has([data-testid="stChatInputTextArea"]) {
    max-width: 1100px;
}
</style>
""")
