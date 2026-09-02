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

/* Primary buttons (Evaluate this position, Find related resources) render
   with the theme's own primaryColor but flat -- no depth, unlike every
   other surface this app treats as "carved" (chat input above, code
   blocks, the user chat bubble). stBaseButton-primary is a real Streamlit
   testid, not a generated Emotion hash. */
button[data-testid="stBaseButton-primary"] {
    box-shadow:
        inset 0 2px 3px rgba(0, 0, 0, 0.35),
        inset 0 -1px 0 rgba(237, 225, 204, 0.2);
}

/* Reset board / Undo last move are plain (kind="secondary") buttons --
   light surfaces, so this reuses the same lighter inset values as the
   chat input / assistant bubble rather than the darker ones tuned for
   primaryColor's oxblood. */
button[data-testid="stBaseButton-secondary"] {
    box-shadow:
        inset 0 2px 3px rgba(43, 31, 23, 0.30),
        inset 0 -1px 0 rgba(237, 225, 204, 0.35);
}

/* The "Ask about this position..." text input and its "Ask" submit
   button (a form, not a plain button -- Streamlit gives form-submit
   buttons their own kind, "secondaryFormSubmit", distinct from a plain
   button's "secondary") are the two remaining flat surfaces next to the
   now-inset chat input above; same treatment for visual consistency. */
div[data-testid="stTextInputRootElement"] {
    background: #DACBA8;
    box-shadow:
        inset 0 2px 3px rgba(43, 31, 23, 0.30),
        inset 0 -1px 0 rgba(237, 225, 204, 0.35);
    border-radius: 2px;
}
button[data-testid="stBaseButton-secondaryFormSubmit"] {
    box-shadow:
        inset 0 2px 3px rgba(43, 31, 23, 0.30),
        inset 0 -1px 0 rgba(237, 225, 204, 0.35);
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
/* Avatar and message content are flex siblings with no gap by default --
   wide content (a long line, a code span) runs right up against the
   avatar's edge with no breathing room. */
div[data-testid="stChatMessage"] {
    gap: 14px;
}
/* stChatMessageContent stretches to fill the row but had no padding of
   its own on the trailing edge -- wide lines ran flush against the
   bubble's right edge (or, for the assistant, right up to the column's
   own boundary, since that bubble's background/shadow live one level up
   on stChatMessage, not on this element). */
div[data-testid="stChatMessageContent"] {
    padding-right: 16px;
}

div[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageContent"][aria-label="Chat message from user"]
) {
    background: #6B1E2B;
    border-radius: 2px;
    /* Matches the assistant bubble's own inset shadow below -- everything
       else this app treats as a "carved" surface (chat input, code
       blocks, that bubble) gets this same depth language; the user
       bubble was the one flat rectangle left. */
    box-shadow:
        inset 0 2px 3px rgba(0, 0, 0, 0.35),
        inset 0 -1px 0 rgba(237, 225, 204, 0.2);
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

/* The draggable board's element container gets overflow-y: auto from
   Streamlit itself (confirmed by inspecting the live DOM) -- fine as long
   as chessboard.js's actual rendered content never exceeds the exact
   pixel height Python requested, but real browsers don't guarantee that:
   subpixel rounding at a given zoom/DPI, or a brief mismatch between the
   container's declared height and the JS side settling into its own
   layout right after a rebuild, is enough to trip it into showing a
   scrollbar (board_component's own _HEIGHT_BUFFER_PX narrows how often
   this happens but can't guarantee never). The board is never meant to
   scroll internally -- forcing this off entirely is more robust than
   chasing an exact pixel match. Scoped via :has() to the one element
   container that actually holds the board, not every stElementContainer
   on the page. */
div[data-testid="stElementContainer"]:has([class*="board-"]) {
    overflow-y: hidden !important;
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

/* layout="wide" is needed for the board panel's 8-column position-editor
   grid, but it also stretches the whole page edge to edge on a wide
   window. stMainBlockContainer wraps the page title and everything below
   it -- with no more tabs, this is the one screen, so capping its width
   directly (confirmed via a live DOM walk from the chat input up) is all
   that's needed, no :has() scoping required the way there was when a
   second, full-width tab existed alongside this row. 1300px, split
   roughly 3:2 between chat and board by the st.columns call itself, not
   by CSS. No margin: auto -- the container already starts at the page's
   own left padding, so capping width alone keeps that edge aligned
   instead of centering the page into a column visually disconnected from
   its own left edge. */
div[data-testid="stMainBlockContainer"] {
    max-width: 1300px;
}

/* Streamlit's own built-in "Connection error" dialog (shown when the
   WebSocket drops, e.g. the dev server got restarted) is injected by the
   framework itself, not this app's own components, so on its own it
   renders in Streamlit's default light theme instead of this app's
   walnut/parchment identity -- confirmed by inspecting the actual dialog
   DOM while it was showing. role="dialog" and stErrorCodeBlock are real,
   stable attributes here, not generated Emotion hashes like most of the
   rest of this file has to fall back to. */
div[data-testid="stDialog"] section[role="dialog"] {
    background: #2B1F17;
    color: #EDE1CC;
}
div[data-testid="stDialog"] section[role="dialog"] h2 {
    color: #EDE1CC;
}
div[data-testid="stDialog"] section[role="dialog"] > div {
    padding: 8px 24px 24px;
}
div[data-testid="stDialog"] [data-testid="stErrorCodeBlock"] {
    margin-top: 12px;
}

/* The top toolbar (Deploy button, hamburger main menu) is framework
   chrome rendered in Streamlit's own default colors, not this app's
   walnut identity. The icon SVGs use fill="currentColor", so setting
   color here (not just on the toolbar background) is what actually
   recolors them, not just the button hit areas around them. */
div[data-testid="stToolbar"] {
    background: #2B1F17;
    color: #EDE1CC;
}
button[data-testid="stBaseButton-header"],
button[data-testid="stMainMenuButton"] {
    color: #EDE1CC;
}
</style>
""")
