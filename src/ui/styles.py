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

/* The scrollable message panel (render_main_screen's st.container(height=
   MESSAGE_PANEL_HEIGHT_PX, border=True), "the chatbox") is the one layout
   wrapper that gets its own distinct surface -- unlike the "Ask about
   this position" form or other plain grouping containers, where the
   individual block elements inside (the text input, the button) already
   carry their own inset treatment and the wrapper is just spacing
   between them, left showing the page's own background on purpose. The
   chatbox holds the whole conversation, not just a couple of controls,
   so it reads as its own sunken panel -- same card tone .rec-card uses,
   plus the same inset depth language every other carved surface in the
   app has.

   [height="560px"], not [height]:not([height="auto"]) -- that broader
   selector was a real bug, not just imprecise: Streamlit sets height="100%"
   (not "auto") on plenty of ordinary nested stVerticalBlocks as part of
   its normal column layout, unrelated to this container's own explicit
   height param -- confirmed live from a real bug report, where it painted
   this same background onto the board's own column block too. Matching
   the literal pixel value ties this rule to MESSAGE_PANEL_HEIGHT_PX in
   chat.py; update both together if that constant ever changes. */
div[data-testid="stVerticalBlock"][height="560px"] {
    background: #F1E8D4;
    box-shadow:
        inset 0 2px 3px rgba(43, 31, 23, 0.30),
        inset 0 -1px 0 rgba(237, 225, 204, 0.35);
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

/* The st.status "Thinking..." panel (shown live while a response
   generates, and the similar "Looking for related resources..." one --
   both st.status, which renders as an stExpander under the hood, the
   only place this app uses one) blended into its own chat bubble's
   background: same tan tone, no visual separation. The dark walnut
   treatment used for code blocks was tried here first and didn't work --
   too heavy/dark for a panel holding a full sentence of rationale text,
   unlike a short code/FEN snippet. Same lighter card tone the message
   panel and .rec-card use instead, still with the inset depth every
   other carved surface in the app has, just not dark. */
div[data-testid="stExpander"] {
    background: #F1E8D4;
    border-radius: 2px;
    box-shadow:
        inset 0 2px 3px rgba(43, 31, 23, 0.30),
        inset 0 -1px 0 rgba(237, 225, 204, 0.35);
}
div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
    color: #2B1F17;
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
   second, full-width tab existed alongside this row. 1700px, split
   roughly 3:2 between chat and board by the st.columns call itself, not
   by CSS -- bumped from 1450px because chat_col's own message panel is
   now a fixed-height scrolling box (see render_main_screen), so a
   too-narrow chat_col means more of a given answer's text wraps onto
   more lines, needing more scrolling within that fixed height to read
   the same content. Widened the whole row rather than shifting the 5:4
   ratio further toward chat, so board_col keeps the same safe margin
   above the ~565px it actually needs (see below) instead of trading
   away that margin for chat's gain. board_col itself is further split
   for the board and its side controls (see _render_board_panel), and
   needs real room for both: at the old 1300px/3:2 split, the board's
   fixed 340px pixel size (see board_component's own size param) didn't
   fit in its ~291px actual sub-column, and chessboard.js has no way to
   shrink to match -- it just overflowed into a horizontal scrollbar.

   align-self: flex-start, not margin -- stMain (this element's parent)
   is a column flex container with align-items: center of its own, which
   was centering this capped-width block with equal empty space on both
   sides regardless of any margin set here (confirmed live: both margins
   computed to 0px, yet the block still sat 310px from the left edge on a
   1920px-wide viewport). align-self overrides align-items for this one
   flex item without touching stMain's rule for anything else it lays
   out, and is what actually pins the page to its own left padding
   instead of floating it in the middle of the window. */
div[data-testid="stMainBlockContainer"] {
    max-width: 1700px;
    align-self: flex-start;
    /* Streamlit's own defaults here are 96px top / 160px bottom -- built
       for a page meant to be scrolled, not this one, where the whole
       point is fitting a fresh, question-less screen inside a laptop
       viewport with no scroll at all (confirmed live: at a 14" MacBook's
       default logical resolution, the unscrolled page needed ~1200px
       against an ~980px-tall screen, before browser chrome even eats into
       that). Trimmed hard, not just tightened -- this is by far the
       single biggest reclaimable chunk on the page. */
    padding-top: 36px;
    padding-bottom: 12px;
}

/* The reliability caveat (render_main_screen, "Answers synthesize...") has
   to stay on the page per CLAUDE.md, but as a quiet footnote, not a second
   headline right under the title -- smaller than st.caption's own default
   size. Margin trimmed too: it sits snug under the title row, part of
   shrinking the page's whole top section, not just this one line.

   #6E5C46, not the #8A7860 this was first shipped at: that lighter shade
   measured 3.49:1 against the page background, below WCAG AA's 4.5:1 floor
   for text this small -- "quieter" was pushed too far into "harder to
   read," not just "less prominent." #6E5C46 computes to 5.25:1, real
   margin above the floor rather than sitting right on it, while still
   reading as a muted footnote next to the full-strength body text
   elsewhere on the page. */
.st-key-reliability_note {
    margin-top: -8px;
    margin-bottom: 4px;
}
.st-key-reliability_note [data-testid="stCaptionContainer"] {
    font-size: 11px;
    color: #6E5C46;
}

/* board_col (render_main_screen's board_panel wrapper, see chat.py) is the
   real floor on the whole page's height -- its own natural content ran
   taller than chat_col's even after the message panel was shrunk, so this
   column, not that panel, decides how tall the row ends up. Streamlit's
   own ~16px gap between every element here (board, Evaluate, the ask
   form, the divider, Find related resources) adds up across six items;
   tightened but not eliminated -- these are still visually distinct
   controls, not one continuous block. The divider specifically carries
   much more of its own margin than a single ruled line needs (measured
   live: a 49px-tall element for what renders as 1px of visible line) and
   gets trimmed further on top of the shared gap reduction. */
.st-key-board_panel {
    gap: 8px;
}
.st-key-board_panel hr {
    /* 4px on both sides measured out to a near-zero gap in practice --
       "Find related resources" sat close enough to overlap the line
       itself. More room below than above: this line's real job is
       separating the board/eval/ask-position controls above it from
       "Find related resources" below, so that side needs the clearer
       break. */
    margin: 4px 0 16px;
}

/* The "Ask about this position" form (position_question_form) is the one
   place left with a visible rectangular border drawn around the page's
   own wood-grain background -- Streamlit's default stForm styling. That
   combination reads as an unfinished card (a frame around a fill that
   doesn't match it), not as intentional spacing, and it sits directly
   between two un-boxed controls (Evaluate above, Find related resources
   below) that already read fine showing the page background because
   neither one draws a border around itself. Dropping the border makes
   this consistent with both neighbors instead of being the one bordered
   wrapper among plain ones -- the input and button inside still carry
   their own distinct styling regardless, so legibility doesn't depend on
   the outer box at all. */
div[data-testid="stForm"] {
    border: none;
    padding: 0;
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
