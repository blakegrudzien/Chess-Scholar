"""Streamlit frontend for the chess RAG assistant. Ties together all four
layers via the agent in src/agent/chess_agent.py.

Known reliability caveats (see CLAUDE.md) surfaced directly in the UI:
- Chat answers synthesize retrieved human text and engine output; they are
  not the model's own independent tactical judgment.
- The "find similar games" comparison (Layer 4) is an approximate,
  illustrative match on exact opening moves, not a positional analysis.
"""

from __future__ import annotations

import html
import itertools
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

# `streamlit run src/app.py` puts src/ itself on sys.path, not the project
# root, so the absolute `from src....` imports below need this first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
import chess  # noqa: E402
import chess.svg  # noqa: E402
import psycopg2.pool  # noqa: E402
import streamlit as st  # noqa: E402

from src.agent.chess_agent import ask  # noqa: E402
from src.embeddings.voyage_embedder import get_voyage_client  # noqa: E402
from src.engine.engine_pool import EnginePool  # noqa: E402
from src.engine.stockfish_eval import get_engine_path  # noqa: E402
from src.ingestion.db_loader import get_connection_pool  # noqa: E402
from src.ingestion.pgn_parser import parse_pgn  # noqa: E402
from src.recommendation.pipeline import (  # noqa: E402
    ChessbaseGameRecommendation,
    LichessStudyRecommendation,
    recommend_resources,
)

# CPU-bound: each concurrent evaluation pins a core for the search duration,
# so this is sized to the deployment's compute, not to how many users we'd
# like to serve. Bump alongside the hosting tier, not in isolation.
ENGINE_POOL_SIZE = 2

# Anthropic's raw text deltas arrive in relatively large pieces (a clause or
# sentence at a time), not the smooth word-by-word reveal seen in Claude.ai
# or ChatGPT -- that reveal is a client-side pacing effect, not a property
# of the network chunks. This regex re-splits each delta into word-sized
# pieces (including leading and trailing whitespace, so pieces concatenate
# back to the exact original text) so it can be paced the same way.
_WORD_SPLIT_RE = re.compile(r"\s*\S+\s*")
STREAM_WORD_DELAY_SECONDS = 0.02

# Streamlit's default chat avatars are a generic face/robot Material icon --
# a visual cue that reads as "generic AI chatbot," working against the
# deliberately non-modern, non-AI-flavored identity built for this app.
# Chess pieces are already this app's own icon language (the board tab
# renders pieces via chess.svg), not a new decoration introduced just for
# the avatars.
#
# st.chat_message's avatar param only accepts emoji from Streamlit's own
# curated allow-list (streamlit.emojis.ALL_EMOJIS), not arbitrary Unicode --
# confirmed by reading _process_avatar_input directly: of the 12 chess piece
# glyphs (U+2654-265F), only "black pawn" (U+265F) happens to be in that
# list, so passing e.g. the knight glyph raised StreamlitAPIException
# ("Failed to load the provided avatar value as an image") instead of
# rendering. Passing a raw SVG string sidesteps the allow-list entirely --
# image_to_url() special-cases strings that look like <svg ...> markup and
# inlines them as a data URI -- and reuses chess.svg.piece(), the same
# renderer already used for the board tab, instead of depending on emoji
# font coverage across viewers' systems.
_CHAT_AVATARS = {
    "user": chess.svg.piece(chess.Piece(chess.PAWN, chess.BLACK), size=32),
    "assistant": chess.svg.piece(chess.Piece(chess.BISHOP, chess.WHITE), size=32),
}

st.set_page_config(page_title="Chess RAG Assistant", layout="wide")

# Portfolio demo backed by a personal ChessBase export; keep it out of search
# engine indexes rather than relying on the URL being merely unlisted.
# st.markdown's unsafe_allow_html doesn't execute <script> tags (React sets
# innerHTML), so this goes through st.html's explicit script-execution opt-in
# instead, rendered inside a sandboxed iframe nested one level inside
# Streamlit's own app frame -- window.top (not window.parent) is needed to
# reach the real top document.
st.html(
    """<script>
    var meta = window.top.document.createElement('meta');
    meta.name = 'robots';
    meta.content = 'noindex, nofollow';
    window.top.document.head.appendChild(meta);
    </script>""",
    unsafe_allow_javascript=True,
)

# Theme extras that .streamlit/config.toml can't express: config.toml covers
# colors, all three font roles, heading weights, and base radius natively
# (see that file's own comments), but a specific inset treatment on the chat
# input and the recommendation cards' accent rail both need real CSS.
#
# The chat-input selector targets Streamlit's own generated Emotion classes
# for stChatInputTextArea's wrapper, confirmed against the live rendered DOM
# (data-testid alone has no visible box, its ancestor wrapper does) rather
# than guessed -- these are an internal, unversioned implementation detail,
# not a public API, and may need re-verifying after a Streamlit upgrade.
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

/* layout="wide" is needed for the board tab's 8-column grid, but it also
   stretches chat prose edge to edge on a wide window -- well past a
   comfortable reading measure. Capping just the panel that contains the
   chat input (rather than assuming DOM order among the three tab panels,
   which didn't actually match :first-of-type in practice) keeps the board
   and upload tabs at full width while giving chat text a fixed column.
   No margin: auto -- the panel already starts at the same left edge as
   the page title and tab bar above it (same parent padding), so capping
   width alone keeps that edge aligned instead of centering the panel
   into a column visually disconnected from the header above it. */
div[data-testid="stTabPanel"]:has([data-testid="stChatInputTextArea"]) {
    max-width: 720px;
}
</style>
""")


@st.cache_resource
def _get_db_pool() -> psycopg2.pool.ThreadedConnectionPool:
    return get_connection_pool()


@st.cache_resource
def _get_engine_pool() -> EnginePool:
    return EnginePool(get_engine_path(), size=ENGINE_POOL_SIZE)


@st.cache_resource
def _get_voyage():
    return get_voyage_client()


@st.cache_resource
def _get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def ask_agent(
    question: str,
    on_step: Callable[[str], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    return ask(
        question,
        _get_db_pool(),
        _get_engine_pool(),
        _get_voyage(),
        client=_get_anthropic_client(),
        on_step=on_step,
        on_chunk=on_chunk,
    )


def _ask_with_status(question: str) -> str:
    """Run ask_agent, showing each tool-calling step live in an st.status
    panel and streaming the final answer into view as it is generated,
    rather than a blank spinner followed by the whole answer appearing at
    once. Renders the final answer itself, so callers should not render
    `answer` again afterward.

    A tool-calling turn and the final answer both start with plain text
    (the system prompt asks for a one-sentence rationale before every tool
    call), and the only way to tell them apart is whether a tool_use block
    shows up by the end of that turn -- so every turn's text streams into
    answer_area first. If the turn turns out to have called a tool, that
    text becomes the status line and answer_area is cleared for the next
    turn; if not, it was the final answer, already fully displayed by the
    time this returns.

    Each delta is re-split into word-sized pieces and revealed one at a
    time with a short pause between them (see STREAM_WORD_DELAY_SECONDS),
    rather than written all at once -- the raw deltas from the API arrive
    in clause-or-sentence-sized pieces, so writing them straight through
    looks like it is appearing in chunks rather than being typed. This adds
    a small amount of wall-clock time to how long the last word takes to
    appear, in exchange for text appearing continuously throughout instead
    of in a few jumps.

    The stop button doesn't need explicit click handling: Streamlit treats
    interactions as implicit yield points during a running script, so any
    click while this is in progress interrupts it at the next word reveal.
    """
    status = st.status("Thinking...", expanded=True)
    stop_placeholder = st.empty()
    stop_placeholder.button("Stop generating", key="stop_generating")
    answer_area = st.empty()

    accumulated_text = ""

    def on_chunk(delta: str) -> None:
        nonlocal accumulated_text
        pieces = _WORD_SPLIT_RE.findall(delta) or [delta]
        for piece in pieces:
            accumulated_text += piece
            answer_area.markdown(accumulated_text)
            time.sleep(STREAM_WORD_DELAY_SECONDS)

    def on_step(text: str) -> None:
        nonlocal accumulated_text
        status.write(text)
        accumulated_text = ""
        answer_area.empty()

    answer = ask_agent(question, on_step=on_step, on_chunk=on_chunk)

    status.update(label="Done", state="complete", expanded=False)
    stop_placeholder.empty()
    return answer


def _render_resource_recommendations() -> None:
    """Offers to look up related Lichess studies and corpus games for the
    most recent question, and renders whatever comes back.

    Only ever shown for the latest exchange, not every past one in the
    history -- keeps the UI focused on what is actually in view rather than
    accumulating a lookup button per message.

    st.session_state.resource_recommendations is the sentinel for "already
    looked up this question": None means not yet requested (show the
    button), a list (possibly empty, meaning nothing was relevant) means it
    has been. Reset to None right after a new answer is appended in
    render_chat_tab, so a fresh question always gets a fresh button.
    """
    history = st.session_state.chat_history
    if len(history) < 2 or history[-1][0] != "assistant":
        return
    question = history[-2][1]

    if st.session_state.resource_recommendations is None:
        if not st.button("Find related resources", key="find_resources", type="primary"):
            return
        # Doherty threshold: this call runs a multi-step tool-calling loop
        # and regularly takes 20+ seconds in practice, past the point where
        # a bare spinner keeps people's attention -- a status panel with a
        # task description is the book's own prescribed pattern for waits
        # this long, matching the same visual language _ask_with_status
        # already uses for the chat agent. This version isn't wired to the
        # pipeline's actual step-by-step tool calls the way the chat one is
        # (that would mean threading on_step callbacks through
        # recommend_resources itself); it's a static but honest description
        # of the stages involved, not live progress.
        with st.status("Looking for related resources...", expanded=True) as status:
            status.write("Searching the study library for relevant chapters.")
            status.write("Checking whether a matching master game exists in the corpus.")
            st.session_state.resource_recommendations = recommend_resources(
                question, _get_db_pool(), _get_voyage(), client=_get_anthropic_client()
            )
            status.update(label="Done", state="complete", expanded=False)

    recommendations = st.session_state.resource_recommendations
    if not recommendations:
        st.caption("Nothing in the study library or corpus was a close enough match to recommend.")
        return

    for rec in recommendations:
        if isinstance(rec, LichessStudyRecommendation):
            st.html(f"""
            <div class="rec-card">
                <p class="rec-kind">Lichess study</p>
                <h4>{html.escape(rec.study_title)}</h4>
                <p class="rec-chapter">{html.escape(rec.chapter_name)}</p>
                <p class="rec-blurb">{html.escape(rec.blurb)}</p>
            </div>
            """)
            st.iframe(rec.embed_url, height=400)
        elif isinstance(rec, ChessbaseGameRecommendation):
            st.html(f"""
            <div class="rec-card">
                <p class="rec-kind">Chessbase game</p>
                <h4>{html.escape(rec.white)} vs {html.escape(rec.black)}</h4>
                <p class="rec-chapter">{html.escape(rec.event)}</p>
                <p class="rec-blurb">{html.escape(rec.blurb)}</p>
            </div>
            """)
            st.caption("From the local corpus, moves only -- no commentary included.")
            st.code(rec.pgn, language=None)


def render_chat_tab() -> None:
    st.caption(
        "Answers synthesize retrieved human commentary and engine output -- "
        "not the model's own independent chess judgment."
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "resource_recommendations" not in st.session_state:
        st.session_state.resource_recommendations = None

    for role, content in st.session_state.chat_history:
        with st.chat_message(role, avatar=_CHAT_AVATARS[role]):
            st.markdown(content)

    question = st.chat_input("Ask about openings, positions, or chess history...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user", avatar=_CHAT_AVATARS["user"]):
            st.markdown(question)
        with st.chat_message("assistant", avatar=_CHAT_AVATARS["assistant"]):
            answer = _ask_with_status(question)
        st.session_state.chat_history.append(("assistant", answer))
        st.session_state.resource_recommendations = None

    _render_resource_recommendations()


def handle_square_click(square: chess.Square) -> None:
    board: chess.Board = st.session_state.board
    selected = st.session_state.selected_square

    if selected is None:
        piece = board.piece_at(square)
        if piece is not None and piece.color == board.turn:
            st.session_state.selected_square = square
        return

    if selected == square:
        st.session_state.selected_square = None
        return

    move = chess.Move(selected, square)
    if move not in board.legal_moves:
        move = chess.Move(selected, square, promotion=chess.QUEEN)
    if move in board.legal_moves:
        board.push(move)
        st.session_state.last_illegal_attempt = None
    else:
        st.session_state.last_illegal_attempt = (selected, square)
    st.session_state.selected_square = None


def render_board_tab() -> None:
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "selected_square" not in st.session_state:
        st.session_state.selected_square = None
    if "last_illegal_attempt" not in st.session_state:
        st.session_state.last_illegal_attempt = None

    board: chess.Board = st.session_state.board
    selected = st.session_state.selected_square

    # Jakob's Law: chess players' existing mental model (lichess, chess.com)
    # is select a piece, see where it can legally go, see what the last move
    # was. None of that was previously shown -- only the selected square
    # itself was highlighted. `squares` is python-chess's own mechanism for
    # legal-move-style highlighting, distinct from the flat single-color
    # `fill` used for the current selection, so the two read as different
    # things rather than one undifferentiated blob of color.
    fill = {}
    legal_destinations = None
    if selected is not None:
        fill[selected] = "#A87F3F"  # brass -- matches the app's own accent
        legal_destinations = chess.SquareSet(
            move.to_square for move in board.legal_moves if move.from_square == selected
        )
    last_move = board.move_stack[-1] if board.move_stack else None

    board_col, controls_col = st.columns([2, 1])
    with board_col:
        svg = chess.svg.board(
            board=board,
            size=400,
            fill=fill,
            squares=legal_destinations,
            lastmove=last_move,
        )
        st.components.v1.html(svg, height=430)
        st.caption(f"Turn: {'White' if board.turn else 'Black'}")
        st.code(board.fen(), language=None)
        if st.session_state.last_illegal_attempt is not None:
            st.warning("That move isn't legal. Try again.")

    with controls_col:
        if st.button("Reset board"):
            st.session_state.board = chess.Board()
            st.session_state.selected_square = None
            st.session_state.last_illegal_attempt = None
            st.rerun()
        if st.button("Undo last move", disabled=not board.move_stack):
            board.pop()
            st.session_state.selected_square = None
            st.rerun()

    st.write("Click a square to select a piece, then click a destination square to move it.")
    for rank in range(8, 0, -1):
        cols = st.columns(8)
        for i, file_letter in enumerate("abcdefgh"):
            square = chess.parse_square(f"{file_letter}{rank}")
            piece = board.piece_at(square)
            label = piece.unicode_symbol() if piece else "·"
            with cols[i]:
                if st.button(label, key=f"sq_{file_letter}{rank}"):
                    handle_square_click(square)
                    st.rerun()

    st.divider()
    if st.button("Evaluate this position with Stockfish", type="primary"):
        _ask_with_status(
            f"Evaluate this chess position and tell me the best move: {board.fen()}. "
            "Use the engine, don't just guess."
        )


def render_pgn_upload_tab() -> None:
    st.caption(
        "This comparison is **illustrative, not authoritative** -- it matches on "
        "exact opening moves, not true positional similarity."
    )
    uploaded = st.file_uploader("Upload a PGN of your own game", type=["pgn"])
    if uploaded is None:
        return

    with tempfile.NamedTemporaryFile(suffix=".pgn", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        # Only the first game is ever used below, so pull at most two from
        # the generator: the first to use, and a second only to learn
        # whether there's more than one, without fully parsing an upload
        # (untrusted input) that could contain a large number of games.
        first_two_games = list(itertools.islice(parse_pgn(tmp_path, source="user_upload"), 2))
    finally:
        os.unlink(tmp_path)

    if not first_two_games:
        st.error("Couldn't find a game in that file.")
        return

    if len(first_two_games) > 1:
        st.info("This file has more than one game. Only the first is analyzed.")

    game = first_two_games[0]
    move_sans = [m.move_san for m in game.moves]
    st.write(f"Parsed **{game.white or '?'} vs {game.black or '?'}** ({len(move_sans)} plies)")
    preview = " ".join(move_sans[:20]) + (" ..." if len(move_sans) > 20 else "")
    st.code(preview, language=None)

    if st.button("Find similar games in the corpus", type="primary"):
        question = (
            "Here is a game I played, as a list of moves in order: "
            f"{', '.join(move_sans)}. Find similar games in the corpus and give me "
            "an illustrative comparison."
        )
        _ask_with_status(question)


def main() -> None:
    st.title("Chess RAG Assistant")
    chat_tab, board_tab, upload_tab = st.tabs(["Chat", "Board Position", "Analyze Your Game"])
    with chat_tab:
        render_chat_tab()
    with board_tab:
        render_board_tab()
    with upload_tab:
        render_pgn_upload_tab()


main()
