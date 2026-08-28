"""Streamlit frontend for the chess RAG assistant. Ties together all four
layers via the agent in src/agent/chess_agent.py.

Known reliability caveats (see CLAUDE.md) surfaced directly in the UI:
- Chat answers synthesize retrieved human text and engine output; they are
  not the model's own independent tactical judgment.
- The "find similar games" comparison (Layer 4) is an approximate,
  illustrative match on exact opening moves, not a positional analysis.
"""

from __future__ import annotations

import os
import sys
import tempfile
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

# CPU-bound: each concurrent evaluation pins a core for the search duration,
# so this is sized to the deployment's compute, not to how many users we'd
# like to serve. Bump alongside the hosting tier, not in isolation.
ENGINE_POOL_SIZE = 2

st.set_page_config(page_title="Chess RAG Assistant", layout="wide")

# Portfolio demo backed by a personal ChessBase export; keep it out of search
# engine indexes rather than relying on the URL being merely unlisted.
# st.markdown's unsafe_allow_html doesn't execute <script> tags (React sets
# innerHTML), so this goes through components.html's sandboxed iframe instead.
# That iframe is nested one level inside Streamlit's own app frame, so
# window.top (not window.parent) is needed to reach the real top document.
st.components.v1.html(
    """<script>
    var meta = window.top.document.createElement('meta');
    meta.name = 'robots';
    meta.content = 'noindex, nofollow';
    window.top.document.head.appendChild(meta);
    </script>""",
    height=0,
)


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


def ask_agent(question: str, on_step=None) -> str:
    return ask(
        question,
        _get_db_pool(),
        _get_engine_pool(),
        _get_voyage(),
        client=_get_anthropic_client(),
        on_step=on_step,
    )


def _ask_with_status(question: str) -> str:
    """Run ask_agent, showing each tool-calling step live in an st.status
    panel instead of a blank spinner -- a full answer can take several
    sequential model round trips, so this both demonstrates the agent's
    layer routing and gives the wait something to look at.
    """
    with st.status("Thinking...", expanded=True) as status:
        answer = ask_agent(question, on_step=status.write)
        status.update(label="Done", state="complete", expanded=False)
    return answer


def render_chat_tab() -> None:
    st.caption(
        "Answers synthesize retrieved human commentary and engine output -- "
        "not the model's own independent chess judgment."
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("Ask about openings, positions, or chess history...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            answer = _ask_with_status(question)
            st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))


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

    fill = {}
    if st.session_state.selected_square is not None:
        fill[st.session_state.selected_square] = "#aaddaa"

    board_col, controls_col = st.columns([2, 1])
    with board_col:
        svg = chess.svg.board(board=board, size=400, fill=fill)
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
    if st.button("Evaluate this position with Stockfish"):
        answer = _ask_with_status(
            f"Evaluate this chess position and tell me the best move: {board.fen()}. "
            "Use the engine, don't just guess."
        )
        st.markdown(answer)


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
        games = list(parse_pgn(tmp_path, source="user_upload"))
    finally:
        os.unlink(tmp_path)

    if not games:
        st.error("Couldn't find a game in that file.")
        return

    game = games[0]
    move_sans = [m.move_san for m in game.moves]
    st.write(f"Parsed **{game.white or '?'} vs {game.black or '?'}** ({len(move_sans)} plies)")
    preview = " ".join(move_sans[:20]) + (" ..." if len(move_sans) > 20 else "")
    st.code(preview, language=None)

    if st.button("Find similar games in the corpus"):
        question = (
            "Here is a game I played, as a list of moves in order: "
            f"{', '.join(move_sans)}. Find similar games in the corpus and give me "
            "an illustrative comparison."
        )
        answer = _ask_with_status(question)
        st.markdown(answer)


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
