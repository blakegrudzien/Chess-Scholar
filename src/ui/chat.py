"""The Chat tab: chat transcript, the board panel alongside it, and the
resource-recommendation cards -- everything that reads or writes
st.session_state.chat_history/board.
"""

from __future__ import annotations

import html
import re
import time
from collections.abc import Callable

import chess
import chess.svg
import streamlit as st

from src.agent.chess_agent import ask
from src.recommendation.pipeline import (
    ChessbaseGameRecommendation,
    LichessStudyRecommendation,
    recommend_resources,
)
from src.ui.resources import get_anthropic_client, get_db_pool, get_engine_pool, get_voyage

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


def ask_agent(
    question: str,
    on_step: Callable[[str], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
    on_position: Callable[[str], None] | None = None,
) -> str:
    return ask(
        question,
        get_db_pool(),
        get_engine_pool(),
        get_voyage(),
        client=get_anthropic_client(),
        on_step=on_step,
        on_chunk=on_chunk,
        on_position=on_position,
    )


def ask_with_status(question: str) -> str:
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

    def on_position(fen: str) -> None:
        try:
            new_board = chess.Board(fen)
        except ValueError:
            return  # the model can pass a malformed FEN to the tool call
            # even though evaluate_chess_position's own validation later
            # rejects it for that turn -- leave the displayed board as it was.
        st.session_state.board = new_board
        # A fresh chess.Board(fen) has no move history, so a selected square
        # or illegal-move warning from before this update no longer
        # corresponds to anything real on the new board.
        st.session_state.selected_square = None
        st.session_state.last_illegal_attempt = None

    answer = ask_agent(question, on_step=on_step, on_chunk=on_chunk, on_position=on_position)

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
    _submit_question, so a fresh question always gets a fresh button.
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
        # this long, matching the same visual language ask_with_status
        # already uses for the chat agent. This version isn't wired to the
        # pipeline's actual step-by-step tool calls the way the chat one is
        # (that would mean threading on_step callbacks through
        # recommend_resources itself); it's a static but honest description
        # of the stages involved, not live progress.
        with st.status("Looking for related resources...", expanded=True) as status:
            status.write("Searching the study library for relevant chapters.")
            status.write("Checking whether a matching master game exists in the corpus.")
            st.session_state.resource_recommendations = recommend_resources(
                question, get_db_pool(), get_voyage(), client=get_anthropic_client()
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
            st.iframe(rec.embed_url, height=320)
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


def _submit_question(question: str, *, fen_context: str | None = None) -> None:
    """Append a user turn, get the agent's answer, and append it -- the one
    path both the main chat input and the board-side "ask about this
    position" box go through, so a follow-up question either way lands in
    the same transcript instead of two disconnected conversations.

    Only ever called from inside chat_col (see render_chat_tab and its
    "pending_board_question" handling below) -- never directly from a
    board-side button. ask_with_status renders live UI (a status panel, a
    streaming answer area) at whatever point in the layout it's called from,
    so calling it directly from a button inside the narrow board column
    would put that live UI in the board column instead of the chat
    transcript. Board-side triggers stash (question, fen) in
    st.session_state.pending_board_question and rerun instead, so the
    actual submission always happens from chat_col on the next script pass.

    fen_context, when given, is sent to the model as part of the question
    (see chess_agent.SYSTEM_PROMPT's "Current board position" instruction)
    and stored alongside the question in chat_history so the transcript can
    show which position a question was about, without the FEN prefix itself
    ever appearing as if the user had typed it.
    """
    st.session_state.chat_history.append(("user", question, fen_context))
    with st.chat_message("user", avatar=_CHAT_AVATARS["user"]):
        st.markdown(question)
        if fen_context is not None:
            st.caption(f"Position: `{fen_context}`")
    sent_question = (
        f"Current board position: {fen_context}\n\n{question}"
        if fen_context is not None
        else question
    )
    with st.chat_message("assistant", avatar=_CHAT_AVATARS["assistant"]):
        answer = ask_with_status(sent_question)
    st.session_state.chat_history.append(("assistant", answer, None))
    st.session_state.resource_recommendations = None


def render_chat_tab() -> None:
    st.caption(
        "Answers synthesize retrieved human commentary and engine output -- "
        "not the model's own independent chess judgment."
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "resource_recommendations" not in st.session_state:
        st.session_state.resource_recommendations = None
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "selected_square" not in st.session_state:
        st.session_state.selected_square = None
    if "last_illegal_attempt" not in st.session_state:
        st.session_state.last_illegal_attempt = None
    if "pending_board_question" not in st.session_state:
        st.session_state.pending_board_question = None

    chat_col, board_col = st.columns([3, 2])

    with chat_col:
        for role, content, fen in st.session_state.chat_history:
            with st.chat_message(role, avatar=_CHAT_AVATARS[role]):
                st.markdown(content)
                if fen is not None:
                    st.caption(f"Position: `{fen}`")

        pending = st.session_state.pending_board_question
        if pending is not None:
            st.session_state.pending_board_question = None
            pending_question, pending_fen = pending
            _submit_question(pending_question, fen_context=pending_fen)

        question = st.chat_input("Ask about openings, positions, or chess history...")
        if question:
            _submit_question(question)

    with board_col:
        _render_board_panel()


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


def _render_board_panel() -> None:
    """The board column: a compact display of the position under
    discussion, plus a quick-eval button and an expander for setting up a
    custom position to ask about.

    Neither button here calls ask_with_status directly -- both stash a
    (question, fen) pair in st.session_state.pending_board_question and
    st.rerun() instead, so the actual submission (and its live status/
    streaming UI) happens from inside chat_col on the next script pass, not
    in this narrow column. See _submit_question's docstring for why.
    """
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

    # size=300, down from the old standalone tab's 400 -- this board now
    # shares horizontal space with chat instead of having the full page
    # width to itself.
    svg = chess.svg.board(
        board=board,
        size=300,
        fill=fill,
        squares=legal_destinations,
        lastmove=last_move,
    )
    st.components.v1.html(svg, height=325)
    st.caption(f"Turn: {'White' if board.turn else 'Black'}")
    st.code(board.fen(), language=None)
    if st.session_state.last_illegal_attempt is not None:
        st.warning("That move isn't legal. Try again.")

    if st.button("Evaluate this position with Stockfish", type="primary"):
        st.session_state.pending_board_question = (
            "Evaluate this chess position and tell me the best move. "
            "Use the engine, don't just guess.",
            board.fen(),
        )
        st.rerun()

    with st.expander("Set up a position to ask about"):
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

        reset_col, undo_col = st.columns(2)
        with reset_col:
            if st.button("Reset board"):
                st.session_state.board = chess.Board()
                st.session_state.selected_square = None
                st.session_state.last_illegal_attempt = None
                st.rerun()
        with undo_col:
            if st.button("Undo last move", disabled=not board.move_stack):
                board.pop()
                st.session_state.selected_square = None
                st.rerun()

        with st.form("position_question_form", clear_on_submit=True):
            position_question = st.text_input("Ask about this position...")
            asked = st.form_submit_button("Ask")
        if asked and position_question:
            st.session_state.pending_board_question = (position_question, board.fen())
            st.rerun()

    st.divider()
    _render_resource_recommendations()
