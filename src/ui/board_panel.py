"""The board column: the board, its controls, and the two ways to ask about
the position shown. chat.py lays this panel out; nothing here imports
chat.py.
"""

from __future__ import annotations

import chess
import streamlit as st

from src.ui.board_component import chess_board
from src.ui.help_text import FEN_HELP, PLY_HELP

# Largest board edge in CSS pixels. The board shrinks below this when its
# column is narrower (see wiring.js), e.g. to 328px on a 360px-wide phone.
BOARD_SIZE_PX = 340


def _attempt_move(source: str, target: str) -> None:
    """Validate and record a {from, to} move from the board against
    python-chess, which has the final say on legality even though the
    browser already checked it with chess.js. Promotions default to a queen,
    matching the browser.

    source and target come from the browser, so anything that isn't valid
    UCI is rejected like an illegal move instead of raising.

    board_generation is bumped for every move, accepted or rejected, so the
    board always hears back (see chess_board()).
    """
    board: chess.Board = st.session_state.board
    try:
        move = chess.Move.from_uci(source + target)
        if move not in board.legal_moves:
            move = chess.Move.from_uci(source + target + "q")
    except chess.InvalidMoveError:
        move = None
    if move is not None and move in board.legal_moves:
        board.push(move)
        st.session_state.last_illegal_attempt = None
    else:
        st.session_state.last_illegal_attempt = (source, target)
    st.session_state.board_generation += 1


# Wording per python-chess Termination reason; variant-only reasons can't
# occur on a chess.Board.
_TERMINATION_TEXT = {
    chess.Termination.CHECKMATE: "Checkmate",
    chess.Termination.STALEMATE: "Draw by stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "Draw by insufficient material",
    chess.Termination.SEVENTYFIVE_MOVES: "Draw by the seventy-five-move rule",
    chess.Termination.FIVEFOLD_REPETITION: "Draw by fivefold repetition",
    chess.Termination.FIFTY_MOVES: "the fifty-move rule",
    chess.Termination.THREEFOLD_REPETITION: "threefold repetition",
}


def game_status(board: chess.Board) -> str | None:
    """A finished game's result, or a draw the side to move could claim;
    None while the game is in progress.

    Under FIDE rules threefold repetition and the fifty-move rule are draws
    a player may claim, while fivefold repetition and the seventy-five-move
    rule end the game, so the two are worded differently.

    Repetition and move-count draws need the move history, which a position
    loaded from a FEN (game replay) doesn't have; checkmate, stalemate and
    insufficient material are still detected there.
    """
    outcome = board.outcome()
    if outcome is not None:
        if outcome.termination is chess.Termination.CHECKMATE:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            return f"Checkmate. {winner} wins."
        return f"{_TERMINATION_TEXT[outcome.termination]}."

    claimable = board.outcome(claim_draw=True)
    if claimable is not None:
        return f"A draw can be claimed here, by {_TERMINATION_TEXT[claimable.termination]}."
    return None


def _render_board_controls(
    current_board: chess.Board, status: str | None, replaying: bool, disabled: bool, *, layout: str
) -> None:
    """Turn, FEN, Reset and Undo. `layout` keeps the two copies' widget keys
    distinct (see render_board_panel)."""
    if status is None:
        st.caption(f"Turn: {'White' if current_board.turn else 'Black'}")
    # Inline code in a caption, matching the transcript's position captions.
    st.caption(f"FEN: `{current_board.fen()}`", help=FEN_HELP)
    if replaying:
        return
    if st.button("Reset board", key=f"reset_board_{layout}", disabled=disabled):
        st.session_state.board = chess.Board()
        st.session_state.last_illegal_attempt = None
        st.session_state.board_generation += 1
        st.rerun()
    if st.button(
        "Undo last move",
        key=f"undo_move_{layout}",
        disabled=disabled or not current_board.move_stack,
    ):
        current_board.pop()
        st.session_state.board_generation += 1
        st.rerun()


def render_board_panel(*, disabled: bool = False) -> None:
    """The board, Reset/Undo, an Evaluate button and an "ask about this
    position" box. While a recommended game is replayed
    (st.session_state.game_path is set), the board is read-only and steps
    through that game, and Evaluate and Ask apply to the ply shown.

    The browser plays legal moves itself (chess.js); each one still reaches
    _attempt_move, where python-chess checks and records it.

    Moving is turned off during replay: st.session_state.board holds the
    free-play game behind the replay, and a move would change it unseen.

    The question buttons set st.session_state.pending_question and rerun,
    so the answer renders in the chat column (see chat._submit_question).

    disabled=True, set while an answer generates, makes every control inert:
    any click would cancel the answer.
    """
    replaying = st.session_state.game_path is not None
    if replaying:
        current_board = chess.Board(st.session_state.game_path[st.session_state.game_path_index])
    else:
        # The stored object itself, so Undo's pop() below changes it.
        current_board = st.session_state.board

    # Controls sit beside the board on wide screens so the board column
    # isn't much taller than the chat; Streamlit stacks them on phones.
    board_display_col, controls_col = st.columns([5, 2])

    with board_display_col:
        # The container's key is the tour's spotlight target. The board is
        # keyed too, so it persists across reruns (see chess_board()).
        with st.container(key="tutorial_board_target"):
            drop = chess_board(
                current_board.fen(),
                size=BOARD_SIZE_PX,
                generation=st.session_state.board_generation,
                draggable=not replaying and not disabled,
                key="chess_board",
            )
        if drop is not None and not replaying and not disabled:
            _attempt_move(drop["from"], drop["to"])
            st.rerun()

    status = game_status(current_board)

    with controls_col:
        # Move feedback and replay controls stay outside the "Board
        # controls" expander, so collapsing it on a phone never hides them.
        if status is not None:
            st.caption(status)

        if replaying:
            index = st.session_state.game_path_index
            last_index = len(st.session_state.game_path) - 1
            st.caption(
                f"{st.session_state.game_path_label}, ply {index} of {last_index}",
                help=PLY_HELP,
            )
            # Stacked, since the column is too narrow for two buttons side by
            # side. The arrow-key shortcuts don't fire while typing in a box.
            if st.button("Previous", shortcut="Left", disabled=disabled or index == 0):
                st.session_state.game_path_index -= 1
                st.session_state.board_generation += 1
                st.rerun()
            if st.button("Next", shortcut="Right", disabled=disabled or index == last_index):
                st.session_state.game_path_index += 1
                st.session_state.board_generation += 1
                st.rerun()
            if st.button("Exit replay", disabled=disabled):
                st.session_state.game_path = None
                st.session_state.game_path_index = 0
                st.session_state.game_path_label = None
                st.session_state.board_generation += 1
                st.rerun()
        else:
            if current_board.is_game_over():
                # No move is legal once the game is over, so point to Reset
                # instead of the usual "try again".
                st.info(f"{status} Reset the board to play again.")
            elif st.session_state.last_illegal_attempt is not None:
                st.warning("That move isn't legal. Try again.")

        # Rendered twice; theme.css shows a plain block on wide screens and
        # a collapsed expander on phones. (CSS can show a closed expander's
        # contents but not make them clickable.)
        with st.container(key="board_controls_wide"):
            _render_board_controls(current_board, status, replaying, disabled, layout="wide")
        with st.container(key="board_controls_phone"), st.expander("Board controls"):
            _render_board_controls(current_board, status, replaying, disabled, layout="phone")

    if st.button(
        "Evaluate this position with Stockfish",
        type="primary",
        key="evaluate_position",
        disabled=disabled,
    ):
        st.session_state.pending_question = (
            "Evaluate this chess position and tell me the best move. "
            "Use the engine, don't just guess.",
            current_board.fen(),
        )
        st.rerun()

    # Narrower than the Evaluate button above. The keyed container is the
    # tour's target: st.form puts its own key on the submit button, not the
    # form, and the two keys must differ.
    form_col, _ = st.columns([3, 1])
    with (
        form_col,
        st.container(key="position_question_form"),
        st.form("position_question_form_widget", clear_on_submit=True),
    ):
        # The label is hidden but kept for screen readers.
        position_question = st.text_input(
            "Ask about this position...",
            placeholder="Ask about this position...",
            label_visibility="collapsed",
        )
        asked = st.form_submit_button("Ask", disabled=disabled)
    if asked and position_question:
        st.session_state.pending_question = (position_question, current_board.fen())
        st.rerun()
