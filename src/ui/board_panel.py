"""The board column: a draggable board, its controls, and the two ways to
ask a question about whatever position is currently shown.

Split out of chat.py, which owns the conversation side of the same screen.
The dependency runs one way -- chat.py composes this panel into its layout,
and nothing here imports chat.py -- so the board's state machine (free play
vs. replaying a recommended game) can be read without the chat transcript's
rendering logic interleaved through it.
"""

from __future__ import annotations

import chess
import streamlit as st

from src.ui.board_component import chess_board
from src.ui.help_text import FEN_HELP, PLY_HELP

# Board edge in CSS pixels. chessboard.js treats this as a floor it will not
# shrink below regardless of container width, which is what makes the column
# ratio below load-bearing rather than cosmetic.
BOARD_SIZE_PX = 340


def _attempt_move(source: str, target: str) -> None:
    """Validate a drag-and-drop {from, to} pair against python-chess, the
    single source of truth for move legality in this app. Promotions default
    to a queen.

    source/target arrive from chess_board() as plain strings chosen by the
    component's JavaScript. In normal use they are a real drag's two
    algebraic squares, but nothing on the Python side enforces that shape, so
    a pair that isn't valid UCI input at all (as opposed to merely an illegal
    move) is rejected the same honest way rather than raising
    InvalidMoveError through to Streamlit's full-traceback error page.

    board_generation is bumped even when the move is rejected. An illegal
    drop leaves board.fen() textually identical to what it was before, so
    without a distinct generation value the component has no signal to
    re-render and snap the piece back -- see chess_board()'s docstring.
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


# Human-readable text per python-chess Termination reason. Only the
# standard-chess terminations appear here; the variant-specific ones
# (VARIANT_WIN and friends) can't arise from a chess.Board.
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
    """A finished game's result, or a draw the side to move could claim --
    None while the game is simply in progress.

    Claimed and automatic draws are reported differently on purpose. Under
    FIDE rules threefold repetition and the fifty-move rule are a player's
    right to claim, not an automatic result, while fivefold repetition and
    the seventy-five-move rule end the game on their own. Collapsing the two
    into one "Draw" message is a common chess-app bug.

    Repetition and move-counter draws need the move history to detect, so
    they can't be found in a position loaded from a bare FEN (game replay,
    where board.move_stack is empty). Checkmate, stalemate and insufficient
    material are properties of the position alone and are still caught there.
    """
    outcome = board.outcome()
    if outcome is not None:
        if outcome.termination is chess.Termination.CHECKMATE:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            return f"Checkmate -- {winner} wins."
        return f"{_TERMINATION_TEXT[outcome.termination]}."

    claimable = board.outcome(claim_draw=True)
    if claimable is not None:
        return f"A draw can be claimed here, by {_TERMINATION_TEXT[claimable.termination]}."
    return None


def render_board_panel() -> None:
    """A draggable board reflecting the position under discussion, plus a
    quick-eval button, Reset/Undo, and a free-text "ask about this position"
    form. While replaying a recommended game (st.session_state.game_path is
    not None) the board is read-only and steps through that game's moves
    instead, with Evaluate/Ask-position operating on whatever ply is shown.

    Optimistic UI, with no client-side legality check: chess_board() lets a
    drop land wherever it was dropped, and _attempt_move validates it against
    python-chess afterward. An illegal drop leaves st.session_state.board
    unchanged, so the next render's still-unmoved data reverts the piece via
    chessboard.js's own diffing.

    Dragging is disabled outright during replay rather than left enabled and
    ignored: st.session_state.board is the free-play board sitting behind the
    visible replay position, not a copy, so a drag left live there would
    corrupt it invisibly.

    Neither question button calls the agent directly. Both stash a
    (question, fen) pair in st.session_state.pending_question and rerun, so
    the submission -- and the live status/streaming UI it renders -- happens
    from the chat column on the next script pass rather than inside this
    narrow one. See chat._submit_question's docstring.
    """
    replaying = st.session_state.game_path is not None
    if replaying:
        current_board = chess.Board(st.session_state.game_path[st.session_state.game_path_index])
    else:
        # The same object held in session_state, not a copy, so Undo's
        # current_board.pop() below mutates the stored board in place.
        current_board = st.session_state.board

    # Turn/FEN/Reset/Undo (or, during replay, the ply label and Prev/Next)
    # sit beside the board rather than stacked below it, to close the height
    # gap between this column and the chat column. The ratio has to leave at
    # least BOARD_SIZE_PX for the board itself: chessboard.js won't shrink
    # below that, so a narrower share overflows at laptop widths.
    board_display_col, controls_col = st.columns([5, 2])

    with board_display_col:
        # Wrapped in a keyed container rather than passing chess_board() its
        # own key=, so the tutorial overlay has a stable
        # `.st-key-tutorial_board_target` selector to spotlight. Keying the
        # component itself would make Streamlit treat it as one persistent
        # instance and stop remounting it when `data` changes, which is the
        # mechanism board_generation relies on for illegal-move snapback and
        # Undo (see chess_board()'s docstring).
        with st.container(key="tutorial_board_target"):
            drop = chess_board(
                current_board.fen(),
                size=BOARD_SIZE_PX,
                generation=st.session_state.board_generation,
                draggable=not replaying,
            )
        if drop is not None and not replaying:
            _attempt_move(drop["from"], drop["to"])
            st.rerun()

    status = game_status(current_board)

    with controls_col:
        # Whose turn it is stops being the useful thing to say once the game
        # is decided, so the result takes that line's place.
        if status is None:
            st.caption(f"Turn: {'White' if current_board.turn else 'Black'}")
        else:
            st.caption(status)
        # A caption with inline code rather than st.code(): a single short
        # FEN doesn't need a full code panel's padding and copy button, and
        # this matches the "Position: `{fen}`" captions in the transcript.
        st.caption(f"FEN: `{current_board.fen()}`", help=FEN_HELP)

        if replaying:
            index = st.session_state.game_path_index
            last_index = len(st.session_state.game_path) - 1
            st.caption(
                f"{st.session_state.game_path_label} -- ply {index} of {last_index}",
                help=PLY_HELP,
            )
            # Stacked rather than side by side: this column is too narrow for
            # two buttons abreast. shortcut= is a native st.button parameter;
            # it stays out of the way while a text input has focus (arrow
            # keys still move the text cursor there) and fires globally
            # otherwise, so replay navigation needs no keyboard component.
            if st.button("Previous", shortcut="Left", disabled=index == 0):
                st.session_state.game_path_index -= 1
                st.session_state.board_generation += 1
                st.rerun()
            if st.button("Next", shortcut="Right", disabled=index == last_index):
                st.session_state.game_path_index += 1
                st.session_state.board_generation += 1
                st.rerun()
            if st.button("Exit replay"):
                st.session_state.game_path = None
                st.session_state.game_path_index = 0
                st.session_state.game_path_label = None
                st.session_state.board_generation += 1
                st.rerun()
        else:
            if current_board.is_game_over():
                # Every drag is rejected once the game is over, so the
                # generic "try again" below would be actively misleading --
                # there is no legal move to try. Reset is the way forward.
                st.info(f"{status} Reset the board to play again.")
            elif st.session_state.last_illegal_attempt is not None:
                st.warning("That move isn't legal. Try again.")

            if st.button("Reset board"):
                st.session_state.board = chess.Board()
                st.session_state.last_illegal_attempt = None
                st.session_state.board_generation += 1
                st.rerun()
            if st.button("Undo last move", disabled=not current_board.move_stack):
                current_board.pop()
                st.session_state.board_generation += 1
                st.rerun()

    if st.button("Evaluate this position with Stockfish", type="primary", key="evaluate_position"):
        st.session_state.pending_question = (
            "Evaluate this chess position and tell me the best move. "
            "Use the engine, don't just guess.",
            current_board.fen(),
        )
        st.rerun()

    # Narrower than the full-width Evaluate button above it: a single short
    # question doesn't need the whole column, and the contrast with a
    # full-width primary action directly above makes the trailing space read
    # as deliberate.
    #
    # Wrapped in a keyed container for the same reason the board is.
    # st.form's own key= lands on its internal FormSubmitter button (it
    # renders as st-key-FormSubmitter-<form_key>-<button_label>), never on
    # the form element, so a `.st-key-position_question_form` selector can
    # only match a wrapper. The form keeps a separate key because Streamlit
    # forbids two elements sharing one key in the same run.
    form_col, _ = st.columns([3, 1])
    with (
        form_col,
        st.container(key="position_question_form"),
        st.form("position_question_form_widget", clear_on_submit=True),
    ):
        # A placeholder rather than a visible label, matching the main chat
        # input's look. label_visibility="collapsed" rather than omitting the
        # label: a real label is still required for screen readers, it just
        # isn't shown.
        position_question = st.text_input(
            "Ask about this position...",
            placeholder="Ask about this position...",
            label_visibility="collapsed",
        )
        asked = st.form_submit_button("Ask")
    if asked and position_question:
        st.session_state.pending_question = (position_question, current_board.fen())
        st.rerun()
