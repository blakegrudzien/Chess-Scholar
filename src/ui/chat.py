"""The conversation half of the app's single screen: the chat transcript
(PGN uploads included), the live progress and streaming UI for an answer in
flight, and the resource-recommendation cards. Owns
st.session_state.chat_history.

render_main_screen also lays out the board column, whose rendering and state
live in board_panel.py.
"""

from __future__ import annotations

import functools
import html
import io
import itertools
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

import anthropic
import chess
import chess.pgn
import chess.svg
import streamlit as st

from src.agent.chess_agent import MAX_PLY_WINDOW, OnPosition, ask
from src.ingestion.pgn_parser import parse_pgn
from src.recommendation.pipeline import (
    ChessbaseGameRecommendation,
    LichessStudyRecommendation,
    Recommendation,
    recommend_resources,
)
from src.ui.board_panel import render_board_panel
from src.ui.conversation_log import log_conversation_best_effort
from src.ui.example_answers import (
    EXAMPLE_QUESTIONS,
    CachedAnswer,
    cached_example_answer,
    replay_cached_answer,
)
from src.ui.help_text import FEN_HELP
from src.ui.input_limits import (
    MAX_QUESTION_CHARS,
    accept_question_length,
    get_shared_question_limit,
)
from src.ui.progress_bar import FINISH_SECONDS, finishing_bar_html, running_bar_html
from src.ui.resources import (
    get_anthropic_client,
    get_db_pool,
    get_engine_pool,
    get_lichess_http_client,
    get_lichess_pacer,
    get_lookup_executor,
    get_voyage,
)

logger = logging.getLogger(__name__)

# Anthropic streams text in clause-sized pieces. Splitting them into words
# (whitespace kept, so the pieces rejoin exactly) lets the first words of
# each turn be revealed at a steady typing pace.
_WORD_SPLIT_RE = re.compile(r"\s*\S+\s*")
STREAM_WORD_DELAY_SECONDS = 0.02

# Only the first words of a turn are paced; pacing a 900-word answer would
# add 15-20 seconds of artificial delay.
MAX_PACED_WORDS_PER_TURN = 40

# The [[diagram: <label>]] marker the system prompt asks the model to place
# where a diagram belongs (see _render_answer_content).
_DIAGRAM_MARKER_RE = re.compile(r"\[\[diagram:\s*(.*?)\s*\]\]")

# A main line plus a couple of sidelines; more stops being scannable.
MAX_INLINE_DIAGRAMS = 4

# Tall enough to match the board column, short enough that a fresh page fits
# a 14" laptop screen without scrolling. Phones drop the fixed height
# (theme.css).
MESSAGE_PANEL_HEIGHT_PX = 560

# Chess pieces instead of Streamlit's robot and face icons. Passed as SVG
# markup because st.chat_message only accepts emoji from its own allow-list,
# which includes just one chess glyph.
_CHAT_AVATARS = {
    "user": chess.svg.piece(chess.Piece(chess.PAWN, chess.BLACK), size=32),
    "assistant": chess.svg.piece(chess.Piece(chess.BISHOP, chess.WHITE), size=32),
}


def ask_agent(
    question: str,
    on_step: Callable[[str], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
    on_position: OnPosition | None = None,
    history: list[dict[str, str]] | None = None,
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
        history=history,
    )


def ask_with_status(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    ask_fn: Callable[..., str] | None = None,
) -> tuple[str, list[tuple[str, str | None]]]:
    """Answer `question` with live feedback: a progress bar, a status line
    per tool call, and the answer streaming in. Renders the finished answer
    itself, diagrams included, so callers should not render it again.

    Returns (answer, touched_fens): the (fen, label) pairs tools reported, in
    order, for chat_history to replay later. label is None except for
    show_opening_line diagrams.

    ask_fn produces the answer through the same callbacks as ask_agent (the
    default, looked up at call time); a saved example answer passes its
    replay here.

    The Stop button needs no handler: any click cancels the running script.
    """
    started_at = time.monotonic()
    progress_area = st.empty()
    progress_area.html(running_bar_html())
    # Status lines, then the live preview of the turn being streamed. A turn
    # is only known to be a status line or the answer once it ends, so the
    # preview sits where a status line would go and on_step keeps it there.
    work_area = st.empty()
    with work_area.container():
        lines_area = st.empty()
        preview_area = st.empty()
    step_lines: list[str] = []
    lines_visible = True

    def show_lines() -> None:
        with lines_area.container():
            for line in step_lines:
                st.markdown(line)

    # Created after the work area so it stays below the streaming text, which
    # the message panel autoscrolls to follow.
    stop_placeholder = st.empty()
    stop_placeholder.button("Stop generating", key="stop_generating")
    final_area = st.empty()

    accumulated_text = ""
    words_this_turn = 0
    touched_fens: list[tuple[str, str | None]] = []

    def on_chunk(delta: str) -> None:
        nonlocal accumulated_text, words_this_turn, lines_visible
        pieces = _WORD_SPLIT_RE.findall(delta) or [delta]
        for piece in pieces:
            accumulated_text += piece
            # Once the turn is clearly the answer, the status lines make way.
            if lines_visible and _looks_like_answer(accumulated_text):
                lines_area.empty()
                lines_visible = False
            # Markers are stripped from the preview too, so raw syntax never
            # flashes up while the closing brackets stream in.
            preview_area.markdown(_DIAGRAM_MARKER_RE.sub("", accumulated_text))
            words_this_turn += 1
            if words_this_turn <= MAX_PACED_WORDS_PER_TURN:
                time.sleep(STREAM_WORD_DELAY_SECONDS)

    def on_step(text: str) -> None:
        nonlocal accumulated_text, words_this_turn, lines_visible
        step_lines.append(text)
        # Also brings the lines back if a long status line was briefly taken
        # for the start of the answer.
        show_lines()
        lines_visible = True
        accumulated_text = ""
        words_this_turn = 0
        preview_area.empty()

    def on_position(fen: str, *, label: str | None = None) -> None:
        # Recorded for the answer's diagrams only. The board is the visitor's
        # own game, and replacing it would erase its move history.
        if not touched_fens or touched_fens[-1][0] != fen:
            touched_fens.append((fen, label))

    if ask_fn is None:
        ask_fn = ask_agent
    try:
        answer = ask_fn(
            question, on_step=on_step, on_chunk=on_chunk, on_position=on_position, history=history
        )
    except anthropic.APIError:
        # Rate limits, timeouts and 5xx errors from the SDK's own requests.
        # tool_runner only catches exceptions raised inside tools.
        logger.exception("ask_agent failed")
        answer = ""

    if not answer.strip():
        # Covers an API failure and a loop that ended without any text.
        answer = (
            "Something interrupted this response before it finished. "
            "Try asking again, possibly with a narrower question."
        )
        final_area.markdown(answer)
    else:
        # final_area, not preview_area: the work area collapses into the
        # "Done" box below and would hide the answer.
        with final_area.container():
            _render_answer_content(answer, touched_fens)

    # The bar runs to full from wherever it is, then the status lines
    # collapse into a "Done" box. The answer is already on screen.
    progress_area.html(finishing_bar_html(time.monotonic() - started_at))
    time.sleep(FINISH_SECONDS)
    progress_area.empty()
    with work_area.container(), st.status("Done", state="complete", expanded=False):
        for line in step_lines:
            st.markdown(line)
    stop_placeholder.empty()
    return answer, touched_fens


# Enough of the answer to carry its topic, which an answer names early.
ANSWER_CONTEXT_CHARS = 1200


def _lookup_context(history: list) -> str:
    """The search context for the resource lookup: the latest question as
    the model saw it (with its board position) plus the start of the answer.

    A question asked from the board ("how should White play here?") names
    nothing searchable on its own; the answer is where the opening gets
    named ("this is the Dutch Defense, ECO A80").
    """
    _, question, fen_context, _ = history[-2]
    lookup = _to_model_text(question, fen_context)
    answer = history[-1][1]
    if answer:
        lookup += f"\n\nThe answer given was:\n{answer[:ANSWER_CONTEXT_CHARS]}"
    return lookup


@dataclass
class _ResourceLookup:
    """A resource lookup for the latest answer, running in the background.
    `revealed` records whether its results have been shown."""

    future: Future[list[Recommendation]]
    revealed: bool = False


def _start_resource_lookup() -> None:
    """Start the lookup for the exchange just answered on a background
    thread, so reading the answer or using the board can't cancel it.
    Replaces any lookup still running for an earlier answer.

    The shared clients are resolved here, on the script thread; the worker
    only makes network calls.
    """
    question = _lookup_context(st.session_state.chat_history)
    db_pool, voyage = get_db_pool(), get_voyage()
    client, http_client, pacer = (
        get_anthropic_client(),
        get_lichess_http_client(),
        get_lichess_pacer(),
    )

    def lookup() -> list[Recommendation]:
        try:
            return recommend_resources(
                question, db_pool, voyage, client=client, http_client=http_client, pacer=pacer
            )
        except Exception:
            # Nothing waits on this thread, so log here; re-raise so the
            # button can show the lookup failed.
            logger.exception("recommend_resources failed")
            raise

    st.session_state.resource_lookup = _ResourceLookup(get_lookup_executor().submit(lookup))


@st.fragment(run_every=2)
def _render_pending_resources_button() -> None:
    """The disabled button shown while a lookup runs. Checks every two
    seconds without rerunning the page, then reruns it once when the lookup
    finishes."""
    lookup = st.session_state.resource_lookup
    if lookup is not None and lookup.future.done():
        st.rerun()
    st.button("Finding related resources...", key="find_resources", disabled=True)


def _render_resource_recommendations(*, disabled: bool = False) -> None:
    """The "Find related resources" button for the latest answer and the
    studies and games it reveals. Disabled until the background lookup
    finishes; then a click shows the results at once.

    The button is always rendered under the same key, disabled when there is
    nothing to show, so the tour can always find it.
    """
    lookup: _ResourceLookup | None = st.session_state.resource_lookup
    if lookup is None:
        st.button("Find related resources", key="find_resources", disabled=True)
        if not st.session_state.chat_history:
            st.caption("Ask something first, then look here for related studies and games.")
        return
    if not lookup.future.done():
        _render_pending_resources_button()
        return
    if lookup.future.exception() is not None:
        st.button("Couldn't load related resources", key="find_resources", disabled=True)
        return
    recommendations = lookup.future.result()
    if not recommendations:
        st.button("No related resources found", key="find_resources", disabled=True)
        return
    if not lookup.revealed:
        if not st.button("Show related resources", key="find_resources", disabled=disabled):
            return
        lookup.revealed = True

    for idx, rec in enumerate(recommendations):
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
            st.caption("From the local corpus, moves only, with no commentary.")
            st.code(rec.pgn, language=None)
            # idx in the key: the same game could be recommended twice.
            if st.button("Play through this game", key=f"play_{idx}_{rec.game_id}"):
                game_path = _game_path_from_pgn(rec.pgn)
                if game_path is None:
                    st.error("Couldn't parse this game's moves.")
                else:
                    st.session_state.game_path = game_path
                    st.session_state.game_path_index = 0
                    st.session_state.game_path_label = f"{rec.white} vs {rec.black} ({rec.event})"
                    st.session_state.board_generation += 1
                    st.rerun()


def _game_path_from_pgn(pgn: str) -> list[str] | None:
    """The FEN after every ply of a PGN's main line, starting position
    first: what the board's Previous/Next buttons step through when a
    recommended game is played back. None if the PGN doesn't parse."""
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None
    board = chess.Board()
    path = [board.fen()]
    for move in game.mainline_moves():
        board.push(move)
        path.append(board.fen())
    return path


def _describe_uploaded_game(
    uploaded_file, user_text: str
) -> tuple[str | None, list[tuple[str, str]]]:
    """Turn a PGN attached to the chat input into the question sent to the
    model (for Layer 4's find_similar_corpus_games): the game's moves, plus
    whatever the user typed, or a default question if they typed nothing.

    Returns (question, notices). question is None when the file holds no
    usable game. notices are (level, text) pairs for the caller to show on
    the run that displays the answer, since a successful upload reruns the
    page straight away.
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pgn", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(uploaded_file.getvalue())
        # Uploads are untrusted: read at most two games (one to use, one to
        # know there were more), and cut a game off at a repetition so a
        # crafted file can't produce an arbitrarily long prompt.
        first_two_games = list(
            itertools.islice(
                parse_pgn(tmp_path, source="user_upload", truncate_at_repetition=True), 2
            )
        )
    except ValueError:
        # Raised for a header containing the game-ID delimiter
        # (hash_utils.check_no_delimiter).
        return None, [("error", "Couldn't find a game in that file.")]
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    # python-chess parses any text, prose included, into a game with zero
    # moves, so an empty move list means "not a PGN".
    if not first_two_games or not first_two_games[0].moves:
        return None, [("error", "Couldn't find a game in that file.")]

    notices: list[tuple[str, str]] = []
    if len(first_two_games) > 1:
        notices.append(("info", "This file has more than one game. Only the first is analyzed."))

    game = first_two_games[0]
    move_sans = [m.move_san for m in game.moves]
    # A movetext error stops the parse there, so only part of the game was
    # read; say so rather than silently analyzing a fragment.
    if game.parse_errors:
        notices.append(
            (
                "warning",
                f"Some moves in that file couldn't be read, so only the first "
                f"{len(move_sans)} are analyzed.",
            )
        )
    # Only the opening goes into the prompt: the similarity search compares
    # at most MAX_PLY_WINDOW plies, and a full game would only add cost.
    if len(move_sans) > MAX_PLY_WINDOW:
        game_description = (
            f"Here is the opening of a game I uploaded, as a list of its first "
            f"{MAX_PLY_WINDOW} moves in order: {', '.join(move_sans[:MAX_PLY_WINDOW])}."
        )
    else:
        game_description = (
            f"Here is a game I uploaded, as a list of moves in order: {', '.join(move_sans)}."
        )
    if user_text.strip():
        return f"{game_description} {user_text}", notices
    # A bare upload gives the model no sense of how much to write, so the
    # default question asks for a short answer.
    return (
        f"{game_description} Find similar games in the corpus and give me a brief, "
        "illustrative comparison: a few sentences, not a full report."
    ), notices


# A streamed turn is only known to be a status line or the answer once it
# ends. Status lines are one sentence (the longest seen is ~215 characters)
# with no line break, so longer text, or text with a line break (answers
# often open with a heading), is treated as the answer starting.
ANSWER_DETECTION_CHARS = 300


def _looks_like_answer(streamed_text: str) -> bool:
    return len(streamed_text) > ANSWER_DETECTION_CHARS or "\n" in streamed_text.lstrip()


DIAGRAM_SIZE_PX = 180


def render_position_thumbnail(fen: str, label: str | None = None) -> None:
    """A small board diagram for a position a tool reported. Assumes a
    valid FEN."""
    if label:
        st.caption(label)
    svg = chess.svg.board(board=chess.Board(fen), size=DIAGRAM_SIZE_PX)
    # margin:0 removes the body's default margin, and line-height:0 removes
    # the gap browsers leave under an inline svg, so the frame is exactly
    # DIAGRAM_SIZE_PX tall.
    st.iframe(f"<body style='margin:0;line-height:0'>{svg}</body>", height=DIAGRAM_SIZE_PX)


def _render_answer_content(text: str, image_fens: list[tuple[str, str | None]]) -> None:
    """Render an answer, putting each labeled diagram at its
    [[diagram: <label>]] marker.

    The system prompt asks the model to write that marker, with the label it
    gave show_opening_line, where the diagram belongs; searching the prose
    for the label instead rarely matches. Markers are always stripped, even
    when no diagram matches. Diagrams without a marker (engine evaluations,
    or a marker the model left out) follow the text in order. At most
    MAX_INLINE_DIAGRAMS are shown.
    """
    by_label: dict[str, list[str]] = {}
    for fen, label in image_fens:
        if label:
            by_label.setdefault(label.lower(), []).append(fen)

    placed_fens: set[str] = set()
    shown = 0
    cursor = 0
    for match in _DIAGRAM_MARKER_RE.finditer(text):
        st.markdown(text[cursor : match.start()])
        cursor = match.end()
        if shown >= MAX_INLINE_DIAGRAMS:
            continue
        marker_label = match.group(1).strip()
        candidates = by_label.get(marker_label.lower())
        if not candidates:
            continue
        fen = candidates.pop(0)
        render_position_thumbnail(fen, marker_label)
        placed_fens.add(fen)
        shown += 1

    st.markdown(text[cursor:])

    for fen, label in image_fens:
        if shown >= MAX_INLINE_DIAGRAMS:
            break
        if fen in placed_fens:
            continue
        render_position_thumbnail(fen, label)
        shown += 1


def _to_model_text(content: str, fen_context: str | None) -> str:
    """The text sent to the model for one user turn: the question, prefixed
    with "Current board position: <FEN>" when it was asked about the board.
    Used for both the current turn and history, so the two always match."""
    if fen_context is not None:
        return f"Current board position: {fen_context}\n\n{content}"
    return content


# Five exchanges of context for follow-up questions; replaying the whole
# session would grow every question's cost with its length.
MAX_HISTORY_MESSAGES = 10

# Every question costs Anthropic (and often Voyage) calls on a public,
# unauthenticated app. This per-session limit stops runaway clicking or a
# simple script without throttling other visitors; it can't stop someone
# opening new sessions, which is what the API spending caps are for.
MAX_REQUESTS_PER_MINUTE = 8
REQUEST_WINDOW_SECONDS = 60


def _build_message_history() -> list[dict[str, str]]:
    """Recent chat_history as the {"role", "content"} dicts ask() expects:
    final text only, no tool calls."""
    recent = st.session_state.chat_history[-MAX_HISTORY_MESSAGES:]
    return [
        {
            "role": role,
            "content": _to_model_text(content, fen_context) if role == "user" else content,
        }
        for role, content, fen_context, _ in recent
    ]


def _within_rate_limits() -> bool:
    """Count one question against the per-session and shared limits, or
    show a warning and return False if either is used up."""
    now = time.monotonic()
    request_times = st.session_state.setdefault("request_times", [])
    request_times[:] = [t for t in request_times if now - t < REQUEST_WINDOW_SECONDS]
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
        st.warning(
            "You're asking questions faster than this demo can keep up with. Try again in a moment."
        )
        return False
    if not get_shared_question_limit().try_acquire():
        st.warning("This demo has reached its question limit for today. Please try again tomorrow.")
        return False
    request_times.append(now)
    return True


def _replay_without_history(
    cached: CachedAnswer, question: str, *, history=None, **callbacks
) -> str:
    """ask_fn for a saved example answer: replays it, ignoring the question
    and history it is called with."""
    return replay_cached_answer(cached, **callbacks)


def _submit_question(question: str, *, fen_context: str | None = None) -> bool:
    """Ask the agent and add the exchange to the transcript. Every question
    goes through here: typed, example prompts, Evaluate and the board's own
    question box.

    Called from the chat column only, since the live answer UI renders
    wherever this runs; board-side buttons set
    st.session_state.pending_question and rerun instead.

    fen_context, if given, is sent to the model with the question and shown
    under it in the transcript.

    An example question with a saved answer (see example_answers.py) is
    answered from it, without calling the API or counting toward the limits.

    Returns False when the rate limit rejected the question. The caller must
    not rerun then, or the warning would vanish before anyone read it.
    """
    cached = cached_example_answer(question, fen_context=fen_context)
    if cached is None and not _within_rate_limits():
        return False
    # The previous answer's lookup no longer applies; its results are dropped.
    st.session_state.resource_lookup = None

    history = _build_message_history()  # earlier turns only
    st.session_state.chat_history.append(("user", question, fen_context, []))
    with st.chat_message("user", avatar=_CHAT_AVATARS["user"]):
        st.markdown(question)
        if fen_context is not None:
            st.caption(f"Position: `{fen_context}`", help=FEN_HELP)
    sent_question = _to_model_text(question, fen_context)
    with st.chat_message("assistant", avatar=_CHAT_AVATARS["assistant"]):
        if cached is not None:
            replay = functools.partial(_replay_without_history, cached)
            answer, touched_fens = ask_with_status(sent_question, history=history, ask_fn=replay)
        else:
            answer, touched_fens = ask_with_status(sent_question, history=history)
    st.session_state.chat_history.append(("assistant", answer, None, touched_fens))
    if cached is not None:
        ready: Future[list[Recommendation]] = Future()
        ready.set_result(cached.recommendations)
        st.session_state.resource_lookup = _ResourceLookup(ready)
    else:
        _start_resource_lookup()

    log_conversation_best_effort(get_db_pool(), question, fen_context, answer)
    return True


def _render_example_prompts(*, disabled: bool = False) -> None:
    """Example questions for an empty conversation, one per layer, so the
    four together show the routing. A click stashes the question as pending
    and reruns, the same path as a typed question."""
    st.caption("Try asking:")
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, key=f"example_prompt_{i}", width="stretch", disabled=disabled):
            st.session_state.pending_question = (example, None)
            st.rerun()


def render_main_screen() -> None:
    # Kept on screen rather than only in the tour or README: it's the
    # caveat about how far the answers can be trusted. Keyed so theme.css
    # can make it a quiet footnote.
    with st.container(key="reliability_note"):
        st.caption(
            "Answers synthesize retrieved human commentary and engine output, "
            "not the model's own independent chess judgment."
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "resource_lookup" not in st.session_state:
        st.session_state.resource_lookup = None
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "last_illegal_attempt" not in st.session_state:
        st.session_state.last_illegal_attempt = None
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "pending_notices" not in st.session_state:
        st.session_state.pending_notices = []
    if "board_generation" not in st.session_state:
        st.session_state.board_generation = 0
    if "game_path" not in st.session_state:
        st.session_state.game_path = None
    if "game_path_index" not in st.session_state:
        st.session_state.game_path_index = 0
    if "game_path_label" not in st.session_state:
        st.session_state.game_path_label = None

    chat_col, board_col = st.columns([3, 2])

    # A question is pending from the run that submits it until the run that
    # answers it; any click during that run would cancel the answer.
    generating = st.session_state.pending_question is not None

    # The board column renders first even though it sits on the right: the
    # answer blocks the script inside the chat column, so this way the board
    # appears disabled for the whole answer instead of staying clickable.
    with board_col, st.container(key="board_panel"):
        render_board_panel(disabled=generating)
        st.divider()
        _render_resource_recommendations(disabled=generating)

    with chat_col:
        # A fixed-height scrolling panel keeps the chat input level with the
        # board; st.chat_input only pins itself to the page bottom at the top
        # level. Streamlit autoscrolls a fixed-height container of chat
        # messages on its own.
        message_panel = st.container(height=MESSAGE_PANEL_HEIGHT_PX, border=True, key="chat_panel")
        with message_panel:
            if not st.session_state.chat_history:
                # Also drawn on the run that answers the first question, so
                # disabled then: a click would cancel the answer.
                _render_example_prompts(disabled=generating)
            for role, content, fen, image_fens in st.session_state.chat_history:
                with st.chat_message(role, avatar=_CHAT_AVATARS[role]):
                    if role == "assistant":
                        _render_answer_content(content, image_fens)
                    else:
                        st.markdown(content)
                    if fen is not None:
                        st.caption(f"Position: `{fen}`", help=FEN_HELP)

            # Any click during an answer cancels it (the Stop button works
            # that way), leaving a question with no reply. Say so instead of
            # looking like a silent failure.
            history = st.session_state.chat_history
            if history and history[-1][0] == "user" and not generating:
                st.info("That answer was interrupted before it finished. Ask again to retry.")

            pending = st.session_state.pending_question
            if pending is not None:
                for level, message in st.session_state.pending_notices:
                    getattr(st, level)(message)
                st.session_state.pending_notices = []
                st.session_state.pending_question = None
                pending_question, pending_fen = pending
                submitted = _submit_question(pending_question, fen_context=pending_fen)
                # Rerun so the example prompts, drawn above while the history
                # was still empty, disappear. Not after a rate-limit
                # rejection, which would erase its warning.
                if submitted:
                    st.rerun()

        submission = st.chat_input(
            "Ask a chess question...",
            max_chars=MAX_QUESTION_CHARS,
            accept_file=True,
            file_type=["pgn"],
        )
        # Said next to the input, where visitors will see it (see
        # conversation_log.py).
        st.caption("Questions and answers are saved to help improve this demo.")
        if submission is not None:
            question = None
            notices: list[tuple[str, str]] = []
            if not accept_question_length(submission.text):
                pass
            elif submission.files:
                question, notices = _describe_uploaded_game(submission.files[0], submission.text)
                if question is None:
                    # No rerun follows a rejected upload, so show these now.
                    for level, message in notices:
                        getattr(st, level)(message)
            elif submission.text.strip():
                question = submission.text
            if question is not None:
                # Stashed and rerun, like the board's buttons, so the board
                # is drawn disabled for the whole answer.
                st.session_state.pending_question = (question, None)
                st.session_state.pending_notices = notices
                st.rerun()
