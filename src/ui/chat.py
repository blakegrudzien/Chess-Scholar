"""The conversation half of the app's single screen: the chat transcript
(PGN attachments included, see render_main_screen's chat_input), the live
streaming/status UI for an in-flight answer, and the resource-recommendation
cards. Owns st.session_state.chat_history.

render_main_screen also composes the board column, but that column's own
rendering and state live in board_panel.py.
"""

from __future__ import annotations

import html
import io
import itertools
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable

import anthropic
import chess
import chess.pgn
import chess.svg
import streamlit as st

from src.agent.chess_agent import OnPosition, ask
from src.ingestion.pgn_parser import parse_pgn
from src.recommendation.pipeline import (
    ChessbaseGameRecommendation,
    LichessStudyRecommendation,
    recommend_resources,
)
from src.ui.board_panel import render_board_panel
from src.ui.conversation_log import log_conversation_best_effort
from src.ui.help_text import FEN_HELP
from src.ui.resources import (
    get_anthropic_client,
    get_db_pool,
    get_engine_pool,
    get_lichess_http_client,
    get_lichess_pacer,
    get_voyage,
)

logger = logging.getLogger(__name__)

# Anthropic's raw text deltas arrive in relatively large pieces (a clause or
# sentence at a time), not the smooth word-by-word reveal seen in Claude.ai
# or ChatGPT -- that reveal is a client-side pacing effect, not a property
# of the network chunks. This regex re-splits each delta into word-sized
# pieces (including leading and trailing whitespace, so pieces concatenate
# back to the exact original text) so it can be paced the same way.
_WORD_SPLIT_RE = re.compile(r"\s*\S+\s*")
STREAM_WORD_DELAY_SECONDS = 0.02

# Matches the literal [[diagram: <label>]] marker SYSTEM_PROMPT instructs
# the model to place inline in its own answer -- see _render_answer_content.
_DIAGRAM_MARKER_RE = re.compile(r"\[\[diagram:\s*(.*?)\s*\]\]")

# Caps the paced reveal to the first N words of any single turn's text.
# Without a ceiling, a long answer -- 900+ words is realistic -- pays
# STREAM_WORD_DELAY_SECONDS on every word, adding 15-20 seconds of purely
# artificial delay on top of real generation and tool-call latency. The
# first N words still get the typewriter feel; the rest appears at once.
MAX_PACED_WORDS_PER_TURN = 40

# Cap on how many inline diagrams one answer shows -- "main line + a couple
# of popular sidelines" is usually 2-4 diagrams; a wall of positions on one
# answer works against scannability.
MAX_INLINE_DIAGRAMS = 4

# Height of the scrollable message panel (see render_main_screen). Sized so
# a fresh, question-less page fits inside a laptop viewport with no vertical
# scroll; a 14" MacBook's default logical resolution is the binding case.
# The board column's natural height is the real floor here, so this is set
# to sit at that height rather than needlessly exceed it.
MESSAGE_PANEL_HEIGHT_PX = 560

# Streamlit's default chat avatars are a generic face/robot Material icon --
# a visual cue that reads as "generic AI chatbot," working against the
# deliberately non-modern, non-AI-flavored identity built for this app.
# Chess pieces are already this app's own icon language (the board renders
# pieces via chess.svg), not a new decoration introduced just for the
# avatars.
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
# renderer used for the board itself, instead of depending on emoji font
# coverage across viewers' systems.
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
    question: str, *, history: list[dict[str, str]] | None = None
) -> tuple[str, list[tuple[str, str | None]]]:
    """Run ask_agent, showing each tool-calling step live in an st.status
    panel and streaming the final answer into view as it generates, rather
    than a blank spinner followed by the whole answer appearing at once.
    Renders everything itself, diagrams included (see _render_answer_content)
    -- callers should not render `answer` again afterward.

    history, if given, is prior turns for the model's context -- see
    _build_message_history.

    Returns (answer, touched_fens): every distinct (fen, label) pair
    on_position reported during this call, in order (immediate repeats
    deduped). label is None except for show_opening_line's calls. Returned
    only so a caller can store it in chat_history for history replay --
    already rendered here, not meant to be rendered again.

    The stop button needs no explicit click handling: Streamlit treats
    interactions as implicit yield points during a running script, so a
    click interrupts this at the next word reveal.
    """
    status = st.status("Thinking...", expanded=True)
    with status:
        # Nested inside the status box, not a sibling of it: a tool-calling
        # turn's rationale and the eventual final answer both start out as
        # plain streamed text (nothing distinguishes them until a tool_use
        # block does or doesn't show up at the end of the turn), so on_step
        # below only has to promote this preview to a permanent
        # status.write() line. As a top-level placeholder instead, a
        # turn's text visibly jumped into the box the moment a tool call
        # was confirmed.
        preview_area = st.empty()
    # Declared after the status box: message_panel autoscrolls to follow
    # streamed content, and a placeholder's DOM position is fixed at
    # creation time, so declaring this first would pin it above the
    # streamed text and scroll it out of view on a long answer.
    stop_placeholder = st.empty()
    stop_placeholder.button("Stop generating", key="stop_generating")
    final_area = st.empty()

    accumulated_text = ""
    words_this_turn = 0
    touched_fens: list[tuple[str, str | None]] = []

    def on_chunk(delta: str) -> None:
        nonlocal accumulated_text, words_this_turn
        pieces = _WORD_SPLIT_RE.findall(delta) or [delta]
        for piece in pieces:
            accumulated_text += piece
            # Strip diagram markers before the live preview, not just at
            # final render -- otherwise raw "[[diagram: ...]]" syntax
            # flashes on screen for the word or two it takes the closing
            # bracket to stream in.
            preview_area.markdown(_DIAGRAM_MARKER_RE.sub("", accumulated_text))
            words_this_turn += 1
            if words_this_turn <= MAX_PACED_WORDS_PER_TURN:
                time.sleep(STREAM_WORD_DELAY_SECONDS)

    def on_step(text: str) -> None:
        nonlocal accumulated_text, words_this_turn
        status.write(text)
        accumulated_text = ""
        words_this_turn = 0
        preview_area.empty()

    def on_position(fen: str, *, label: str | None = None, update_board: bool = True) -> None:
        # `and game_path is None`: during replay, st.session_state.board is
        # the free-play board sitting *behind* the visible replay, invisible
        # but still live -- without this guard it would get silently
        # overwritten by any position-touching tool call that turn.
        if update_board and st.session_state.game_path is None:
            try:
                new_board = chess.Board(fen)
            except ValueError:
                return  # a malformed FEN from the model -- leave the board as it was
            st.session_state.board = new_board
            # A fresh chess.Board(fen) has no move history, so an illegal-move
            # warning from before this update no longer corresponds to
            # anything real on the new board.
            st.session_state.last_illegal_attempt = None
            st.session_state.board_generation += 1
        # show_opening_line already guarantees a valid fen (replayed via
        # python-chess before this is ever called), so no re-validation
        # needed on the update_board=False path.
        if not touched_fens or touched_fens[-1][0] != fen:
            touched_fens.append((fen, label))

    try:
        answer = ask_agent(
            question, on_step=on_step, on_chunk=on_chunk, on_position=on_position, history=history
        )
    except anthropic.APIError:
        # Covers rate limits, timeouts, dropped connections, and transient
        # 5xx/overloaded errors -- tool_runner only catches exceptions a
        # *tool* raises (see build_tools' own docstring), so a failure in
        # the SDK's own calls to Anthropic would otherwise crash the whole
        # Streamlit script mid-answer.
        logger.exception("ask_agent failed")
        answer = ""

    if not answer.strip():
        # The tool-calling loop can end on a turn whose only content was a
        # tool call (rare -- see chess_agent.ask()'s docstring), which would
        # otherwise render an empty chat bubble with no indication anything
        # went wrong. Also doubles as the API-error fallback above.
        answer = (
            "Something interrupted this response before it finished -- "
            "try asking again, possibly with a narrower question."
        )
        final_area.markdown(answer)
    else:
        # The completed answer goes to final_area, not preview_area --
        # preview_area is nested inside the status box (see this function's
        # own docstring), which is about to collapse to "Done" a few lines
        # down, so leaving the final answer there would hide it inside a
        # collapsed expander the user has to click open. final_area was
        # declared as a sibling of the status box specifically so the
        # answer stays visible once the box collapses.
        with final_area.container():
            _render_answer_content(answer, touched_fens)

    status.update(label="Done", state="complete", expanded=False)
    preview_area.empty()
    stop_placeholder.empty()
    return answer, touched_fens


def _render_resource_recommendations() -> None:
    """Offers to look up related Lichess studies and corpus games for the
    most recent question, and renders whatever comes back. Only shown for
    the latest exchange, not every past one in the history.

    The button always renders, even on a fresh page -- disabled rather than
    absent, both as a visible preview of the feature and so it's a stable
    tutorial_overlay spotlight target (a conditionally-existing element
    can't be reliably spotlighted from a fresh page).

    st.session_state.resource_recommendations is the sentinel for "already
    looked up this question": None means not yet requested, a list (possibly
    empty) means it has been. Reset to None in _submit_question after a new
    answer, so a fresh question always gets a fresh button.
    """
    history = st.session_state.chat_history
    eligible = len(history) >= 2 and history[-1][0] == "assistant"

    if st.session_state.resource_recommendations is None:
        # secondary, not primary: "Evaluate this position with Stockfish" is
        # the page's one primary action -- two competing primary-styled
        # buttons would dilute that visual hierarchy.
        clicked = st.button(
            "Find related resources",
            key="find_resources",
            type="secondary",
            disabled=not eligible,
        )
        if not eligible:
            st.caption("Ask something first, then look here for related studies and games.")
            return
        if not clicked:
            return
        question = history[-2][1]
        # Doherty threshold: this regularly takes 20+ seconds, past the
        # point where a bare spinner keeps people's attention. A static but
        # honest description of the stages involved, not live progress --
        # unlike ask_with_status, this isn't wired to recommend_resources'
        # actual tool calls via on_step.
        with st.status("Looking for related resources...", expanded=True) as status:
            status.write("Searching the study library for relevant chapters.")
            status.write("Checking whether a matching master game exists in the corpus.")
            try:
                st.session_state.resource_recommendations = recommend_resources(
                    question,
                    get_db_pool(),
                    get_voyage(),
                    client=get_anthropic_client(),
                    http_client=get_lichess_http_client(),
                    pacer=get_lichess_pacer(),
                )
            except anthropic.APIError:
                # This call chains Anthropic + Voyage + live Lichess HTTP
                # behind one click; caught here (inside the `with`, so
                # `status` is still live to update) rather than around it,
                # so resource_recommendations stays None and the button
                # reappears on the next rerun.
                logger.exception("recommend_resources failed")
                status.update(
                    label="Something went wrong looking that up. Try again in a moment.",
                    state="error",
                )
                return
            status.update(label="Done", state="complete", expanded=False)

    recommendations = st.session_state.resource_recommendations
    if not recommendations:
        st.caption("Nothing in the study library or corpus was a close enough match to recommend.")
        return

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
            st.caption("From the local corpus, moves only -- no commentary included.")
            st.code(rec.pgn, language=None)
            # key qualified with idx, not just rec.game_id -- recommend_chessbase_game
            # (pipeline.py) has no dedup, so the model could in principle recommend
            # the same game twice in one turn, which would collide on game_id alone.
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
    """Every ply's FEN in a PGN's mainline, starting position included at
    index 0 -- the sequence the board panel's Prev/Next steps through
    for a "Play through this game" recommendation.

    Returns None if the PGN doesn't parse. Defensive, not expected in
    practice: structured_search.game_moves_as_pgn (the only source of
    rec.pgn today) reconstructs PGN via python-chess's own serializer from
    moves already validated with board.parse_san while building it, so it
    should always be well-formed -- but nothing forces that guarantee to
    hold for every future caller of this helper, and a malformed PGN
    shouldn't crash the page.
    """
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None
    board = chess.Board()
    path = [board.fen()]
    for move in game.mainline_moves():
        board.push(move)
        path.append(board.fen())
    return path


def _describe_uploaded_game(uploaded_file, user_text: str) -> str | None:
    """Turn a PGN attached to the chat input (see render_main_screen's
    accept_file=True) into the model-facing question text for Layer 4's
    find_similar_corpus_games: a plain-language description of the game's
    moves, combined with whatever the user typed alongside it, or a
    sensible default question if they attached the file with no message
    of its own.

    Returns None (after showing st.error itself, since this always runs
    right before a rerun -- there's no later point a caller could still
    surface the message) if the file has no parseable game.
    """
    # Initialized before the write, not inside it: if the write itself
    # fails, the finally below still has a defined name to check instead of
    # raising NameError and masking the real exception.
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pgn", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(uploaded_file.getvalue())
        # Only the first game is ever used below, so pull at most two from
        # the generator: the first to use, and a second only to learn
        # whether there's more than one, without fully parsing an upload
        # (untrusted input) that could contain a large number of games.
        # truncate_at_repetition bounds a single game's move count too --
        # otherwise an adversarial upload (e.g. two pieces shuffled back
        # and forth) could inflate move_sans below into an arbitrarily
        # long, expensive prompt with no cap at all.
        first_two_games = list(
            itertools.islice(
                parse_pgn(tmp_path, source="user_upload", truncate_at_repetition=True), 2
            )
        )
    except ValueError:
        # compute_game_id (parse_pgn -> parse_game -> compute_game_id)
        # deliberately raises ValueError if a header contains ID_DELIMITER
        # (see hash_utils.check_no_delimiter) -- rare, but this is
        # untrusted user input, and it's exactly the kind of thing someone
        # poking at the upload feature might hit. The same honest message
        # used below for a genuinely unparseable file already covers this
        # case too; no need for a second, more specific one.
        st.error("Couldn't find a game in that file.")
        return None
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    if not first_two_games:
        st.error("Couldn't find a game in that file.")
        return None
    if len(first_two_games) > 1:
        st.info("This file has more than one game. Only the first is analyzed.")

    move_sans = [m.move_san for m in first_two_games[0].moves]
    game_description = (
        f"Here is a game I uploaded, as a list of moves in order: {', '.join(move_sans)}."
    )
    if user_text.strip():
        return f"{game_description} {user_text}"
    # No message of their own to go on, so this default has to name a
    # length -- unlike a typed question, which already implies "answer
    # this much and stop", a bare upload gives the model nothing to
    # calibrate against, and the general efficiency nudge in SYSTEM_PROMPT
    # (about tool-call count, not prose length) doesn't cover that.
    return (
        f"{game_description} Find similar games in the corpus and give me a brief, "
        "illustrative comparison -- a few sentences, not a full report."
    )


DIAGRAM_SIZE_PX = 180


def render_position_thumbnail(fen: str, label: str | None = None) -> None:
    """A small, plain board diagram for a position a tool call actually
    verified (see ask_with_status's touched_fens) -- no highlights or
    arrows yet, those need annotation data this app's ingestion pipeline
    currently discards (see CLAUDE.md's Board input note). Malformed FENs
    are the caller's problem to guard against; this assumes a valid one.
    """
    if label:
        st.caption(label)
    svg = chess.svg.board(board=chess.Board(fen), size=DIAGRAM_SIZE_PX)
    # chess.svg.board's own <svg> tag is exactly DIAGRAM_SIZE_PX square
    # (its width/height attributes say so), but two things about the
    # surrounding HTML document st.iframe builds around it add invisible
    # height on top of that: the browser's own default ~8px <body> margin
    # (killed by margin:0 below), and -- less obvious, and the part that
    # was still overflowing with only the margin fix -- an <svg> is an
    # inline element by default, so the browser reserves a few pixels of
    # line-height beneath it for text descenders, the same "mystery gap"
    # that shows up under a bare <img> in a div. line-height:0 on body
    # collapses that reserved space to nothing, so the real content height
    # matches DIAGRAM_SIZE_PX exactly instead of a few px more.
    st.iframe(f"<body style='margin:0;line-height:0'>{svg}</body>", height=DIAGRAM_SIZE_PX)


def _render_answer_content(text: str, image_fens: list[tuple[str, str | None]]) -> None:
    """Renders `text` as markdown, replacing each [[diagram: <label>]]
    marker (see SYSTEM_PROMPT's show_opening_line instruction) with the
    matching labeled diagram, instead of every diagram dumped after the
    full answer regardless of what it's actually discussing.

    The marker is a literal token the model was explicitly told to place
    at the point in its own answer where a diagram belongs, using the same
    label it already passed to show_opening_line. Searching the answer's
    free-form prose for that label instead almost never matches -- nothing
    obliges the model to repeat a tool argument verbatim in its synthesis
    -- so diagrams would land at the end regardless. A marker the model is
    told to write is a real contract; a string search against text it was
    never told to shape around that string is not.

    Every marker match is stripped from the visible text whether or not it
    resolves to a diagram -- a label with nothing left to match (unknown
    label, or a repeated label with no diagrams left) just disappears
    rather than leaking raw "[[diagram: ...]]" syntax into the chat.
    Diagrams with no matching marker at all (every unlabeled one --
    evaluate_chess_position, find_similar_corpus_games, see
    ask_with_status's docstring -- plus any labeled one the model forgot to
    place a marker for) render after the full text, in original order.
    Capped at MAX_INLINE_DIAGRAMS total.
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
    """The exact text sent to the model for one user turn -- content as
    typed, with the "Current board position" prefix prepended when the
    turn was about a specific position (see chess_agent.SYSTEM_PROMPT).
    Used both for the current turn and to reconstruct past ones for
    history, so the two can never drift apart.
    """
    if fen_context is not None:
        return f"Current board position: {fen_context}\n\n{content}"
    return content


# Bounds how many prior chat_history entries get replayed as context on each
# call -- unbounded history would grow every question's token cost (and
# latency/cost) linearly with the whole session's length. 10 entries is 5
# user/assistant exchanges, generous for a follow-up-question demo without
# letting a long session's cost run away.
MAX_HISTORY_MESSAGES = 10

# This app has no auth, and every question bills at least one Anthropic call
# (often several, across tool-calling turns) plus a Voyage embedding call
# whenever search_annotations fires -- with a public deployment URL, nothing
# else stands between a visitor and this app's own API budget. Session-local,
# not a deployment-wide counter: it doesn't stop a determined attacker
# spinning up fresh sessions (st.session_state resets per session), but it
# does stop the far likelier case -- a stuck retry, someone mashing an
# example-prompt button, or a casual script -- without penalizing every other
# concurrent visitor the way a shared counter would. Pair with a hard
# spending cap in the Anthropic/Voyage dashboards for the backstop this can't
# provide on its own.
MAX_REQUESTS_PER_MINUTE = 8
REQUEST_WINDOW_SECONDS = 60


def _build_message_history() -> list[dict[str, str]]:
    """The most recent chat_history entries, translated into the plain
    {"role", "content"} dicts ask() expects -- see ask()'s own docstring
    for why this replays only final text, not full tool-call traces.
    """
    recent = st.session_state.chat_history[-MAX_HISTORY_MESSAGES:]
    return [
        {
            "role": role,
            "content": _to_model_text(content, fen_context) if role == "user" else content,
        }
        for role, content, fen_context, _ in recent
    ]


def _submit_question(question: str, *, fen_context: str | None = None) -> None:
    """Append a user turn, get the agent's answer, and append it -- the one
    path both the main chat input and the board-side "ask about this
    position" box go through, so a follow-up question either way lands in
    the same transcript instead of two disconnected conversations.

    Only ever called from inside chat_col (see render_main_screen and its
    "pending_question" handling below) -- never directly from a
    board-side button. ask_with_status renders live UI (a status panel, a
    streaming answer area) at whatever point in the layout it's called from,
    so calling it directly from a button inside the narrow board column
    would put that live UI in the board column instead of the chat
    transcript. Board-side triggers stash (question, fen) in
    st.session_state.pending_question and rerun instead, so the
    actual submission always happens from chat_col on the next script pass.

    fen_context, when given, is sent to the model as part of the question
    (see chess_agent.SYSTEM_PROMPT's "Current board position" instruction)
    and stored alongside the question in chat_history so the transcript can
    show which position a question was about, without the FEN prefix itself
    ever appearing as if the user had typed it.

    Rate-limited first, before anything else: see MAX_REQUESTS_PER_MINUTE's
    own comment. Checked (and the attempt recorded) ahead of every other
    branch below so a throttled request never reaches ask_with_status and
    never touches chat_history -- a rejected question shouldn't appear in
    the transcript as if it had been asked and silently ignored.
    """
    now = time.monotonic()
    request_times = st.session_state.setdefault("request_times", [])
    request_times[:] = [t for t in request_times if now - t < REQUEST_WINDOW_SECONDS]
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
        st.warning(
            "You're asking questions faster than this demo can keep up with -- "
            "try again in a moment."
        )
        return
    request_times.append(now)

    history = _build_message_history()  # prior turns, before this one is appended below
    st.session_state.chat_history.append(("user", question, fen_context, []))
    with st.chat_message("user", avatar=_CHAT_AVATARS["user"]):
        st.markdown(question)
        if fen_context is not None:
            st.caption(f"Position: `{fen_context}`", help=FEN_HELP)
    sent_question = _to_model_text(question, fen_context)
    with st.chat_message("assistant", avatar=_CHAT_AVATARS["assistant"]):
        answer, touched_fens = ask_with_status(sent_question, history=history)
    st.session_state.chat_history.append(("assistant", answer, None, touched_fens))
    st.session_state.resource_recommendations = None

    # See conversation_log.log_conversation_best_effort's own docstring:
    # it guarantees on its own that a logging failure can't reach here.
    log_conversation_best_effort(get_db_pool(), question, fen_context, answer)


def _render_example_prompts() -> None:
    """Shown only on an empty conversation (render_main_screen checks
    chat_history before calling this) -- teaches the chat's actual range by
    demonstration, one example per layer/feature. Clicking one submits it,
    the same as typing it and pressing enter would.

    Stashes into st.session_state.pending_question and reruns rather than
    calling _submit_question directly: this runs *before* render_main_
    screen's chat_history loop in the same script pass, so submitting
    inline here would render the new pair once from this call and again
    from that loop right after -- every message doubled. Rerunning lets a
    fresh pass see chat_history as already non-empty, so this function is
    skipped and the loop renders the new pair exactly once.
    """
    st.caption("Try asking:")
    # One example per layer, so the four together demonstrate the routing
    # rather than four variations on the same backend.
    examples = [
        # Layers 1 + 2 together: corpus statistics synthesized with strategic
        # prose. The flagship query for this project (see CLAUDE.md).
        "How should White meet the Sicilian Defense?",
        "Evaluate 1. e4 e5 2. Qh5 for White",  # Layer 3, engine grounding
        "What's the plan behind an isolated queen pawn?",  # Layer 2, conceptual
        # Layer 1 alone: a piece-placement aggregation with no conceptual
        # half, so one example exercises the structured-search path on its own.
        "Where does White's knight usually end up in the Najdorf?",
    ]
    for i, example in enumerate(examples):
        if st.button(example, key=f"example_prompt_{i}", width="stretch"):
            st.session_state.pending_question = (example, None)
            st.rerun()


def render_main_screen() -> None:
    # The walkthrough content that used to live here (what the chat/board/
    # find-resources/PGN-upload each do) is now covered by app.py's own
    # render_tutorial_trigger() (src/ui/tutorial_overlay) spotlight tour --
    # a second, redundant explanation of the same UI right below it was the
    # user's own call to cut once both had been tried side by side.
    #
    # This one line survives on its own, not folded into the tour: a
    # standing caption, not something that only appears if a viewer happens
    # to click "How this works" -- CLAUDE.md requires this caveat stay
    # visible in the UI, not just documented, and the tour's steps are
    # about what each control does, not about the answers' reliability.
    #
    # Keyed wrapper so styles.py can size this one caption down without
    # touching every other st.caption on the page -- present, but a quiet
    # footnote under the header rather than a second headline competing
    # with the title above it.
    with st.container(key="reliability_note"):
        st.caption(
            "Answers synthesize retrieved human commentary and engine output, "
            "not the model's own independent chess judgment."
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "resource_recommendations" not in st.session_state:
        st.session_state.resource_recommendations = None
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "last_illegal_attempt" not in st.session_state:
        st.session_state.last_illegal_attempt = None
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "board_generation" not in st.session_state:
        st.session_state.board_generation = 0
    if "game_path" not in st.session_state:
        st.session_state.game_path = None
    if "game_path_index" not in st.session_state:
        st.session_state.game_path_index = 0
    if "game_path_label" not in st.session_state:
        st.session_state.game_path_label = None

    chat_col, board_col = st.columns([3, 2])

    with chat_col:
        # A fixed-height, internally scrolling panel, not chat_input's own
        # "pin to the bottom of the page" trick -- that trick only works
        # when chat_input sits at the top level of the app; nested in this
        # column (alongside board_col), it just renders inline after the
        # last message instead, so its position on the page drifted with
        # the conversation's length -- sometimes level with the board,
        # sometimes well below it. autoscroll left at its default (None):
        # Streamlit auto-enables it for a fixed-height container holding
        # st.chat_message elements, exactly this case, so a new message
        # scrolls itself into view without extra plumbing here.
        message_panel = st.container(height=MESSAGE_PANEL_HEIGHT_PX, border=True, key="chat_panel")
        with message_panel:
            if not st.session_state.chat_history:
                _render_example_prompts()
            for role, content, fen, image_fens in st.session_state.chat_history:
                with st.chat_message(role, avatar=_CHAT_AVATARS[role]):
                    if role == "assistant":
                        _render_answer_content(content, image_fens)
                    else:
                        st.markdown(content)
                    if fen is not None:
                        st.caption(f"Position: `{fen}`", help=FEN_HELP)

            pending = st.session_state.pending_question
            if pending is not None:
                st.session_state.pending_question = None
                pending_question, pending_fen = pending
                _submit_question(pending_question, fen_context=pending_fen)
                # Rerun rather than let this script run fall through to the
                # code below: chat_history was still empty at the `if not
                # st.session_state.chat_history:` check above (it's ABOVE
                # this block), so _render_example_prompts() already rendered
                # once this run despite chat_history being non-empty now --
                # a fresh rerun starts over with chat_history correctly
                # non-empty from the start, so that check skips it.
                st.rerun()

        submission = st.chat_input(
            "Ask about openings, positions, or chess history...",
            accept_file=True,
            file_type=["pgn"],
        )
        # Questions and answers are saved to the conversation_log table (see
        # src/ui/conversation_log.py). Stated here, next to the input itself,
        # rather than only in the README: the people typing into a public
        # demo are not the people reading its documentation, and this app's
        # own premise is disclosing its limits up front instead of leaving
        # them to be discovered.
        st.caption("Questions and answers are saved to help improve this demo.")
        if submission is not None:
            question = None
            if submission.files:
                question = _describe_uploaded_game(submission.files[0], submission.text)
            elif submission.text:
                question = submission.text
            if question is not None:
                with message_panel:
                    _submit_question(question)

    with board_col, st.container(key="board_panel"):
        # Keyed so styles.py can tighten the default ~16px gap Streamlit puts
        # between every element in this column. This column's natural content
        # runs taller than the chat column's, making it the binding
        # constraint on total page height -- see stMainBlockContainer's
        # padding comment in styles.py for the rest of that budget.
        render_board_panel()
        # Composed here rather than called from inside the board panel: the
        # recommendation cards are a separate concern that happens to share
        # this column, and keeping the call here is what lets board_panel.py
        # stay independent of the chat module it would otherwise import.
        st.divider()
        _render_resource_recommendations()
