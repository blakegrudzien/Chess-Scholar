"""The main (and only) screen: chat transcript (PGN attachments included --
see render_main_screen's chat_input), the board panel alongside it, and
the resource-recommendation cards -- everything that reads or writes
st.session_state.chat_history/board.
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
from src.ui.board_component import chess_board
from src.ui.conversation_log import log_conversation_best_effort
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

# Shared help= text for every raw FEN/ply caption in this file (the chat
# transcript's "Position: ..." captions, and the board panel's own "FEN: "/
# "-- ply N of M" captions) -- these are real chess notation terms with no
# obvious meaning to a reader who doesn't already play, and st.caption's
# help= renders a small hover tooltip for exactly this, rather than
# spelling either term out inline every time and cluttering what's meant
# to be a compact, glanceable caption.
_FEN_HELP = (
    "FEN (Forsyth-Edwards Notation): a compact text format that fully "
    "encodes a chess position -- where every piece is, whose turn it is, "
    "and a few other rules-relevant details."
)
_PLY_HELP = "A ply is one player's move -- White's 1st move is ply 1, Black's reply is ply 2, etc."

# Caps the paced reveal to the first N words of any single turn's text --
# without this, a long final answer (a real one ran 900+ words this
# session) pays STREAM_WORD_DELAY_SECONDS on every single word with no
# ceiling, adding 15-20+ seconds of pure artificial delay on top of the
# real generation/tool-call latency of an already-long multi-turn answer.
# The first N words still get the smooth typewriter feel; the rest of a
# long answer (or a tool-calling turn's rationale, which gets discarded by
# on_step a moment later regardless) appears immediately.
MAX_PACED_WORDS_PER_TURN = 40

# Cap on how many inline diagrams one answer shows -- "main line + a couple
# of popular sidelines" is usually 2-4 diagrams; a wall of positions on one
# answer works against scannability.
MAX_INLINE_DIAGRAMS = 4

# Height of the scrollable message panel (see render_main_screen). Trimmed
# down from 660 as part of fitting a fresh, question-less page inside a
# laptop viewport with no vertical scroll (a 14" MacBook's default logical
# resolution is the binding case) -- board_col's own natural height ends up
# the real floor either way (confirmed live), so this doesn't run short of
# board_col's typical height so much as stop needlessly exceeding it.
MESSAGE_PANEL_HEIGHT_PX = 560

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
    panel and streaming the final answer into view as it is generated,
    rather than a blank spinner followed by the whole answer appearing at
    once. Once the answer is complete, replaces the streamed plain text
    with the same text interleaved with any position diagrams it
    references (see _render_answer_content) -- so this renders everything
    itself, and callers should not render `answer` or its diagrams again
    afterward.

    history, if given, is prior turns the model should have context on --
    see _build_message_history and ask()'s own docstring. Without it, this
    call has no memory of anything asked earlier in the session.

    Returns (answer, touched_fens) -- touched_fens is every distinct
    (fen, label) pair on_position reported during this call (immediate
    repeat fens deduped), in the order they came up. label is None except
    for show_opening_line's calls. Sourced from any tool call that touches
    a real, verified position -- evaluate_chess_position, find_similar_
    corpus_games's top match, and show_opening_line's replayed sequence
    (see chess_agent.build_tools's on_position docstring for why each
    needs its own plumbing). Returned only so a caller can store it in
    chat_history for history replay -- already rendered here, not meant
    to be rendered again.

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
    looks like it is appearing in chunks rather than being typed. Only the
    first MAX_PACED_WORDS_PER_TURN words of any one turn get this treatment;
    past that, text still appears immediately (no delay), just not word by
    word -- a long final answer (or a multi-tool-call question's several
    rationale turns, each discarded a moment later anyway) shouldn't pay
    linear-in-length artificial delay on top of already-real generation and
    tool-call latency.

    The stop button doesn't need explicit click handling: Streamlit treats
    interactions as implicit yield points during a running script, so any
    click while this is in progress interrupts it at the next word reveal.

    Declared after answer_area, not before it -- a real, reported bug:
    message_panel autoscrolls to follow new content as it streams in (see
    render_main_screen), and a placeholder's position in the DOM is fixed
    at the point it's created, not where its content later fills in. With
    the button declared first, it stayed pinned above the streamed text,
    so a long answer scrolled it out of view above the fold right when a
    user most wants to reach it. Declaring it after answer_area puts it
    right below the growing text instead, exactly where autoscroll already
    keeps the view anchored.
    """
    status = st.status("Thinking...", expanded=True)
    answer_area = st.empty()
    stop_placeholder = st.empty()
    stop_placeholder.button("Stop generating", key="stop_generating")

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
            answer_area.markdown(_DIAGRAM_MARKER_RE.sub("", accumulated_text))
            words_this_turn += 1
            if words_this_turn <= MAX_PACED_WORDS_PER_TURN:
                time.sleep(STREAM_WORD_DELAY_SECONDS)

    def on_step(text: str) -> None:
        nonlocal accumulated_text, words_this_turn
        status.write(text)
        accumulated_text = ""
        words_this_turn = 0
        answer_area.empty()

    def on_position(fen: str, *, label: str | None = None, update_board: bool = True) -> None:
        # and game_path is None: while replaying a recommended game,
        # st.session_state.board is the free-play board sitting *behind*
        # the replay, invisible but still live -- without this guard, an
        # Evaluate/Ask-position call made against a replay ply (or any
        # other tool call the model makes that turn, e.g.
        # find_similar_corpus_games reporting an unrelated game's
        # position) would silently overwrite it the moment on_position
        # fires, corrupting it with no visible symptom until the user
        # exits replay and finds their game changed underneath them.
        if update_board and st.session_state.game_path is None:
            try:
                new_board = chess.Board(fen)
            except ValueError:
                return  # the model can pass a malformed FEN to the tool call
                # even though evaluate_chess_position's own validation later
                # rejects it for that turn -- leave the displayed board as it was.
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
        # Covers every real network/API failure mode (rate limit, timeout,
        # connection drop, a transient 5xx/overloaded error from Anthropic
        # itself) -- none of this was caught anywhere before. tool_runner
        # only catches exceptions a *tool* raises (see build_tools' own
        # docstring); the SDK's own calls to Anthropic sit outside that,
        # so this used to propagate all the way up as a raw, unhandled
        # exception, crashing the whole Streamlit script mid-answer.
        # logger.exception, not just logger.error: this is a real failure
        # worth a full traceback in the logs, not just a one-line note.
        logger.exception("ask_agent failed")
        answer = ""

    if not answer.strip():
        # The tool-calling loop can end on a turn whose only content was a
        # tool call (see chess_agent.ask()'s docstring on how final_text is
        # tracked) -- rare, but when it happens the chat bubble would
        # otherwise render nothing at all with no indication anything went
        # wrong. A short, honest placeholder beats silence. Also the
        # fallback for the API-error case just above: an empty answer
        # already has well-tested, working UI for "let the user know and
        # let them retry" -- no need for a second, parallel message.
        answer = (
            "Something interrupted this response before it finished -- "
            "try asking again, possibly with a narrower question."
        )
        answer_area.markdown(answer)
    else:
        # answer_area currently holds the plain streamed text (no
        # diagrams) -- .empty() + re-entering .container() replaces that
        # whole group of elements rather than appending after it, so the
        # interleaved version takes its place instead of duplicating it.
        answer_area.empty()
        with answer_area.container():
            _render_answer_content(answer, touched_fens)

    status.update(label="Done", state="complete", expanded=False)
    stop_placeholder.empty()
    return answer, touched_fens


def _render_resource_recommendations() -> None:
    """Offers to look up related Lichess studies and corpus games for the
    most recent question, and renders whatever comes back.

    Only ever shown for the latest exchange, not every past one in the
    history -- keeps the UI focused on what is actually in view rather than
    accumulating a lookup button per message.

    The button itself always renders, even on a fresh page with no
    conversation yet -- disabled rather than absent, so it's a visible
    preview of a feature ("try this once you've asked something") instead
    of UI that pops into existence with no warning partway through a
    session. This is also what makes it a viable tutorial_overlay target:
    an element that only exists conditionally can't be reliably spotlighted
    from a fresh page, where the tour is most likely to be opened.

    st.session_state.resource_recommendations is the sentinel for "already
    looked up this question": None means not yet requested (show the
    button), a list (possibly empty, meaning nothing was relevant) means it
    has been. Reset to None right after a new answer is appended in
    _submit_question, so a fresh question always gets a fresh button.
    """
    history = st.session_state.chat_history
    eligible = len(history) >= 2 and history[-1][0] == "assistant"

    if st.session_state.resource_recommendations is None:
        # secondary, not primary -- "Evaluate this position with Stockfish"
        # is the page's one primary action; two competing primary-styled
        # buttons dilute the visual hierarchy that color is supposed to
        # establish (Refactoring UI's "distinguish an interface's actions
        # by importance" principle -- one dominant action per view, not
        # several equally loud ones).
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
                # Same unguarded-API-call gap as ask_with_status's call to
                # ask_agent (see that fix's own comment) -- this call
                # chains Anthropic + Voyage + live Lichess HTTP behind one
                # click with nothing catching a failure in any of them.
                # Left inside the `with` block (not wrapped around it) so
                # `status` is still live to update here, and resource_
                # recommendations stays None -- the button reappears on
                # the next rerun instead of wrongly claiming nothing was
                # found.
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
    index 0 -- the sequence _render_board_panel's Prev/Next steps through
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
    with tempfile.NamedTemporaryFile(suffix=".pgn", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        # Only the first game is ever used below, so pull at most two from
        # the generator: the first to use, and a second only to learn
        # whether there's more than one, without fully parsing an upload
        # (untrusted input) that could contain a large number of games.
        first_two_games = list(itertools.islice(parse_pgn(tmp_path, source="user_upload"), 2))
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
    label it already passed to show_opening_line -- an earlier version of
    this tried to *find* that label by searching the answer's free-form
    prose for it, which almost never matched (nothing obliges the model to
    repeat a tool argument verbatim in its synthesis), so diagrams still
    landed at the end regardless. A marker the model is told to write is a
    real contract; a string search against text it wasn't told to shape
    around that string is not.

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
            st.caption(f"Position: `{fen_context}`", help=_FEN_HELP)
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
    chat_history before calling this) -- teaches the chat's actual range
    by demonstration, one example per layer/feature, the way ChatGPT/
    Claude.ai/Gemini all show suggested prompts on a fresh conversation,
    rather than by upfront explanation (see the "How this works" spotlight
    tour, app.py's render_tutorial_trigger, for that). Clicking one submits
    it, the same as typing it and pressing enter would.

    Stashes into st.session_state.pending_question and reruns rather than
    calling _submit_question directly -- a real, reproduced bug: this
    function is called from inside the `if not st.session_state.
    chat_history:` branch in render_main_screen, *before* that same
    script run's `for ... in st.session_state.chat_history:` loop below
    it. Calling _submit_question inline here mutates and renders the new
    question/answer pair on the spot, and then the loop right after
    renders that same now-nonempty chat_history all over again, in the
    same run -- every message doubled, plus the example buttons still
    visible above them since the `if not chat_history` check had already
    committed to true for this run. Stashing and rerunning, the same
    pattern _render_board_panel's own triggers already use for the
    identical reason (see that function's docstring), lets a fresh script
    run make the correct decision: chat_history is non-empty from the
    start, so this function is skipped entirely and the loop renders the
    new pair exactly once.
    """
    st.caption("Try asking:")
    examples = [
        # The flagship demo query from CLAUDE.md: Layer 1 stats + Layer 2
        # strategic prose, synthesized together.
        "How should White meet the Sicilian Defense?",
        "Evaluate 1. e4 e5 2. Qh5 for White",  # Layer 3, engine grounding
        "What's the plan behind an isolated queen pawn?",  # Layer 2, conceptual
        # Trend synthesis (CLAUDE.md), a distinct supported feature none
        # of the other three examples touch.
        "How has the King's Indian Defense's popularity changed over time?",
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
                        st.caption(f"Position: `{fen}`", help=_FEN_HELP)

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
        # Keyed purely so styles.py can tighten the default ~16px gap
        # Streamlit puts between every element in this column -- board_col
        # was the binding constraint on the whole page's height (its own
        # natural content ran taller than chat_col's), so this is a big
        # part of fitting an unscrolled, question-less page inside a
        # laptop viewport (see stMainBlockContainer's own padding comment
        # in styles.py for the rest of that budget).
        _render_board_panel()


def _attempt_move(source: str, target: str) -> None:
    """Validate a drag-and-drop {from, to} pair against python-chess -- the
    single source of truth for legality, mirroring how handle_square_click
    used to work but for a one-shot from/to pair instead of two separate
    clicks. Falls back to auto-queen promotion, same as before.

    source/target come back from chess_board() as plain strings the custom
    component's own JS chose to send -- in normal use always a real drag's
    two algebraic squares, but nothing on the Python side enforces that
    shape, so a malformed pair (not a legal chess.Move.from_uci input at
    all, as opposed to merely an illegal move) must be handled the same
    honest way as any other illegal drop instead of raising InvalidMoveError
    straight through to Streamlit's default full-traceback error page.

    Always bumps board_generation, including on rejection: an illegal drop
    leaves board.fen() textually identical to what it was before the drop,
    so without a distinct generation value the component has no signal that
    it needs to re-render and snap the piece back -- see chess_board()'s own
    docstring for why this is required, not just belt-and-suspenders.
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


def _render_board_panel() -> None:
    """The board column: a draggable board reflecting the position under
    discussion, plus a quick-eval button, Reset/Undo, and a free-text
    "ask about this position" form -- or, while replaying a recommended
    game (st.session_state.game_path is not None; see
    _render_resource_recommendations' "Play through this game" button), a
    read-only board stepping through that game's moves instead, with
    Evaluate/Ask-position operating on whatever ply is currently shown.

    Optimistic UI, no client-side legality check (see board_component's own
    docstring): chess_board() lets a drop land wherever it was dropped, and
    _attempt_move validates it here against python-chess. An illegal drop
    leaves st.session_state.board unchanged, so the next render's `data`
    (the still-unmoved FEN) reverts the piece visually via chessboard.js's
    own diffing -- no explicit snapback handling needed on either side.
    Disabled entirely during replay (draggable=False) rather than left on
    and silently ignored: _attempt_move mutates st.session_state.board
    unconditionally, which during replay is the free-play board sitting
    *behind* the visible replay position, not a copy -- a drag left enabled
    there would corrupt it invisibly while the piece you actually see just
    snaps back with no feedback at all.

    chess_board() only ever returns a given drop once -- it's a Streamlit
    "trigger" value, documented as transient (resets to None after one
    script run), unlike the older declare_component API's return values,
    which persist until explicitly replaced. No dedup bookkeeping needed.

    Neither button below calls ask_with_status directly -- both stash a
    (question, fen) pair in st.session_state.pending_question and
    st.rerun() instead, so the actual submission (and its live status/
    streaming UI) happens from inside chat_col on the next script pass, not
    in this narrow column. See _submit_question's docstring for why.
    """
    replaying = st.session_state.game_path is not None
    if replaying:
        current_board = chess.Board(st.session_state.game_path[st.session_state.game_path_index])
    else:
        # The same object stored in session_state, not a copy -- Undo's
        # current_board.pop() below still mutates st.session_state.board
        # in place, exactly as it did before this function had a
        # replay/free-play distinction to make.
        current_board = st.session_state.board

    # Turn/FEN/Reset/Undo (or, during replay, the ply label and Prev/Next)
    # sit beside the board rather than stacked below it -- purely a height
    # trade: board_col was running much taller than chat_col (usually
    # near-empty until a conversation starts), and moving these compact,
    # low-priority controls beside the board instead of under it closes
    # most of that gap. The Evaluate button and the ask form stay
    # full-width below both -- those are the primary actions, not
    # incidental status/controls, and read better as one wide row each
    # than squeezed into this side column too.
    #
    # [5, 2], not [3, 2] -- chess_board()'s size=340 below is a floor, not
    # a target: chessboard.js fills its container when there's more than
    # 340px to give it, but never shrinks under that regardless of how
    # little room board_display_col actually has. [3, 2] passed that floor
    # at a 14" MacBook's own default width (confirmed live: a 28px real
    # overflow into controls_col, not just a tight fit) despite looking
    # fine at the wider viewports actually tested at the time -- [5, 2]
    # keeps real margin above 340px at that narrower width too.
    board_display_col, controls_col = st.columns([5, 2])

    with board_display_col:
        # Wrapped in a keyed container (not passed as chess_board()'s own
        # `key=`) purely so the tutorial overlay has a stable
        # `.st-key-tutorial_board_target` selector to spotlight -- giving
        # chess_board() itself a key would make Streamlit treat it as one
        # persistent instance and stop remounting it when `data` changes,
        # which is exactly the mechanism board_generation relies on for
        # illegal-move snapback and Undo (see chess_board()'s docstring).
        with st.container(key="tutorial_board_target"):
            # size=340, down from the old standalone tab's 400 -- this board
            # shares horizontal space with chat instead of the full page.
            drop = chess_board(
                current_board.fen(),
                size=340,
                generation=st.session_state.board_generation,
                draggable=not replaying,
            )
        if drop is not None and not replaying:
            _attempt_move(drop["from"], drop["to"])
            st.rerun()

    with controls_col:
        st.caption(f"Turn: {'White' if current_board.turn else 'Black'}")
        # A caption with inline code, not st.code() -- a single short FEN
        # line doesn't need its own full dark code panel (heavy padding, a
        # copy button); the same subtle inline-code style already used for
        # the "Position: `{fen}`" captions elsewhere in this file reads as
        # plain, quiet text instead of a block that sticks out -- and
        # wraps naturally across a few lines in this narrower column.
        st.caption(f"FEN: `{current_board.fen()}`", help=_FEN_HELP)

        if replaying:
            index = st.session_state.game_path_index
            last_index = len(st.session_state.game_path) - 1
            st.caption(
                f"{st.session_state.game_path_label} -- ply {index} of {last_index}",
                help=_PLY_HELP,
            )
            # Stacked, not side by side, matching Reset/Undo's own layout
            # below -- this column is too narrow for two buttons abreast.
            # shortcut="Left"/"Right": a native st.button param (confirmed
            # live, not assumed) that correctly stays out of the way while
            # any text input has focus (arrow keys still move the text
            # cursor normally there) but fires globally otherwise, so this
            # needs no custom keyboard-handling component.
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
            if st.session_state.last_illegal_attempt is not None:
                st.warning("That move isn't legal. Try again.")

            # No more expander: it existed only to visually corral the old
            # 8x8 click-grid the user found "ugly and disjointed", and with
            # dragging directly on the board replacing that grid, these are
            # core, expected board interactions, not something to hide
            # behind a click.
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

    # Narrower than the Evaluate button above it, not full board_col width --
    # a single short question doesn't need the whole column, and the
    # trailing empty column reads as intentional breathing room rather than
    # a layout accident since Evaluate (a full-width primary action) is
    # directly above it for contrast.
    #
    # Wrapped in st.container(key="position_question_form"), same reason
    # tutorial_board_target wraps the board: st.form's own key= lands on
    # its internal FormSubmitter button (confirmed live -- it renders as
    # st-key-FormSubmitter-<form_key>-<button_label>), never on the form's
    # own element, so tutorial_overlay's ".st-key-position_question_form"
    # selector could never have matched anything. The form itself keeps a
    # separate, unrelated key (Streamlit forbids two elements sharing one
    # key in the same run).
    form_col, _ = st.columns([3, 1])
    with (
        form_col,
        st.container(key="position_question_form"),
        st.form("position_question_form_widget", clear_on_submit=True),
    ):
        # placeholder (text inside the box), not the visible label above it
        # -- matches the main chat_input's own look, which shows its
        # prompt the same way. label_visibility="collapsed", not omitting
        # the label entirely: a real label is still required for
        # screen-reader accessibility, just not shown visually.
        position_question = st.text_input(
            "Ask about this position...",
            placeholder="Ask about this position...",
            label_visibility="collapsed",
        )
        asked = st.form_submit_button("Ask")
    if asked and position_question:
        st.session_state.pending_question = (position_question, current_board.fen())
        st.rerun()

    st.divider()
    _render_resource_recommendations()
