"""The main (and only) screen: chat transcript (PGN attachments included --
see render_main_screen's chat_input), the board panel alongside it, and
the resource-recommendation cards -- everything that reads or writes
st.session_state.chat_history/board.
"""

from __future__ import annotations

import html
import io
import itertools
import os
import re
import tempfile
import time
from collections.abc import Callable

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
from src.ui.resources import get_anthropic_client, get_db_pool, get_engine_pool, get_voyage

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

# Height of the scrollable message panel (see render_main_screen) -- roughly
# matches board_col's own typical height (board + its side controls +
# Evaluate button + the ask-position form), so neither column reads as
# oddly taller than the other on a fresh page load with no messages yet.
MESSAGE_PANEL_HEIGHT_PX = 600

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
    """
    status = st.status("Thinking...", expanded=True)
    stop_placeholder = st.empty()
    stop_placeholder.button("Stop generating", key="stop_generating")
    answer_area = st.empty()

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

    answer = ask_agent(
        question, on_step=on_step, on_chunk=on_chunk, on_position=on_position, history=history
    )

    if not answer.strip():
        # The tool-calling loop can end on a turn whose only content was a
        # tool call (see chess_agent.ask()'s docstring on how final_text is
        # tracked) -- rare, but when it happens the chat bubble would
        # otherwise render nothing at all with no indication anything went
        # wrong. A short, honest placeholder beats silence.
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
    history = _build_message_history()  # prior turns, before this one is appended below
    st.session_state.chat_history.append(("user", question, fen_context, []))
    with st.chat_message("user", avatar=_CHAT_AVATARS["user"]):
        st.markdown(question)
        if fen_context is not None:
            st.caption(f"Position: `{fen_context}`")
    sent_question = _to_model_text(question, fen_context)
    with st.chat_message("assistant", avatar=_CHAT_AVATARS["assistant"]):
        answer, touched_fens = ask_with_status(sent_question, history=history)
    st.session_state.chat_history.append(("assistant", answer, None, touched_fens))
    st.session_state.resource_recommendations = None


def render_main_screen() -> None:
    st.caption(
        "Answers synthesize retrieved human commentary and engine output -- "
        "not the model's own independent chess judgment."
    )
    st.caption(
        "Attach a PGN of your own game to the chat below to find similar games in the corpus."
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "resource_recommendations" not in st.session_state:
        st.session_state.resource_recommendations = None
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "last_illegal_attempt" not in st.session_state:
        st.session_state.last_illegal_attempt = None
    if "pending_board_question" not in st.session_state:
        st.session_state.pending_board_question = None
    if "board_generation" not in st.session_state:
        st.session_state.board_generation = 0
    if "game_path" not in st.session_state:
        st.session_state.game_path = None
    if "game_path_index" not in st.session_state:
        st.session_state.game_path_index = 0
    if "game_path_label" not in st.session_state:
        st.session_state.game_path_label = None

    chat_col, board_col = st.columns([5, 4])

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
        message_panel = st.container(height=MESSAGE_PANEL_HEIGHT_PX, border=True)
        with message_panel:
            for role, content, fen, image_fens in st.session_state.chat_history:
                with st.chat_message(role, avatar=_CHAT_AVATARS[role]):
                    if role == "assistant":
                        _render_answer_content(content, image_fens)
                    else:
                        st.markdown(content)
                    if fen is not None:
                        st.caption(f"Position: `{fen}`")

            pending = st.session_state.pending_board_question
            if pending is not None:
                st.session_state.pending_board_question = None
                pending_question, pending_fen = pending
                _submit_question(pending_question, fen_context=pending_fen)

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

    with board_col:
        _render_board_panel()


def _attempt_move(source: str, target: str) -> None:
    """Validate a drag-and-drop {from, to} pair against python-chess -- the
    single source of truth for legality, mirroring how handle_square_click
    used to work but for a one-shot from/to pair instead of two separate
    clicks. Falls back to auto-queen promotion, same as before.

    Always bumps board_generation, including on rejection: an illegal drop
    leaves board.fen() textually identical to what it was before the drop,
    so without a distinct generation value the component has no signal that
    it needs to re-render and snap the piece back -- see chess_board()'s own
    docstring for why this is required, not just belt-and-suspenders.
    """
    board: chess.Board = st.session_state.board
    move = chess.Move.from_uci(source + target)
    if move not in board.legal_moves:
        move = chess.Move.from_uci(source + target + "q")
    if move in board.legal_moves:
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
    (question, fen) pair in st.session_state.pending_board_question and
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
    board_display_col, controls_col = st.columns([2, 1])

    with board_display_col:
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
        st.caption(f"FEN: `{current_board.fen()}`")

        if replaying:
            index = st.session_state.game_path_index
            last_index = len(st.session_state.game_path) - 1
            st.caption(f"{st.session_state.game_path_label} -- ply {index} of {last_index}")
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

    if st.button("Evaluate this position with Stockfish", type="primary"):
        st.session_state.pending_board_question = (
            "Evaluate this chess position and tell me the best move. "
            "Use the engine, don't just guess.",
            current_board.fen(),
        )
        st.rerun()

    with st.form("position_question_form", clear_on_submit=True):
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
        st.session_state.pending_board_question = (position_question, current_board.fen())
        st.rerun()

    st.divider()
    _render_resource_recommendations()
