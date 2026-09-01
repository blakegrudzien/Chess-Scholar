"""Agent orchestration: Claude Sonnet 5 native tool-calling across the four
layers (structured search, vector RAG, Stockfish, personalized similarity).
Per CLAUDE.md, routing is done by the model via tool-calling, not a
hand-rolled intent classifier.
"""

from __future__ import annotations

from collections.abc import Callable

import anthropic
import chess
import psycopg2.pool
import voyageai
from anthropic import beta_tool

from src.engine.engine_pool import EngineBusyError, EnginePool
from src.engine.stockfish_eval import DEFAULT_DEPTH, evaluate_position
from src.ingestion.db_loader import get_connection_with_timeout
from src.personalization.similarity import find_similar_games as _find_similar_games
from src.rag.vector_search import search_chunks
from src.search.structured_search import common_moves_at_ply, eco_summary, piece_placement_frequency

MODEL = "claude-sonnet-5"
# Doubled from the original 4096 after a real, reproduced failure: a turn
# combining several tool calls (e.g. three show_opening_line diagrams plus
# their rationale) plus growing conversation history can consume enough
# output tokens that generation gets cut off mid-tool-call, producing a
# tool_use block with an incomplete `input` dict -- the direct cause of a
# KeyError crash in _report_position_update (see its own defensive fix)
# and almost certainly a contributor to turns ending without ever reaching
# a clean text synthesis (see _recover_synthesis). More headroom doesn't
# eliminate the possibility, just makes it meaningfully less likely.
MAX_TOKENS = 8192

SYSTEM_PROMPT = """You are a chess research assistant/mentor with access to a corpus \
of grandmaster games, book/annotation text, and a chess engine, via tools.

Tool selection:
- get_eco_summary / get_piece_placement / get_common_moves_at_ply: \
deterministic statistics from the game database (Layer 1). Use these for \
"how often", "where does X usually go", "what's the most common move" questions.
- search_annotations: semantic search over human-written commentary and book \
text (Layer 2). Use this for "why", "what's the plan", and conceptual/strategic \
questions. This returns a synthesis of retrieved human text, not your own \
independent judgment -- attribute ideas to the retrieved material.
- evaluate_chess_position: Stockfish ground truth (Layer 3). You MUST call \
this before making any claim about whether a move or position is good, \
winning, a mistake, or a blunder. Never judge tactical soundness from your \
own knowledge alone -- you are not a substitute for engine analysis.
- find_similar_corpus_games: opening-move-prefix matching against a user's \
own game (Layer 4). This is an approximate, illustrative comparison based on \
exact opening moves, not a rigorous positional match -- always tell the user \
this is illustrative, not authoritative, when you use it.
- show_opening_line: renders a labeled board diagram for a specific move \
sequence. For opening-theory questions (ECO stats, common-move, or \
strategic-plan answers), proactively call this for the main line and any \
named sidelines you discuss, instead of only describing moves in prose -- \
readers should be able to see the position, not just read algebraic \
notation. Scoped to opening theory, not a substitute for \
evaluate_chess_position when judging whether a move or plan is good.

Combine tools when a question calls for it (e.g. stats + commentary for an \
opening-profile question). Be direct about which tool(s) you used.

For broad "walk me through" or "what are the options against X" questions, \
prefer one well-chosen tool call per layer over several near-duplicate \
calls (e.g. one get_eco_summary for the most relevant ECO code, not one \
per code in the family) -- every tool call is a real network round trip, \
and a focused, useful answer that arrives promptly beats an exhaustive one \
that takes minutes. Two or three show_opening_line diagrams (a main line \
plus the most relevant sidelines) is plenty; skip minor branches.

Before calling a tool, state in one short sentence which layer you're using \
and why -- e.g. "Checking Layer 2 for strategic ideas about isolated pawns." \
This sentence is shown to the user live while they wait; keep it brief and \
write it as a standalone status update, not as part of your eventual answer.

A user message may start with a line like "Current board position: <FEN>" -- \
this means they set up that position on their own board and are asking about \
it specifically. Treat it as the position in question, not something to \
independently derive or guess at.
"""

# Fallback status text per tool, shown if a turn's tool call arrives with no
# accompanying sentence from the model (see _report_tool_steps). Doubles as
# a human readable label for which of the four layers each tool belongs to.
TOOL_LABELS: dict[str, str] = {
    "get_eco_summary": "Layer 1 (structured search): pulling opening statistics.",
    "get_piece_placement": "Layer 1 (structured search): checking piece placement frequency.",
    "get_common_moves_at_ply": "Layer 1 (structured search): checking common moves.",
    "search_annotations": "Layer 2 (semantic search): searching commentary and book text.",
    "evaluate_chess_position": "Layer 3 (Stockfish): evaluating the position.",
    "find_similar_corpus_games": "Layer 4 (similarity search): comparing against the corpus.",
    "show_opening_line": "Rendering a position diagram.",
}


DB_RETRYABLE_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)

# Original attempt plus one retry on a dropped connection. Named rather than
# inlined as a bare 2, so the retry budget is a single, intentional value
# instead of a number a reader has to infer the meaning of.
MAX_QUERY_ATTEMPTS = 2


def build_tools(
    db_pool: psycopg2.pool.ThreadedConnectionPool,
    engine_pool: EnginePool,
    voyage_client: voyageai.Client,
    on_position: Callable[..., None] | None = None,
) -> list[Callable]:
    """Build the tool functions for one session, bound to the given DB pool,
    Stockfish engine pool, and Voyage client.

    Each tool call checks a connection out of `db_pool` and returns it
    afterward, rather than holding one connection for the whole session --
    this is what lets concurrent sessions run without sharing a connection.
    On a dead connection (Neon closes idle connections in practice), the
    pool discards it and hands back a fresh one, all inside the tool call:
    the Anthropic SDK's tool_runner catches every exception a tool raises
    and turns it into a tool_result error sent back to the model, so a
    raised psycopg2 error never reaches the caller of `ask()` to trigger a
    retry there.

    on_position, if given, is also called directly from
    find_similar_corpus_games with its top match's FEN (when it has one) --
    unlike evaluate_chess_position's FEN, which is a tool *argument* the
    model supplies and ask() can read straight off the tool_use block, this
    one only exists in the tool's *output* (a DB query result the model
    never sees as a discrete value), so it needs this direct, in-line
    callback instead of a shared message-scanning mechanism.

    Full signature callers can expect: on_position(fen, *, label=None,
    update_board=True). show_opening_line is the one caller that passes
    label and update_board=False -- an illustrative example line isn't the
    position a caller's UI should treat as "currently under discussion" the
    way an actual evaluation or matched game is.
    """

    def _query(fn: Callable, *args, **kwargs):
        conn = get_connection_with_timeout(db_pool)
        for attempt in range(MAX_QUERY_ATTEMPTS):
            try:
                result = fn(conn, *args, **kwargs)
            except DB_RETRYABLE_ERRORS:
                # This connection is confirmed dead either way: discard it
                # rather than returning it to the pool for the next checkout
                # to fail on too.
                db_pool.putconn(conn, close=True)
                if attempt == MAX_QUERY_ATTEMPTS - 1:
                    raise
                conn = get_connection_with_timeout(db_pool)
                continue
            except Exception:
                # Not a connection problem, so the connection itself is
                # still healthy: return it normally before letting the
                # caller's error (e.g. a bad argument) propagate.
                db_pool.putconn(conn)
                raise
            else:
                db_pool.putconn(conn)
                return result
        # Every iteration above ends in return or raise (the last attempt's
        # except clause always raises instead of looping again), so this is
        # unreachable. Stated explicitly rather than left as an implicit gap:
        # it tells a type checker (and a future reader who changes
        # MAX_QUERY_ATTEMPTS) that falling out of the loop is a bug, not a
        # valid path returning None.
        raise AssertionError("unreachable: every _query attempt returns or raises")

    @beta_tool
    def get_eco_summary(eco_code: str) -> str:
        """Get game count, White/Black/draw breakdown, and average game
        length for a chess opening in the corpus.

        Args:
            eco_code: ECO opening code, e.g. "C50", "D12", "B90".
        """
        summary = _query(eco_summary, eco_code)
        if summary.game_count == 0:
            return f"No games found for ECO {eco_code}."
        return (
            f"{eco_code}: {summary.game_count} games. "
            f"White wins {summary.white_wins}, Black wins {summary.black_wins}, "
            f"draws {summary.draws}. Average game length "
            f"{summary.avg_ply_count:.1f} plies."
        )

    @beta_tool
    def get_piece_placement(
        eco_code: str, piece: str, color: str = "both", max_ply: int = 20
    ) -> str:
        """Find the most common destination squares for a piece type during
        the opening phase of games in a given ECO code.

        Args:
            eco_code: ECO opening code, e.g. "D12".
            piece: Piece letter: P, N, B, R, Q, or K.
            color: "white", "black", or "both". Defaults to "both".
            max_ply: How many half-moves into the game to consider. Defaults to 20.
        """
        try:
            results = _query(
                piece_placement_frequency, eco_code, piece, color=color, max_ply=max_ply
            )
        except ValueError as exc:
            return f"Invalid input: {exc}"
        if not results:
            return f"No data for piece {piece} in ECO {eco_code}."
        return "; ".join(f"{r.square} ({r.count}x)" for r in results)

    @beta_tool
    def get_common_moves_at_ply(eco_code: str, ply: int, limit: int = 5) -> str:
        """Find the most frequently played move at an exact half-move number
        among games in a given ECO code.

        Args:
            eco_code: ECO opening code.
            ply: Half-move number (1 = White's 1st move, 2 = Black's 1st move, etc.)
            limit: Max number of moves to return. Defaults to 5.
        """
        results = _query(common_moves_at_ply, eco_code, ply, limit=limit)
        if not results:
            return f"No data at ply {ply} for ECO {eco_code}."
        return "; ".join(f"{r.move_san} ({r.count}x)" for r in results)

    @beta_tool
    def search_annotations(query: str, limit: int = 5) -> str:
        """Semantic search over human-written game annotations and book
        commentary for conceptual/strategic ideas, plans, and explanations.

        Args:
            query: Natural-language description of the idea or concept to search for.
            limit: Max number of results. Defaults to 5.
        """
        results = _query(search_chunks, voyage_client, query, limit=limit)
        if not results:
            return "No relevant annotations found."
        return "\n".join(f"- {r.text}" for r in results)

    @beta_tool
    def evaluate_chess_position(fen: str, depth: int = DEFAULT_DEPTH) -> str:
        """Get Stockfish's ground-truth evaluation of a chess position.
        Always call this before judging whether a move or position is good.

        Args:
            fen: The position in FEN notation.
            depth: Search depth; higher is more accurate but slower. Defaults to 18.
        """
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            return f"Invalid FEN: {exc}"
        try:
            with engine_pool.checkout() as engine:
                result = evaluate_position(engine, board, depth=depth)
        except EngineBusyError as exc:
            return str(exc)
        if result.mate_in is not None:
            return (
                f"Mate in {result.mate_in}. Best move: {result.best_move_san}. "
                f"Line: {' '.join(result.pv_san)}"
            )
        return (
            f"Evaluation: {result.score_cp} centipawns (from the side to move's "
            f"perspective). Best move: {result.best_move_san}. "
            f"Line: {' '.join(result.pv_san)}"
        )

    @beta_tool
    def find_similar_corpus_games(moves: list[str], max_ply: int = 20, limit: int = 5) -> str:
        """Find games in the corpus with the longest matching opening-move
        sequence to a user-provided game. This is an approximate,
        illustrative comparison based on exact opening moves, not a
        rigorous positional match.

        Args:
            moves: Moves in SAN notation from the user's game, e.g. ["e4", "e5", "Nf3"].
            max_ply: How many half-moves to compare. Defaults to 20.
            limit: Max number of similar games to return. Defaults to 5.
        """
        try:
            results = _query(_find_similar_games, moves, max_ply=max_ply, limit=limit)
        except ValueError as exc:
            return f"Invalid input: {exc}"
        if not results:
            return "No similar games found."
        if on_position is not None and results[0].fen_after is not None:
            on_position(results[0].fen_after)
        return "\n".join(
            f"{r.matching_plies} matching plies: {r.white} vs {r.black} "
            f"({r.year}, {r.eco_code}, {r.result})"
            for r in results
        )

    @beta_tool
    def show_opening_line(moves: list[str], label: str) -> str:
        """Render a labeled board diagram for a specific, named sequence of
        opening moves -- e.g. the main line of a variation, or a named
        sideline you're discussing by name. Moves are replayed and validated
        for legality; an illegal move fails the call rather than showing
        something wrong. Scoped to opening theory (a short, known sequence
        from the starting position), not a substitute for
        evaluate_chess_position when judging whether a move or plan is good.

        Args:
            moves: Moves in SAN notation from the starting position, e.g. ["Nf3", "d5", "g3"].
            label: A short name for this line, shown next to the diagram,
                e.g. "Main line" or "Yugoslav queenside expansion".
        """
        board = chess.Board()
        for i, san in enumerate(moves, start=1):
            try:
                board.push_san(san)
            except ValueError:
                return f"'{san}' (move {i}) isn't legal in that sequence -- diagram not shown."
        if on_position is not None:
            on_position(board.fen(), label=label, update_board=False)
        return f"Shown: {label} ({' '.join(moves)})"

    return [
        get_eco_summary,
        get_piece_placement,
        get_common_moves_at_ply,
        search_annotations,
        evaluate_chess_position,
        find_similar_corpus_games,
        show_opening_line,
    ]


def _report_tool_steps(message, on_step: Callable[[str], None]) -> None:
    """Surface one status line for a turn that calls tool(s), preferring the
    model's own stated rationale (see the "before calling a tool" system
    prompt instruction) and falling back to TOOL_LABELS if a turn's tool_use
    happens to arrive with no accompanying sentence.
    """
    tool_names = [block.name for block in message.content if block.type == "tool_use"]
    if not tool_names:
        return
    rationale = "".join(block.text for block in message.content if block.type == "text").strip()
    if not rationale:
        rationale = " / ".join(TOOL_LABELS.get(name, f"Calling {name}...") for name in tool_names)
    on_step(rationale)


def _report_position_update(message, on_position: Callable[..., None]) -> None:
    """Surface the FEN behind a turn's evaluate_chess_position call, if any --
    the only tool whose FEN is a direct *argument* the model supplies (as
    opposed to find_similar_corpus_games' DB-derived top match, or
    show_opening_line's replayed-from-SAN result, both reported via their
    own in-line on_position calls instead of this message-scanning path).

    Reads the raw tool_use block directly, independent of whether the SDK
    later successfully dispatches the actual tool call -- a turn cut off
    mid-generation (see MAX_TOKENS's comment) can yield a tool_use block
    whose `input` is missing "fen" entirely. block.input.get(...) rather
    than block.input[...] is deliberate: a real, reproduced KeyError here
    crashed the whole app, for a case where simply not reporting a position
    update (leaving the displayed board as it was) is a fine fallback.
    """
    for block in message.content:
        if block.type == "tool_use" and block.name == "evaluate_chess_position":
            fen = block.input.get("fen")
            if fen is not None:
                on_position(fen)
            return


def _recover_synthesis(
    client: anthropic.Anthropic, runner, on_chunk: Callable[[str], None] | None
) -> str:
    """Force a real answer when the tool-calling loop above ends without
    one, by continuing the exact conversation the runner already built and
    explicitly asking for the synthesis that turn should have produced.

    runner._params["messages"] is an internal, unversioned attribute, not a
    guess -- confirmed no public equivalent exists in this SDK version by
    listing the runner's own public API directly (only append_messages,
    set_messages_params, generate_tool_call_response, and until_done are
    exposed; none return the accumulated message list). Anthropic's own
    compaction_control implementation, a few lines away in this same SDK
    file, reads this identical private attribute for the same reason --
    that's the basis for treating it as a reasonable extension point here,
    not a fragile guess, though it may need revisiting on an SDK upgrade.
    """
    messages = list(runner._params["messages"])
    # A trailing assistant message with an unresolved tool_use block can't
    # be followed directly by a new user turn -- the API requires a
    # matching tool_result immediately after any tool_use. Same fix
    # compaction_control's own code applies, for the identical reason.
    if messages and messages[-1]["role"] == "assistant":
        non_tool_blocks = [
            block
            for block in messages[-1]["content"]
            if not (isinstance(block, dict) and block.get("type") == "tool_use")
        ]
        if non_tool_blocks:
            messages[-1] = {**messages[-1], "content": non_tool_blocks}
        else:
            messages = messages[:-1]

    messages.append(
        {
            "role": "user",
            "content": (
                "Please give your complete answer now, based on everything you've found so far."
            ),
        }
    )
    response = client.beta.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, messages=messages
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    if on_chunk is not None and text:
        on_chunk(text)
    return text


def ask(
    question: str,
    db_pool: psycopg2.pool.ThreadedConnectionPool,
    engine_pool: EnginePool,
    voyage_client: voyageai.Client,
    client: anthropic.Anthropic | None = None,
    on_step: Callable[[str], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
    on_position: Callable[..., None] | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Answer one question, routing across the four layers via tool-calling.
    Returns the final response text.

    If `on_step` is given, it's called once per turn that includes a tool
    call, with a short status string describing which layer is being used
    and why -- meant for a caller to show live while the request is in
    flight, since a full answer can take several sequential tool-calling
    round trips.

    If `on_chunk` is given, it's called with each raw text delta as it
    streams in, for every turn, not only the final one. There's no way to
    know in advance whether a given turn will turn out to be the final
    answer or a tool-calling turn's one-sentence rationale, since the
    tool_use block (if any) only appears once the turn completes -- so
    on_chunk fires for both, and a caller distinguishes them after the fact:
    on_step firing for a turn means its chunks were the rationale, not the
    final answer.

    If `on_position` is given, it's called with the FEN whenever a turn
    calls evaluate_chess_position -- meant for a caller to keep a displayed
    board in sync with whatever position the conversation just touched.

    `history`, if given, is prior turns as plain {"role", "content"} dicts
    (each assistant entry just its final text, no tool_use/tool_result
    blocks replayed) prepended before `question`. Without it, every call is
    a fresh, context-free question -- the model has no memory of anything
    asked earlier in the same session.
    """
    client = client or anthropic.Anthropic()
    tools = build_tools(db_pool, engine_pool, voyage_client, on_position=on_position)

    messages = [*(history or []), {"role": "user", "content": question}]
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=messages,
        stream=True,
    )

    final_text = ""
    last_message = None
    for turn in runner:
        for delta in turn.text_stream:
            if on_chunk is not None:
                on_chunk(delta)
        message = turn.get_final_message()
        last_message = message
        if on_step is not None:
            _report_tool_steps(message, on_step)
        if on_position is not None:
            _report_position_update(message, on_position)
        final_text = "".join(block.text for block in message.content if block.type == "text")

    if not final_text.strip() and last_message is not None:
        # Observed in practice, not just theoretically possible: the loop
        # above can end without ever producing a clean text-only synthesis
        # turn, even though tool_runner's own termination logic (read
        # directly from its source) is only supposed to exit normally once
        # generate_tool_call_response() finds no more tool calls to make --
        # every other stop condition (a refusal, hitting max_iterations,
        # which this app doesn't set) was checked and ruled out. Rather
        # than silently return nothing, force one.
        if on_step is not None:
            on_step("Recovering an incomplete response...")
        final_text = _recover_synthesis(client, runner, on_chunk)

    return final_text
