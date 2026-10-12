"""The chat agent: Claude with native tool calling across the four layers
(structured search, vector RAG, Stockfish, personalized similarity). The
model decides which tools to call; there is no separate intent classifier.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import anthropic
import chess
import psycopg2.pool
import voyageai
from anthropic import beta_tool

from src.config import CLAUDE_MODEL
from src.engine.engine_pool import EngineBusyError, EnginePool
from src.engine.stockfish_eval import DEFAULT_DEPTH, PositionEval, evaluate_position
from src.ingestion.db_loader import query_with_retry
from src.personalization.similarity import find_similar_games as _find_similar_games
from src.rag.vector_search import search_chunks
from src.search.structured_search import common_moves_at_ply, eco_summary, piece_placement_frequency

logger = logging.getLogger(__name__)


class OnPosition(Protocol):
    """The signature of every on_position callback. A Protocol rather than
    Callable[..., None] so the type checker catches a misspelled keyword."""

    def __call__(self, fen: str, *, label: str | None = None) -> None: ...


# A turn with several tool calls can run out of output tokens mid-call,
# leaving a tool_use block with incomplete input (see
# _report_position_update and _recover_synthesis). 8192 makes that rare.
MAX_TOKENS = 8192

# Caps tool-calling turns per question. Every turn is a paid API call on a
# public app, and the UI's rate limit counts questions, not turns.
MAX_AGENT_TURNS = 10

# Caps on numbers the model fills in itself: a huge `limit` would flood the
# context with results, and a long candidate list would tie up the small
# engine pool. Generous enough that real questions never hit them.
MAX_SEARCH_LIMIT = 20
MAX_CANDIDATE_MOVES = 8
MAX_PLY_WINDOW = 60

SYSTEM_PROMPT = """You are a chess research assistant/mentor with access to a corpus \
of grandmaster games, book/annotation text, and a chess engine, via tools.

Tool selection:
- get_eco_summary / get_piece_placement / get_common_moves_at_ply: \
deterministic statistics from the game database (Layer 1). Use these for \
"how often", "where does X usually go", "what's the most common move" questions.
- search_annotations: semantic search over human-written commentary and book \
text (Layer 2). Use this for "why", "what's the plan", and conceptual/strategic \
questions. This returns a synthesis of retrieved human text, not your own \
independent judgment. Attribute ideas to the retrieved material.
- evaluate_chess_position: Stockfish ground truth (Layer 3) for a single \
position. You MUST call this (or compare_candidate_moves) before making \
any claim about whether a move or position is good, winning, a mistake, \
or a blunder. Never judge tactical soundness from your own knowledge \
alone; you are not a substitute for engine analysis.
- compare_candidate_moves: Stockfish ground truth (Layer 3) for several \
candidate moves from the same position at once. Use this instead of \
calling evaluate_chess_position once per candidate whenever a question \
asks you to compare, rank, or choose among multiple replies (e.g. "how \
should Black meet this", "which of these responses is best"). It runs \
the candidates concurrently, so one call here is much faster than several \
separate evaluate_chess_position calls for the same comparison.
- find_similar_corpus_games: opening-move-prefix matching against a user's \
own game (Layer 4). This is an approximate, illustrative comparison based on \
exact opening moves, not a rigorous positional match. Always tell the user \
this is illustrative, not authoritative, when you use it.
- show_opening_line: renders a labeled board diagram for a specific move \
sequence. For opening-theory questions (ECO stats, common-move, or \
strategic-plan answers), proactively call this for the main line and any \
named sidelines you discuss, instead of only describing moves in prose. \
Readers should be able to see the position, not just read algebraic \
notation. Scoped to opening theory, not a substitute for \
evaluate_chess_position when judging whether a move or plan is good. In \
your final answer, mark exactly where each diagram belongs by writing \
[[diagram: <label>]] on its own line at that point, using the exact same \
label text you passed to show_opening_line, e.g. [[diagram: Main line]]. \
A diagram is never visible before your final answer: it renders inside that \
answer, at the marker, or at the end of it if you leave the marker out. \
Never refer to a diagram as one the reader has already seen.

Combine tools when a question calls for it (e.g. stats + commentary for an \
opening-profile question). Be direct about which tool(s) you used.

For broad "walk me through" or "what are the options against X" questions, \
prefer one well-chosen tool call per layer over several near-duplicate \
calls (e.g. one get_eco_summary for the most relevant ECO code, not one \
per code in the family). Every tool call is a real network round trip, \
and a focused, useful answer that arrives promptly beats an exhaustive one \
that takes minutes. Two or three show_opening_line diagrams (a main line \
plus the most relevant sidelines) is plenty; skip minor branches.

Before calling a tool, state in one short sentence which layer you're using \
and why, e.g. "Checking Layer 2 for strategic ideas about isolated pawns." \
Keep it to that one sentence. It is a progress label, not part of your \
answer, and it is discarded once the tool call finishes.

Your last turn (the one that calls no tools) is the entire answer, and \
the only thing the reader is left with. Nothing you wrote in an earlier turn \
survives, and the reader never sees any of it. So write that last turn as a \
complete, standalone answer to the question, as though you had said nothing \
before it: no "bottom line" or "in short" wrap-up of reasoning they cannot \
see, and no referring back to an earlier turn. Every statistic, evaluation \
and line that matters has to appear there in full, stated fresh.

A user message may start with a line like "Current board position: <FEN>"; \
this means they set up that position on their own board and are asking about \
it specifically. Treat it as the position in question, not something to \
independently derive or guess at.
"""

# Status text per tool, used when the model calls a tool without first
# writing its one-sentence status (see _report_tool_steps).
TOOL_LABELS: dict[str, str] = {
    "get_eco_summary": "Layer 1 (structured search): pulling opening statistics.",
    "get_piece_placement": "Layer 1 (structured search): checking piece placement frequency.",
    "get_common_moves_at_ply": "Layer 1 (structured search): checking common moves.",
    "search_annotations": "Layer 2 (semantic search): searching commentary and book text.",
    "evaluate_chess_position": "Layer 3 (Stockfish): evaluating the position.",
    "compare_candidate_moves": "Layer 3 (Stockfish): comparing candidate moves.",
    "find_similar_corpus_games": "Layer 4 (similarity search): comparing against the corpus.",
    "show_opening_line": "Rendering a position diagram.",
}


def _format_position_eval(result: PositionEval) -> str:
    """Shared by evaluate_chess_position and compare_candidate_moves, so a
    single-position eval and one candidate's line in a comparison read
    identically.
    """
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


def build_tools(
    db_pool: psycopg2.pool.ThreadedConnectionPool,
    engine_pool: EnginePool,
    voyage_client: voyageai.Client,
    on_position: OnPosition | None = None,
) -> list[Callable]:
    """Build the tool functions for one session, bound to the given DB pool,
    Stockfish engine pool, and Voyage client.

    Each tool call checks a connection out of `db_pool` and returns it
    afterward rather than holding one for the whole session, so concurrent
    sessions don't share a connection; query_with_retry (db_loader.py)
    handles a dead connection transparently within the call.

    on_position, if given, is called with the FEN behind a position-touching
    tool call: (fen, *, label=None), so the caller can show it as a diagram.
    Most calls come from _report_position_update scanning
    evaluate_chess_position's tool_use block; find_similar_corpus_games
    calls it directly instead, since its FEN only exists in the tool's
    *output*, not an argument the model supplies. show_opening_line passes
    the label it was given, which the answer uses to place its diagram.
    """

    def _query(fn: Callable, *args, **kwargs):
        return query_with_retry(db_pool, fn, *args, **kwargs)

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
                piece_placement_frequency,
                eco_code,
                piece,
                color=color,
                max_ply=min(max_ply, MAX_PLY_WINDOW),
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
        results = _query(common_moves_at_ply, eco_code, ply, limit=min(limit, MAX_SEARCH_LIMIT))
        if not results:
            return f"No data at ply {ply} for ECO {eco_code}."
        return "; ".join(f"{r.move_san} ({r.count}x)" for r in results)

    @beta_tool
    def search_annotations(query: str, limit: int = 5) -> str:
        """Semantic search over human-written game annotations and book
        commentary for conceptual/strategic ideas, plans, and explanations.

        Each result is prefixed with its cosine distance from the query
        (0.0 = identical meaning, higher = less related). This search always
        returns the closest chunks it has, even when nothing in the corpus
        is genuinely on topic. Treat a high distance as weak evidence and
        say so rather than presenting it as an authoritative source.

        Args:
            query: Natural-language description of the idea or concept to search for.
            limit: Max number of results. Defaults to 5.
        """
        results = _query(search_chunks, voyage_client, query, limit=min(limit, MAX_SEARCH_LIMIT))
        if not results:
            return "No relevant annotations found."
        return "\n".join(f"- [distance {r.distance:.2f}] {r.text}" for r in results)

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
        return _format_position_eval(result)

    @beta_tool
    def compare_candidate_moves(fen: str, moves: list[str], depth: int = DEFAULT_DEPTH) -> str:
        """Evaluate several candidate moves from the same starting position
        at once, for "which of these is best" questions, e.g. comparing
        ways to meet an opening try. Prefer this over calling
        evaluate_chess_position once per candidate whenever you're
        comparing multiple moves from the same position: each candidate's
        search runs on its own engine from the pool at the same time,
        instead of one after another, so comparing several moves costs
        close to the same wall-clock time as evaluating just one.

        Args:
            fen: The starting position in FEN notation, before any candidate move.
            moves: Candidate moves in SAN notation to compare, e.g. ["g6", "Qe7", "Qf6"].
            depth: Search depth per candidate; higher is more accurate but slower. Defaults to 18.
        """
        try:
            base_board = chess.Board(fen)
        except ValueError as exc:
            return f"Invalid FEN: {exc}"

        # Each candidate ties up an engine shared by every visitor, so long
        # lists are truncated (and the model told) rather than rejected.
        dropped = max(0, len(moves) - MAX_CANDIDATE_MOVES)
        moves = moves[:MAX_CANDIDATE_MOVES]

        def _evaluate_one(move_san: str) -> str:
            candidate_board = base_board.copy()
            try:
                candidate_board.push_san(move_san)
            except ValueError:
                return f"{move_san}: not legal in that position."
            try:
                with engine_pool.checkout() as engine:
                    result = evaluate_position(engine, candidate_board, depth=depth)
            except EngineBusyError as exc:
                return f"{move_san}: {exc}"
            return f"{move_san}: {_format_position_eval(result)}"

        # One worker per engine: checkout() never blocks, so extra workers
        # would fail with EngineBusyError instead of waiting their turn.
        with ThreadPoolExecutor(max_workers=engine_pool.size) as executor:
            lines = list(executor.map(_evaluate_one, moves))
        if dropped:
            lines.append(
                f"({dropped} further candidate(s) not evaluated: this tool compares at "
                f"most {MAX_CANDIDATE_MOVES} moves per call. Call it again for the rest "
                f"if they still matter.)"
            )
        return "\n".join(lines)

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
            results = _query(
                _find_similar_games,
                moves,
                max_ply=min(max_ply, MAX_PLY_WINDOW),
                limit=min(limit, MAX_SEARCH_LIMIT),
            )
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
        opening moves, e.g. the main line of a variation, or a named
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
                return f"'{san}' (move {i}) isn't legal in that sequence; no diagram prepared."
        if on_position is not None:
            on_position(board.fen(), label=label)
        # Says the diagram isn't visible yet: told it was "shown", the model
        # writes as if the reader already sees it, though nothing renders
        # until its [[diagram]] marker in the final answer.
        return (
            f"Diagram prepared for '{label}' ({' '.join(moves)}). The reader cannot see it "
            f"yet. It appears only inside the answer you write, at the point where you put "
            f"[[diagram: {label}]], so describe the position in full there rather than "
            f"referring to it as something already shown."
        )

    return [
        get_eco_summary,
        get_piece_placement,
        get_common_moves_at_ply,
        search_annotations,
        evaluate_chess_position,
        compare_candidate_moves,
        find_similar_corpus_games,
        show_opening_line,
    ]


def _report_tool_steps(message, on_step: Callable[[str], None]) -> None:
    """Report one status line for a tool-calling turn: the model's own
    one-sentence rationale, or TOOL_LABELS if it wrote none."""
    tool_names = [block.name for block in message.content if block.type == "tool_use"]
    if not tool_names:
        return
    rationale = "".join(block.text for block in message.content if block.type == "text").strip()
    if len(rationale) > 400:
        # Only the final turn's text is kept as the answer, so analysis the
        # model puts in a long rationale is lost. Logged to track how often
        # that happens despite the prompt.
        logger.warning(
            "Unusually long tool-call rationale (%d chars), likely discarded content: %r",
            len(rationale),
            rationale[:200],
        )
    if not rationale:
        rationale = " / ".join(TOOL_LABELS.get(name, f"Calling {name}...") for name in tool_names)
    on_step(rationale)


def _report_position_update(message, on_position: OnPosition) -> None:
    """Report the FEN of a turn's evaluate_chess_position call, read from
    the tool_use block itself (the other position tools report their own).

    Uses .get(): a turn cut off mid-generation (see MAX_TOKENS) can leave a
    tool_use block with no "fen", and skipping the report is the right
    fallback.
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
    """Get a real answer when the tool-calling loop ends without one, by
    continuing the runner's conversation with an explicit request for it.

    Reads runner._params["messages"], a private SDK attribute: the public
    API can't return the accumulated messages, and the SDK's own
    compaction_control reads the same attribute. Check it on SDK upgrades.
    """
    messages = list(runner._params["messages"])
    # The API requires a tool_result right after any tool_use, so an
    # unanswered tool_use at the end is dropped before the new user turn.
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
        model=CLAUDE_MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, messages=messages
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
    on_position: OnPosition | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Answer one question, letting the model route across the four layers
    with tool calls. Returns the final answer text.

    on_step(text): called once per tool-calling turn with a short status
    line.

    on_chunk(delta): called with every streamed text delta, from every turn.
    A turn's text is only known to be a status line or the answer once the
    turn ends; if on_step fires for a turn, its chunks were the status line.

    on_position(fen, label=None): called with the positions tools touch, so
    the caller can show them as diagrams.

    history: earlier turns as {"role", "content"} dicts (final text only),
    sent before `question` so follow-up questions have context.
    """
    client = client or anthropic.Anthropic()
    tools = build_tools(db_pool, engine_pool, voyage_client, on_position=on_position)

    messages = [*(history or []), {"role": "user", "content": question}]
    runner = client.beta.messages.tool_runner(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=messages,
        stream=True,
        max_iterations=MAX_AGENT_TURNS,
    )

    final_text = ""
    last_message = None
    turn_count = 0
    ask_start = time.monotonic()
    for turn in runner:
        turn_start = time.monotonic()
        for delta in turn.text_stream:
            if on_chunk is not None:
                on_chunk(delta)
        message = turn.get_final_message()
        last_message = message
        turn_count += 1
        tool_names = [block.name for block in message.content if block.type == "tool_use"]
        # Per-turn timing, so slowness can be traced to extra turns or to a
        # slow turn.
        logger.info(
            "turn %d: %.2fs, tools=%s", turn_count, time.monotonic() - turn_start, tool_names
        )
        if on_step is not None:
            _report_tool_steps(message, on_step)
        if on_position is not None:
            _report_position_update(message, on_position)
        final_text = "".join(block.text for block in message.content if block.type == "text")

    # If the loop ended on a tool-calling turn, final_text is that turn's
    # status line, not an answer. That happens when MAX_AGENT_TURNS is hit
    # (tool_runner just stops, with no error) and, occasionally, earlier for
    # reasons tool_runner doesn't report. Checking for tool calls, not just
    # empty text, tells a short answer from an interrupted one.
    interrupted_mid_work = last_message is not None and any(
        block.type == "tool_use" for block in last_message.content
    )
    logger.info(
        "ask() finished: %d turn(s), %.2fs total, final_text=%d chars, interrupted=%s",
        turn_count,
        time.monotonic() - ask_start,
        len(final_text),
        interrupted_mid_work,
    )

    if last_message is not None and (not final_text.strip() or interrupted_mid_work):
        if interrupted_mid_work:
            logger.warning(
                "Loop ended on a tool-calling turn after %d turn(s) (ceiling is %d); "
                "forcing a synthesis rather than returning that turn's rationale.",
                turn_count,
                MAX_AGENT_TURNS,
            )
        if on_step is not None:
            on_step("Recovering an incomplete response...")
        final_text = _recover_synthesis(client, runner, on_chunk)

    return final_text
