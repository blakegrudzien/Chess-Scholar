"""Agent orchestration: Claude Sonnet 5 native tool-calling across the four
layers (structured search, vector RAG, Stockfish, personalized similarity).
Per CLAUDE.md, routing is done by the model via tool-calling, not a
hand-rolled intent classifier.
"""

from __future__ import annotations

from collections.abc import Callable

import anthropic
import chess
import chess.engine
import psycopg2.extensions
import voyageai
from anthropic import beta_tool

from src.engine.stockfish_eval import evaluate_position
from src.personalization.similarity import find_similar_games as _find_similar_games
from src.rag.vector_search import search_chunks
from src.search.structured_search import common_moves_at_ply, eco_summary, piece_placement_frequency

MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are a chess research assistant with access to a corpus \
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

Combine tools when a question calls for it (e.g. stats + commentary for an \
opening-profile question). Be direct about which tool(s) you used.
"""


DB_RETRYABLE_ERRORS = (psycopg2.OperationalError, psycopg2.InterfaceError)


def build_tools(
    conn: psycopg2.extensions.connection,
    reconnect: Callable[[], psycopg2.extensions.connection],
    engine: chess.engine.SimpleEngine,
    voyage_client: voyageai.Client,
) -> list[Callable]:
    """Build the tool functions for one session, bound to the given DB
    connection, Stockfish engine, and Voyage client.

    `reconnect` must return a fresh, live connection -- called if `conn` turns
    out to be dead (Neon closes idle connections in practice). This has to be
    handled *inside* each tool, not by the caller of `ask()`: the Anthropic
    SDK's tool_runner catches every exception a tool raises and turns it into
    a tool_result error sent back to the model, so a raised psycopg2 error
    never reaches the caller of `ask()` to trigger a reconnect there.
    """
    conn_box = [conn]

    def _query(fn: Callable, *args, **kwargs):
        try:
            return fn(conn_box[0], *args, **kwargs)
        except DB_RETRYABLE_ERRORS:
            conn_box[0] = reconnect()
            return fn(conn_box[0], *args, **kwargs)

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
    def evaluate_chess_position(fen: str, depth: int = 16) -> str:
        """Get Stockfish's ground-truth evaluation of a chess position.
        Always call this before judging whether a move or position is good.

        Args:
            fen: The position in FEN notation.
            depth: Search depth; higher is more accurate but slower. Defaults to 16.
        """
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            return f"Invalid FEN: {exc}"
        result = evaluate_position(engine, board, depth=depth)
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
        return "\n".join(
            f"{r.matching_plies} matching plies: {r.white} vs {r.black} "
            f"({r.year}, {r.eco_code}, {r.result})"
            for r in results
        )

    return [
        get_eco_summary,
        get_piece_placement,
        get_common_moves_at_ply,
        search_annotations,
        evaluate_chess_position,
        find_similar_corpus_games,
    ]


def ask(
    question: str,
    conn: psycopg2.extensions.connection,
    reconnect: Callable[[], psycopg2.extensions.connection],
    engine: chess.engine.SimpleEngine,
    voyage_client: voyageai.Client,
    client: anthropic.Anthropic | None = None,
) -> str:
    """Answer one question, routing across the four layers via tool-calling.
    Returns the final response text.
    """
    client = client or anthropic.Anthropic()
    tools = build_tools(conn, reconnect, engine, voyage_client)

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=[{"role": "user", "content": question}],
    )

    last_message = None
    for message in runner:
        last_message = message

    if last_message is None:
        return ""
    return "".join(block.text for block in last_message.content if block.type == "text")
