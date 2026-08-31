"""Layer 1 -- deterministic structured search over games/moves
(scripts/init_db.sql). python-chess-shaped data queried via raw SQL, no LLM
judgment involved: move/position/pattern queries and statistical aggregation,
e.g. "where does the knight usually land in this ECO code" -- the stats half
of the flagship opening-profile demo (paired with Layer 2 prose about plans).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import chess
import chess.pgn
import psycopg2.extensions

Color = Literal["white", "black", "both"]

_VALID_PIECES = {"P", "N", "B", "R", "Q", "K"}


@dataclass
class EcoSummary:
    eco_code: str
    game_count: int
    white_wins: int
    black_wins: int
    draws: int
    avg_ply_count: float | None


@dataclass
class SquareFrequency:
    square: str
    count: int


@dataclass
class MoveFrequency:
    move_san: str
    count: int


@dataclass
class NarrativeGameCandidate:
    game_id: str
    white: str | None
    black: str | None
    event: str | None
    year: int | None
    eco_code: str | None
    result: str | None
    annotation_chunk_count: int


def _piece_match_clause(piece: str, color: Color) -> tuple[str, str]:
    """Return (SQL condition on m.piece, bound value) for the requested color."""
    normalized = piece.upper()
    if normalized not in _VALID_PIECES:
        raise ValueError(f"Invalid piece {piece!r}; expected one of {sorted(_VALID_PIECES)}")
    if color == "white":
        return "m.piece = %s", normalized
    if color == "black":
        return "m.piece = %s", normalized.lower()
    if color == "both":
        return "UPPER(m.piece) = UPPER(%s)", normalized
    raise ValueError(f"Invalid color {color!r}; expected 'white', 'black', or 'both'")


def eco_summary(conn: psycopg2.extensions.connection, eco_code: str) -> EcoSummary:
    """Game count, result breakdown, and average game length for an ECO code."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                count(*) AS game_count,
                count(*) FILTER (WHERE result = '1-0') AS white_wins,
                count(*) FILTER (WHERE result = '0-1') AS black_wins,
                count(*) FILTER (WHERE result = '1/2-1/2') AS draws
            FROM games
            WHERE eco_code = %s
            """,
            (eco_code,),
        )
        game_count, white_wins, black_wins, draws = cur.fetchone()

        cur.execute(
            """
            SELECT avg(game_lengths.ply_count)
            FROM (
                SELECT m.game_id, max(m.ply) AS ply_count
                FROM moves m
                JOIN games g ON g.game_id = m.game_id
                WHERE g.eco_code = %s
                GROUP BY m.game_id
            ) AS game_lengths
            """,
            (eco_code,),
        )
        (avg_ply_count,) = cur.fetchone()

    return EcoSummary(
        eco_code=eco_code,
        game_count=game_count,
        white_wins=white_wins,
        black_wins=black_wins,
        draws=draws,
        avg_ply_count=float(avg_ply_count) if avg_ply_count is not None else None,
    )


def piece_placement_frequency(
    conn: psycopg2.extensions.connection,
    eco_code: str,
    piece: str,
    *,
    color: Color = "both",
    max_ply: int | None = 20,
    limit: int = 10,
) -> list[SquareFrequency]:
    """Most common destination squares for `piece` among games in `eco_code`.

    Restricted to the first `max_ply` half-moves (the opening phase) unless
    max_ply is None, in which case the whole game is considered.
    """
    piece_clause, piece_value = _piece_match_clause(piece, color)
    query = f"""
        SELECT m.to_sq, count(*) AS cnt
        FROM moves m
        JOIN games g ON g.game_id = m.game_id
        WHERE g.eco_code = %s
          AND {piece_clause}
          AND (%s::int IS NULL OR m.ply <= %s)
        GROUP BY m.to_sq
        ORDER BY cnt DESC, m.to_sq
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (eco_code, piece_value, max_ply, max_ply, limit))
        rows = cur.fetchall()

    return [SquareFrequency(square=square, count=count) for square, count in rows]


def common_moves_at_ply(
    conn: psycopg2.extensions.connection, eco_code: str, ply: int, *, limit: int = 5
) -> list[MoveFrequency]:
    """Most frequent SAN moves played at an exact ply among games in `eco_code`."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.move_san, count(*) AS cnt
            FROM moves m
            JOIN games g ON g.game_id = m.game_id
            WHERE g.eco_code = %s AND m.ply = %s
            GROUP BY m.move_san
            ORDER BY cnt DESC, m.move_san
            LIMIT %s
            """,
            (eco_code, ply, limit),
        )
        rows = cur.fetchall()

    return [MoveFrequency(move_san=move_san, count=count) for move_san, count in rows]


def select_narrative_game(
    conn: psycopg2.extensions.connection, eco_codes: list[str]
) -> NarrativeGameCandidate | None:
    """Pick one ChessBase-sourced game to show, moves only, for the
    "narrative" lane of the study recommendation feature -- a different
    source than the other two recommendation lanes (drill/concept), which
    draw from the Lichess-backed quality-classifier cascade. This lane
    instead draws from the corpus's own already-curated master games,
    stripped down to just the moves (see game_moves_as_pgn): a game's move
    sequence is historical fact, not the copyrightable part of a ChessBase
    export, so nothing here needs the classifier pipeline's licensing care.

    Restricted to source = 'chessbase': that's this project's only source
    of GM-level annotated games (see CLAUDE.md). Rows with source =
    'lichess' in the same table are raw volume data for Layer 1 stats, not
    necessarily GM games, and aren't what this lane is for.

    Among games matching any of eco_codes, prefers the one with the most
    game_annotation chunks originally attached: the original annotator
    judging a game worth explaining at length is a real notability signal,
    even though none of that text is ever shown here -- only the moves are.
    Returns None if no chessbase game matches any of the given codes.
    """
    if not eco_codes:
        raise ValueError("eco_codes must be non-empty")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT g.game_id, g.white, g.black, g.event, g.year, g.eco_code,
                   g.result, count(c.game_id) AS annotation_chunk_count
            FROM games g
            LEFT JOIN chunks c
              ON c.game_id = g.game_id AND c.source_type = 'game_annotation'
            WHERE g.source = 'chessbase' AND g.eco_code = ANY(%s)
            GROUP BY g.game_id, g.white, g.black, g.event, g.year, g.eco_code, g.result
            ORDER BY annotation_chunk_count DESC, g.game_id
            LIMIT 1
            """,
            (eco_codes,),
        )
        row = cur.fetchone()

    if row is None:
        return None
    game_id, white, black, event, year, eco_code, result, chunk_count = row
    return NarrativeGameCandidate(
        game_id=game_id,
        white=white,
        black=black,
        event=event,
        year=year,
        eco_code=eco_code,
        result=result,
        annotation_chunk_count=chunk_count,
    )


def game_moves_as_pgn(conn: psycopg2.extensions.connection, game_id: str) -> str | None:
    """Reconstruct a complete, valid PGN document for one game directly
    from games + moves: headers plus movetext in SAN, deliberately no
    comments or NAGs. That's true by construction, not just by omission --
    moves has no annotation column at all (annotations live only in
    chunks, a separate table this never touches), so there's nothing to
    accidentally include.

    Built via chess.pgn.Game and python-chess's own serializer rather than
    formatting header strings by hand, so header values are escaped
    correctly (e.g. a player name containing a quote character) instead of
    risking malformed PGN from a naive f-string.

    Returns None if game_id doesn't exist.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT white, black, event, year, eco_code, result FROM games WHERE game_id = %s",
            (game_id,),
        )
        header_row = cur.fetchone()
        if header_row is None:
            return None
        white, black, event, year, eco_code, result = header_row

        cur.execute(
            "SELECT move_san FROM moves WHERE game_id = %s ORDER BY ply",
            (game_id,),
        )
        move_sans = [row[0] for row in cur.fetchall()]

    game = chess.pgn.Game()
    game.headers["Event"] = event or "?"
    game.headers["White"] = white or "?"
    game.headers["Black"] = black or "?"
    game.headers["Date"] = f"{year}.??.??" if year else "????.??.??"
    game.headers["Result"] = result or "*"
    if eco_code:
        game.headers["ECO"] = eco_code

    node = game
    board = chess.Board()
    for san in move_sans:
        move = board.parse_san(san)
        node = node.add_variation(move)
        board.push(move)

    return str(game)
