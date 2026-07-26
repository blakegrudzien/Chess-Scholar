"""Layer 1 -- deterministic structured search over games/moves
(scripts/init_db.sql). python-chess-shaped data queried via raw SQL, no LLM
judgment involved: move/position/pattern queries and statistical aggregation,
e.g. "where does the knight usually land in this ECO code" -- the stats half
of the flagship opening-profile demo (paired with Layer 2 prose about plans).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
