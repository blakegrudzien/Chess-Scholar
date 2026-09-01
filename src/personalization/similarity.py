"""Layer 4 -- personalized structural similarity. A user uploads a PGN; we
find corpus games that share the longest matching opening-move prefix.

This is a deliberately simple, explainable metric (exact SAN match on a
common opening line), not a rigorous positional-similarity model -- per
CLAUDE.md, Layer 4 is approximate, and any UI surfacing these results
should call them an "illustrative comparison," not authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg2.extensions


@dataclass
class SimilarGame:
    game_id: str
    white: str | None
    black: str | None
    year: int | None
    eco_code: str | None
    result: str | None
    matching_plies: int
    fen_after: str | None


def find_similar_games(
    conn: psycopg2.extensions.connection,
    user_moves: list[str],
    max_ply: int = 20,
    limit: int = 10,
) -> list[SimilarGame]:
    """Rank corpus games by how many of the first `max_ply` half-moves they
    share with `user_moves`, counted from the start of the game (a longest
    common opening prefix, not a fuzzy or positional match).
    """
    if not user_moves:
        raise ValueError("user_moves must be non-empty")

    target = list(enumerate(user_moves[:max_ply], start=1))
    values_clause = ", ".join(["(%s, %s)"] * len(target))
    params: list[object] = [item for ply, san in target for item in (ply, san)]

    query = f"""
        WITH target(ply, move_san) AS (VALUES {values_clause}),
        matched AS (
            SELECT
                m.game_id,
                min(CASE WHEN m.move_san <> t.move_san THEN m.ply END) AS first_mismatch_ply,
                count(*) AS plies_checked
            FROM moves m
            JOIN target t ON t.ply = m.ply
            GROUP BY m.game_id
        )
        SELECT
            g.game_id, g.white, g.black, g.year, g.eco_code, g.result,
            COALESCE(matched.first_mismatch_ply - 1, matched.plies_checked) AS match_length,
            fm.fen_after
        FROM matched
        JOIN games g ON g.game_id = matched.game_id
        LEFT JOIN moves fm ON fm.game_id = matched.game_id
            AND fm.ply = COALESCE(matched.first_mismatch_ply - 1, matched.plies_checked)
        ORDER BY match_length DESC, g.game_id
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(query, [*params, limit])
        rows = cur.fetchall()

    return [
        SimilarGame(
            game_id=game_id,
            white=white,
            black=black,
            year=year,
            eco_code=eco_code,
            result=result,
            matching_plies=match_length,
            fen_after=fen_after,
        )
        for game_id, white, black, year, eco_code, result, match_length, fen_after in rows
    ]
