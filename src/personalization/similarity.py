"""Layer 4: compares an uploaded game with the corpus by the longest run of
identical opening moves.

A simple, explainable measure rather than positional similarity, so results
are presented as an illustrative comparison, not an authoritative one.
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
    share with `user_moves` from the start (the longest common opening
    prefix). Games sharing no moves are left out.
    """
    if not user_moves:
        raise ValueError("user_moves must be non-empty")

    target = list(enumerate(user_moves[:max_ply], start=1))
    # values_clause is only repeated "(%s, %s)" placeholders; the user's
    # moves are bound as parameters. Built by hand because psycopg2 has no
    # placeholder for a VALUES list of variable length.
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
        -- A game that differs from the first move shares nothing to compare.
        WHERE COALESCE(matched.first_mismatch_ply - 1, matched.plies_checked) > 0
        ORDER BY match_length DESC, g.game_id
        LIMIT %s
    """  # noqa: S608 (values_clause holds only placeholders, see above)

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
