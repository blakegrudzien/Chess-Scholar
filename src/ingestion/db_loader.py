"""Load parsed PGN records (src/ingestion/pgn_parser.py) into the games/moves
tables defined in scripts/init_db.sql.

Inserts are idempotent: game_id is a deterministic content hash (see
pgn_parser._game_id), so re-running ingestion on the same PGN file hits
ON CONFLICT DO NOTHING instead of creating duplicate rows.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable

import psycopg2
import psycopg2.extensions
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from src.ingestion.pgn_parser import GameRecord, parse_pgn

GAMES_INSERT = """
    INSERT INTO games (game_id, white, black, event, year, eco_code, result, source)
    VALUES %s
    ON CONFLICT (game_id) DO NOTHING
"""

MOVES_INSERT = """
    INSERT INTO moves (
        game_id, ply, move_san, move_uci, from_sq, to_sq,
        piece, is_capture, captured_piece, material_delta, fen_after
    )
    VALUES %s
    ON CONFLICT (game_id, ply) DO NOTHING
"""


def get_connection() -> psycopg2.extensions.connection:
    load_dotenv()
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url)


def load_games(
    games: Iterable[GameRecord], conn: psycopg2.extensions.connection
) -> tuple[int, int]:
    """Insert games and their moves. Returns (games_inserted, moves_inserted),
    counting only rows that weren't already present.
    """
    game_rows = []
    move_rows = []
    for game in games:
        game_rows.append(
            (
                game.game_id,
                game.white,
                game.black,
                game.event,
                game.year,
                game.eco_code,
                game.result,
                game.source,
            )
        )
        move_rows.extend(
            (
                game.game_id,
                move.ply,
                move.move_san,
                move.move_uci,
                move.from_sq,
                move.to_sq,
                move.piece,
                move.is_capture,
                move.captured_piece,
                move.material_delta,
                move.fen_after,
            )
            for move in game.moves
        )

    games_inserted = 0
    moves_inserted = 0
    with conn.cursor() as cur:
        if game_rows:
            execute_values(cur, GAMES_INSERT, game_rows)
            games_inserted = cur.rowcount
        if move_rows:
            execute_values(cur, MOVES_INSERT, move_rows)
            moves_inserted = cur.rowcount

    conn.commit()
    return games_inserted, moves_inserted


def _main() -> None:
    parser = argparse.ArgumentParser(description="Load a PGN file into the games/moves tables.")
    parser.add_argument("--input", required=True, help="Path to a PGN file")
    parser.add_argument("--source", default="lichess", help="Value for the games.source column")
    args = parser.parse_args()

    games = parse_pgn(args.input, source=args.source)
    conn = get_connection()
    try:
        games_inserted, moves_inserted = load_games(games, conn)
    finally:
        conn.close()

    print(f"Inserted {games_inserted} new game(s), {moves_inserted} new move(s) from {args.input}")


if __name__ == "__main__":
    _main()
