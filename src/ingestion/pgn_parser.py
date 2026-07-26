"""Parse PGN files into records matching the games/moves tables in scripts/init_db.sql."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn

from src.ingestion.pgn_encoding import PGN_DECODE_ERRORS

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass
class MoveRecord:
    ply: int
    move_san: str
    move_uci: str
    from_sq: str
    to_sq: str
    piece: str
    is_capture: bool
    captured_piece: str | None
    material_delta: int
    fen_after: str


@dataclass
class GameRecord:
    game_id: str
    white: str | None
    black: str | None
    event: str | None
    year: int | None
    eco_code: str | None
    result: str | None
    source: str
    moves: list[MoveRecord] = field(default_factory=list)


def parse_year(date_header: str | None) -> int | None:
    if not date_header:
        return None
    year_str = date_header[:4]
    return int(year_str) if year_str.isdigit() else None


def compute_game_id(headers: chess.pgn.Headers, move_sans: list[str]) -> str:
    """Deterministic id derived from headers + SAN move sequence.

    Used as the natural key for idempotent ON CONFLICT DO NOTHING inserts in
    db_loader, and shared with annotation_extractor.py so annotation chunks
    extracted from the same game agree on game_id and can join back to it.
    """
    move_text = " ".join(move_sans)
    canonical = "|".join(
        [
            headers.get("White", ""),
            headers.get("Black", ""),
            headers.get("Event", ""),
            headers.get("Date", ""),
            headers.get("Round", ""),
            headers.get("Result", ""),
            move_text,
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _captured_piece_symbol(board: chess.Board, move: chess.Move) -> str | None:
    """Must be called before board.push(move)."""
    if not board.is_capture(move):
        return None
    if board.is_en_passant(move):
        return chess.piece_symbol(chess.PAWN)
    captured = board.piece_at(move.to_square)
    return captured.symbol() if captured else None


def _material_delta(board: chess.Board, move: chess.Move) -> int:
    """Value of the piece captured by this move, if any. Must be called before push."""
    if not board.is_capture(move):
        return 0
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    return PIECE_VALUES[captured.piece_type] if captured else 0


def parse_game(game: chess.pgn.Game, source: str) -> GameRecord:
    headers = game.headers
    moves: list[MoveRecord] = []

    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        # SAN, capture info, and material delta all depend on the position
        # *before* the move is made, so they must be computed before push().
        move_san = board.san(move)
        piece = board.piece_at(move.from_square)
        move_record = MoveRecord(
            ply=ply,
            move_san=move_san,
            move_uci=move.uci(),
            from_sq=chess.square_name(move.from_square),
            to_sq=chess.square_name(move.to_square),
            piece=piece.symbol() if piece else "",
            is_capture=board.is_capture(move),
            captured_piece=_captured_piece_symbol(board, move),
            material_delta=_material_delta(board, move),
            fen_after="",
        )
        board.push(move)
        move_record.fen_after = board.fen()
        moves.append(move_record)

    return GameRecord(
        game_id=compute_game_id(headers, [move.move_san for move in moves]),
        white=headers.get("White"),
        black=headers.get("Black"),
        event=headers.get("Event"),
        year=parse_year(headers.get("Date")),
        eco_code=headers.get("ECO"),
        result=headers.get("Result"),
        source=source,
        moves=moves,
    )


def parse_pgn(path: str | Path, source: str) -> Iterator[GameRecord]:
    """Yield one GameRecord per game found in the PGN file at `path`."""
    with open(path, encoding="utf-8", errors=PGN_DECODE_ERRORS) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            yield parse_game(game, source)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PGN file and report a summary.")
    parser.add_argument("--input", required=True, help="Path to a PGN file")
    parser.add_argument("--source", default="lichess", help="Value for the games.source column")
    args = parser.parse_args()

    game_count = 0
    move_count = 0
    for game in parse_pgn(args.input, source=args.source):
        game_count += 1
        move_count += len(game.moves)

    print(f"Parsed {game_count} game(s), {move_count} ply total from {args.input}")


if __name__ == "__main__":
    _main()
