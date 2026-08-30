"""Extract prose annotations (move comments + NAGs) from ChessBase PGN exports
into chunk-ready records matching the chunks table in scripts/init_db.sql.

See pgn_encoding.py for how Windows-1252/UTF-8 encoding issues in ChessBase
exports are handled. NAG glyphs (numeric move-quality codes like NAG 1 = "!")
are mapped to their standard symbols and folded into the comment prose, since
raw NAG numbers aren't useful embedding text. PGN computer-annotation glyphs
embedded in comments (e.g. "[%evp 0,79,...]" eval curves, "[%clk 0:05:23]"
clock times, "[%cal ...]"/"[%csl ...]" GUI arrows/highlights) are stripped
for the same reason.

game_id is computed the same way as pgn_parser.compute_game_id (headers + SAN
move sequence) so annotation chunks extracted here join back to the games
inserted by db_loader via the same key.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from src.ingestion.pgn_encoding import PGN_DECODE_ERRORS
from src.ingestion.pgn_parser import compute_game_id, parse_year

# Standard PGN Numeric Annotation Glyphs for move quality / evaluation.
# Subset covering what ChessBase annotation exports commonly emit.
NAG_SYMBOLS: dict[int, str] = {
    1: "!",
    2: "?",
    3: "!!",
    4: "??",
    5: "!?",
    6: "?!",
    10: "=",
    13: "∞",
    14: "⩲",
    15: "⩱",
    16: "±",
    17: "∓",
    18: "+-",
    19: "-+",
}


@dataclass
class AnnotationChunk:
    source_type: str
    game_id: str | None
    source_title: str | None
    author: str | None
    year: int | None
    eco_code: str | None
    ply_or_page: str | None
    text: str


def _nags_to_prose(nags: set[int]) -> str:
    return "".join(NAG_SYMBOLS[nag] for nag in sorted(nags) if nag in NAG_SYMBOLS)


_PGN_GLYPH_RE = re.compile(r"\[%[^\]]*\]")


def _strip_computer_glyphs(comment: str) -> str:
    """Remove GUI-only annotation glyphs (eval curves, clocks, arrows, square
    highlights); they aren't meaningful text for RAG embedding.
    """
    without_glyphs = _PGN_GLYPH_RE.sub("", comment)
    return re.sub(r"\s+", " ", without_glyphs).strip()


def _iter_plies_with_comments(
    game: chess.pgn.Game,
) -> Iterator[tuple[int, str, str, set[int]]]:
    """Walk one parsed chapter/game's mainline, yielding
    (ply, move_san, glyph_stripped_comment, nags) for every move -- comment
    is an empty string for moves with no annotation, not omitted, so
    callers can tell "no comment" from "ply doesn't exist" and still see
    every move if they need to (compute_game_id needs the full move
    sequence, not just the annotated plies).

    SAN is computed before each push, matching pgn_parser.parse_game,
    since python-chess needs the pre-move board state to disambiguate
    algebraic notation. Shared by _extract_game_annotations (structures
    this into chunks-table-shaped records) and extract_chapter_comment_text
    (joins it into one readable preview block) so the walk itself is
    written once.
    """
    board = game.board()
    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        move_san = board.san(move)
        board.push(move)
        yield ply, move_san, _strip_computer_glyphs(node.comment), node.nags


def _move_label(move_san: str, nags: set[int]) -> str:
    nag_prose = _nags_to_prose(nags)
    return f"{move_san}{nag_prose}" if nag_prose else move_san


def extract_chapter_comment_text(game: chess.pgn.Game) -> str:
    """Concatenate every prose comment in one parsed chapter/game into a
    single readable block: the pre-game comment first (if any), then one
    line per annotated move as "SAN<nags>: comment".

    For the study labeling tool (scripts/label_studies_app.py), which needs
    a quick human-readable preview of a Lichess study chapter's annotations
    to judge quality without leaving the app -- not the chunks-table-shaped
    output extract_annotations produces, since a preview reader wants the
    prose, not a game_id to join chunks back to a games row.
    """
    lines = []
    pre_game_comment = _strip_computer_glyphs(game.comment)
    if pre_game_comment:
        lines.append(pre_game_comment)

    for _, move_san, comment, nags in _iter_plies_with_comments(game):
        if not comment:
            continue
        lines.append(f"{_move_label(move_san, nags)}: {comment}")

    return "\n".join(lines)


def _extract_game_annotations(
    game: chess.pgn.Game, source_title: str | None, author: str | None
) -> Iterator[AnnotationChunk]:
    headers = game.headers
    year = parse_year(headers.get("Date"))
    eco_code = headers.get("ECO")

    ply_annotations = list(_iter_plies_with_comments(game))
    move_sans = [move_san for _, move_san, _, _ in ply_annotations]
    game_id = compute_game_id(headers, move_sans)

    pre_game_comment = _strip_computer_glyphs(game.comment)
    if pre_game_comment:
        yield AnnotationChunk(
            source_type="game_annotation",
            game_id=game_id,
            source_title=source_title,
            author=author,
            year=year,
            eco_code=eco_code,
            ply_or_page="0",
            text=pre_game_comment,
        )

    for ply, move_san, comment, nags in ply_annotations:
        if not comment:
            continue
        move_label = _move_label(move_san, nags)
        yield AnnotationChunk(
            source_type="game_annotation",
            game_id=game_id,
            source_title=source_title,
            author=author,
            year=year,
            eco_code=eco_code,
            ply_or_page=str(ply),
            text=f"{move_label}: {comment}",
        )


def extract_annotations(
    path: str | Path, source_title: str | None = None, author: str | None = None
) -> Iterator[AnnotationChunk]:
    """Yield one AnnotationChunk per commented move (plus a pre-game chunk if
    the PGN has a starting comment) across all games in the file at `path`.
    """
    with open(path, encoding="utf-8", errors=PGN_DECODE_ERRORS) as pgn_stream:
        while True:
            game = chess.pgn.read_game(pgn_stream)
            if game is None:
                break
            yield from _extract_game_annotations(game, source_title, author)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract annotation chunks from a ChessBase PGN export."
    )
    parser.add_argument("--input", required=True, help="Path to a PGN file")
    parser.add_argument("--source-title", default=None, help="Value for chunks.source_title")
    parser.add_argument("--author", default=None, help="Value for chunks.author")
    args = parser.parse_args()

    chunk_count = 0
    for _ in extract_annotations(args.input, source_title=args.source_title, author=args.author):
        chunk_count += 1

    print(f"Extracted {chunk_count} annotation chunk(s) from {args.input}")


if __name__ == "__main__":
    _main()
