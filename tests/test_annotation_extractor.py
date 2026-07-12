import pytest

from src.ingestion.annotation_extractor import extract_annotations
from src.ingestion.pgn_parser import parse_pgn

SAMPLE_PGN = (
    '[Event "Test Event"]\n'
    '[Site "?"]\n'
    '[Date "1995.03.10"]\n'
    '[Round "1"]\n'
    '[White "Alice"]\n'
    '[Black "Bob"]\n'
    '[Result "1-0"]\n'
    '[ECO "C51"]\n'
    "\n"
    "{Pre-game notes: an Evans Gambit demo.} 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 "
    "4. b4 Bxb4!? {Accepting the gambit.} 5. c3 Ba5 6. d4 exd4 7. O-O 1-0\n"
)


@pytest.fixture
def pgn_file(tmp_path):
    path = tmp_path / "annotated.pgn"
    path.write_text(SAMPLE_PGN, encoding="utf-8")
    return path


def test_extracts_one_chunk_per_comment(pgn_file):
    chunks = list(extract_annotations(pgn_file))
    assert len(chunks) == 2

    pre_game, move_chunk = chunks
    assert pre_game.ply_or_page == "0"
    assert pre_game.text == "Pre-game notes: an Evans Gambit demo."

    assert move_chunk.ply_or_page == "8"  # 4...Bxb4 is the 8th half-move
    assert move_chunk.text == "Bxb4!?: Accepting the gambit."


def test_chunk_metadata_matches_game_headers(pgn_file):
    chunks = list(extract_annotations(pgn_file, source_title="My Book", author="Jane Doe"))
    for chunk in chunks:
        assert chunk.source_type == "game_annotation"
        assert chunk.year == 1995
        assert chunk.eco_code == "C51"
        assert chunk.source_title == "My Book"
        assert chunk.author == "Jane Doe"


def test_moves_without_comments_are_skipped(pgn_file):
    chunks = list(extract_annotations(pgn_file))
    ply_labels = {chunk.ply_or_page for chunk in chunks}
    assert "1" not in ply_labels  # 1. e4 has no comment or NAG


def test_game_id_matches_pgn_parser_for_same_game(pgn_file):
    # chunks.game_id must join back to games.game_id inserted by db_loader,
    # so both extractors need to agree on the id for the same underlying game.
    expected_game_id = next(parse_pgn(pgn_file, source="lichess")).game_id
    chunks = list(extract_annotations(pgn_file))
    assert all(chunk.game_id == expected_game_id for chunk in chunks)


def test_glyph_only_comment_produces_no_chunk(tmp_path):
    # PGN comments containing only a GUI directive (eval curve, clock, arrow,
    # square highlight) aren't meaningful prose and shouldn't become a chunk.
    text = SAMPLE_PGN.replace(
        "{Pre-game notes: an Evans Gambit demo.}", "{[%evp 0,13,30,17,14,-9,25,35,38,41,11,14,5]}"
    )
    path = tmp_path / "glyph_only.pgn"
    path.write_text(text, encoding="utf-8")

    chunks = list(extract_annotations(path))
    assert len(chunks) == 1  # only the Bxb4 comment survives; pre-game chunk is dropped
    assert chunks[0].ply_or_page == "8"


def test_glyph_mixed_with_prose_keeps_prose_only(tmp_path):
    text = SAMPLE_PGN.replace(
        "Accepting the gambit.", "Accepting the gambit. [%cal Ge4e5] Sharp play follows."
    )
    path = tmp_path / "glyph_mixed.pgn"
    path.write_text(text, encoding="utf-8")

    chunks = list(extract_annotations(path))
    move_chunk = chunks[1]
    assert "%cal" not in move_chunk.text
    assert move_chunk.text == "Bxb4!?: Accepting the gambit. Sharp play follows."


def test_windows_1252_fallback_decoding(tmp_path):
    # ChessBase exports are sometimes Windows-1252 rather than UTF-8;
    # an en dash (U+2013) is a byte that's invalid as standalone UTF-8
    # but valid in cp1252, so this exercises the fallback path.
    text = SAMPLE_PGN.replace("Accepting the gambit.", "Accepting the gambit – a sharp try.")
    path = tmp_path / "cp1252.pgn"
    path.write_bytes(text.encode("cp1252"))

    chunks = list(extract_annotations(path))
    move_chunk = chunks[1]
    assert "–" in move_chunk.text
    assert move_chunk.text == "Bxb4!?: Accepting the gambit – a sharp try."
