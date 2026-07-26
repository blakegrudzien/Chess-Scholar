from src.ingestion.pgn_encoding import read_pgn_text


def test_valid_utf8_decodes_unchanged(tmp_path):
    text = "café — a clean UTF-8 file\n"
    path = tmp_path / "utf8.pgn"
    path.write_bytes(text.encode("utf-8"))
    assert read_pgn_text(path) == text


def test_whole_file_cp1252_recovers_correctly(tmp_path):
    text = "Accepting the gambit – a sharp try.\n"
    path = tmp_path / "cp1252.pgn"
    path.write_bytes(text.encode("cp1252"))
    assert read_pgn_text(path) == text


def test_isolated_bad_byte_falls_back_without_corrupting_valid_utf8(tmp_path):
    # Mirrors real ChessBase export corruption found in data/raw/game3.pgn:
    # a comment with a correctly UTF-8-encoded character (U+00B4) followed by
    # a raw Windows-1252 byte (0xDF = 'ß') that isn't valid UTF-8 on its own.
    raw = "White´s pieces. Wei".encode() + b"\xdf" + b"en Figuren."
    path = tmp_path / "mixed.pgn"
    path.write_bytes(raw)

    assert read_pgn_text(path) == "White´s pieces. Weißen Figuren."
