import logging
from unittest.mock import patch

from src.recommendation.study_pgn import iter_study_chapters

TWO_CHAPTER_PGN = (
    '[Event "Chapter 1: Intro"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n'
    "1. e4 e5 *\n\n"
    '[Event "Chapter 2: Follow-up"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n'
    "1. d4 d5 *\n"
)


def test_iter_study_chapters_yields_one_game_per_chapter():
    chapters = list(iter_study_chapters(TWO_CHAPTER_PGN))
    assert len(chapters) == 2
    assert chapters[0].headers["Event"] == "Chapter 1: Intro"
    assert chapters[1].headers["Event"] == "Chapter 2: Follow-up"


def test_iter_study_chapters_empty_text_yields_nothing():
    assert list(iter_study_chapters("")) == []


def test_iter_study_chapters_single_chapter():
    chapters = list(iter_study_chapters('[Event "Only"]\n\n1. e4 *\n'))
    assert len(chapters) == 1


def test_iter_study_chapters_logs_a_warning_instead_of_silently_truncating(caplog):
    """Regression test: a parse failure used to stop the generator with a
    bare `except Exception: return` and no logging at all -- this backs
    both the quality classifier's training features and the search
    index's embedding text, so a chapter silently dropped here degrades
    either with nothing pointing at why.
    """
    with (
        patch("chess.pgn.read_game", side_effect=[object(), ValueError("bad pgn")]),
        caplog.at_level(logging.WARNING),
    ):
        chapters = list(iter_study_chapters(TWO_CHAPTER_PGN))

    assert len(chapters) == 1  # the first (fake) chapter, before the failure
    assert "parse error" in caplog.text
