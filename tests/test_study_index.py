from src.recommendation.study_index import (
    MAX_ANNOTATION_CHARS_FOR_EMBEDDING,
    MIN_LIKES_FOR_RECOMMENDATION,
    build_embedding_text,
    is_eligible_for_recommendation,
)


def test_is_eligible_requires_both_likes_and_quality():
    assert is_eligible_for_recommendation(likes=100, quality_probability=0.9) is True
    assert is_eligible_for_recommendation(likes=5, quality_probability=0.9) is False
    assert is_eligible_for_recommendation(likes=100, quality_probability=0.1) is False


def test_is_eligible_at_exact_thresholds():
    assert is_eligible_for_recommendation(
        likes=MIN_LIKES_FOR_RECOMMENDATION, quality_probability=0.5
    )
    assert not is_eligible_for_recommendation(
        likes=MIN_LIKES_FOR_RECOMMENDATION - 1, quality_probability=0.5
    )


def test_is_eligible_respects_custom_quality_threshold():
    assert not is_eligible_for_recommendation(
        likes=100, quality_probability=0.6, quality_threshold=0.7
    )
    assert is_eligible_for_recommendation(likes=100, quality_probability=0.6, quality_threshold=0.5)


def test_build_embedding_text_includes_title_and_chapter_titles():
    candidate = {
        "title": "Caro-Kann Basics",
        "chapter_titles": ["Intro", "Advance Variation"],
        "pgn": "",
    }
    text = build_embedding_text(candidate)
    assert "Caro-Kann Basics" in text
    assert "Intro" in text
    assert "Advance Variation" in text


def test_build_embedding_text_includes_annotation_prose():
    candidate = {
        "title": "Study",
        "chapter_titles": ["Chapter 1"],
        "pgn": (
            '[Event "Chapter 1"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n'
            "{This explains the key idea behind the whole system.} 1. e4 e5 *\n"
        ),
    }
    text = build_embedding_text(candidate)
    assert "key idea behind the whole system" in text


def test_build_embedding_text_truncates_long_annotation():
    long_comment = "word " * 1000  # far more than MAX_ANNOTATION_CHARS_FOR_EMBEDDING
    candidate = {
        "title": "Study",
        "chapter_titles": [],
        "pgn": (
            f'[Event "Chapter 1"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n'
            f"{{{long_comment}}} 1. e4 *\n"
        ),
    }
    text = build_embedding_text(candidate)
    # title line + newline, plus at most MAX_ANNOTATION_CHARS_FOR_EMBEDDING of prose
    assert len(text) <= len("Study\n") + MAX_ANNOTATION_CHARS_FOR_EMBEDDING


def test_build_embedding_text_handles_no_annotations():
    candidate = {
        "title": "Bare Study",
        "chapter_titles": ["Chapter 1"],
        "pgn": '[Event "Chapter 1"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n1. e4 e5 *\n',
    }
    text = build_embedding_text(candidate)
    assert text == "Bare Study\nChapter 1"
