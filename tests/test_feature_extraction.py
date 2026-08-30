from src.recommendation.feature_extraction import FEATURE_NAMES, extract_features

CHAPTER_1 = (
    '[Event "Chapter 1"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n'
    "{Intro text here} 1. e4 $1 {Central control} e5 *\n"
)
CHAPTER_2 = '[Event "Chapter 2"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n1. d4 {Queen pawn} *\n'

CANDIDATE = {
    "study_id": "test123",
    "title": "Test Study",
    "likes": 42,
    "member_usernames": ["alice", "bob"],
    "pgn": CHAPTER_1 + "\n\n" + CHAPTER_2,
}


def test_extract_features_computes_expected_values():
    features = extract_features(CANDIDATE)

    assert features.study_id == "test123"
    assert features.num_chapters == 2
    assert features.total_plies == 3  # e4, e5, d4
    assert features.num_annotated_plies == 2  # e4 and d4 have comments; e5 doesn't
    assert features.comment_density == 2 / 3
    # "Intro text here" (3) + "Central control" (2) + "Queen pawn" (2) = 7
    assert features.total_comment_word_count == 7
    assert features.avg_comment_word_count == 3.5
    assert features.has_pre_game_comment is True
    assert features.num_nag_annotations == 1  # only e4 has $1
    assert features.likes == 42
    assert features.num_members == 2
    assert features.title_length == len("Test Study")


def test_as_vector_matches_feature_names_order_and_length():
    features = extract_features(CANDIDATE)
    vector = features.as_vector()
    assert len(vector) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vector)


def test_extract_features_handles_a_study_with_no_annotations():
    candidate = {
        "study_id": "bare",
        "title": "Bare",
        "likes": 0,
        "member_usernames": [],
        "pgn": '[Event "Only"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n1. e4 e5 *\n',
    }
    features = extract_features(candidate)
    assert features.num_annotated_plies == 0
    assert features.comment_density == 0.0
    assert features.avg_comment_word_count == 0.0
    assert features.has_pre_game_comment is False
