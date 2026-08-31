import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.recommendation.quality_classifier import (
    breakdown_metrics,
    build_feature_matrix,
    build_pipeline,
    compute_metrics,
    cross_validated_predictions,
    load_labeled_dataset,
)


def _candidate(study_id: str, likes: int = 10, pgn: str | None = None) -> dict:
    return {
        "study_id": study_id,
        "title": f"Study {study_id}",
        "likes": likes,
        "member_usernames": ["alice"],
        "pgn": pgn or '[Event "E"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n1. e4 e5 *\n',
    }


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_labeled_dataset_excludes_skipped_and_non_target_genres(tmp_path):
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(
        candidates_path,
        [_candidate("c1"), _candidate("d1"), _candidate("n1"), _candidate("skip1")],
    )
    _write_jsonl(
        labels_path,
        [
            {"study_id": "c1", "genre": "concept", "phase": "opening", "quality": "recommend"},
            {"study_id": "d1", "genre": "drill", "phase": "general", "quality": "reject"},
            {"study_id": "n1", "genre": "narrative", "phase": "general", "quality": "reject"},
            {"study_id": "skip1", "skipped": True},
        ],
    )

    examples = load_labeled_dataset(candidates_path, labels_path)

    assert {e.features.study_id for e in examples} == {"c1", "d1"}


def test_load_labeled_dataset_maps_quality_to_binary(tmp_path):
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(candidates_path, [_candidate("c1"), _candidate("c2")])
    _write_jsonl(
        labels_path,
        [
            {"study_id": "c1", "genre": "concept", "phase": "opening", "quality": "recommend"},
            {"study_id": "c2", "genre": "concept", "phase": "opening", "quality": "reject"},
        ],
    )

    examples = load_labeled_dataset(candidates_path, labels_path)
    quality_by_id = {e.features.study_id: e.quality for e in examples}
    assert quality_by_id == {"c1": 1, "c2": 0}


def test_load_labeled_dataset_preserves_missing_phase_as_none(tmp_path):
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(candidates_path, [_candidate("c1")])
    _write_jsonl(labels_path, [{"study_id": "c1", "genre": "concept", "quality": "recommend"}])

    examples = load_labeled_dataset(candidates_path, labels_path)
    assert examples[0].phase is None


def test_build_feature_matrix_shape_and_order(tmp_path):
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(candidates_path, [_candidate("c1"), _candidate("c2")])
    _write_jsonl(
        labels_path,
        [
            {"study_id": "c1", "genre": "concept", "phase": "opening", "quality": "recommend"},
            {"study_id": "c2", "genre": "concept", "phase": "opening", "quality": "reject"},
        ],
    )
    examples = load_labeled_dataset(candidates_path, labels_path)

    x, y = build_feature_matrix(examples)
    assert x.shape == (2, len(examples[0].features.as_vector()))
    assert list(y) == [1, 0]


def _synthetic_feature_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """Values shaped like real StudyFeatures.as_vector() output: every
    log1p-transformed column (all but comment_density and
    has_pre_game_comment -- see _LOG_TRANSFORM_FEATURES) is a non-negative
    count in the real pipeline, by construction of feature_extraction.py,
    so build_pipeline's log1p step relies on that. Plain rng.normal() would
    produce negative values and log1p(x < -1) is NaN -- not a bug in
    build_pipeline, just an unrealistic input shape a random normal draw
    doesn't match.
    """
    x = rng.integers(0, 50, size=(n, 11)).astype(float)
    x[:, 3] = rng.uniform(0, 1, size=n)  # comment_density, index 3
    x[:, 6] = rng.integers(0, 2, size=n)  # has_pre_game_comment, index 6
    return x


def test_build_pipeline_fits_and_predicts_probabilities():
    rng = np.random.default_rng(0)
    x = _synthetic_feature_matrix(rng, 40)
    y = (x[:, 3] > 0.5).astype(int)  # cleanly separable on comment_density

    pipeline = build_pipeline(LogisticRegression())
    pipeline.fit(x, y)
    proba = pipeline.predict_proba(x)[:, 1]

    assert proba.shape == (40,)
    assert ((proba >= 0) & (proba <= 1)).all()
    # A clearly separable signal should be recovered reasonably well.
    assert ((proba >= 0.5).astype(int) == y).mean() > 0.8


def test_compute_metrics_matches_hand_computed_values():
    y_true = np.array([1, 1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.3, 0.2, 0.6])  # one false negative, one false positive
    metrics = compute_metrics(y_true, y_proba)

    # predicted: [1, 1, 0, 0, 1] at threshold 0.5
    assert metrics.accuracy == 3 / 5
    assert metrics.precision == 2 / 3  # 2 true positives, 1 false positive
    assert metrics.recall == 2 / 3  # 2 true positives, 1 false negative
    assert metrics.roc_auc is not None


def test_compute_metrics_roc_auc_none_when_single_class():
    y_true = np.array([1, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.7])
    assert compute_metrics(y_true, y_proba).roc_auc is None


def test_breakdown_metrics_respects_min_group_size():
    y_true = np.array([1, 0, 1, 0, 1])
    y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    groups = ["a", "a", "a", "b", "b"]  # group "b" has only 2 examples

    result = breakdown_metrics(y_true, y_proba, groups, min_group_size=3, min_minority_class_size=0)
    assert result["a"] is not None
    assert result["b"] is None


def test_breakdown_metrics_respects_min_minority_class_size():
    # Group "a" has 10 examples total (clears a size-5 minimum) but only 1
    # of the minority class -- this is exactly the real "unknown"-phase
    # case that motivated the guard: total size alone isn't enough.
    y_true = np.array([1] * 9 + [0])
    y_proba = np.array([0.9] * 9 + [0.1])
    groups = ["a"] * 10

    result = breakdown_metrics(y_true, y_proba, groups, min_group_size=5, min_minority_class_size=3)
    assert result["a"] is None


def test_breakdown_metrics_maps_none_group_to_unknown():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    groups = [None, None, None, None, None, None]

    result = breakdown_metrics(y_true, y_proba, groups, min_group_size=3, min_minority_class_size=1)
    assert "unknown" in result
    assert result["unknown"] is not None


def test_cross_validated_predictions_covers_every_example_in_valid_range():
    rng = np.random.default_rng(1)
    x = _synthetic_feature_matrix(rng, 50)
    y = (x[:, 3] > 0.5).astype(int)

    pipeline = build_pipeline(LogisticRegression())
    proba = cross_validated_predictions(pipeline, x, y, n_splits=5, random_state=0)

    assert proba.shape == (50,)
    assert ((proba >= 0) & (proba <= 1)).all()
