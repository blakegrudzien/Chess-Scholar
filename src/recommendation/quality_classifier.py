"""Quality classifier for the study recommendation feature: given a
candidate's structural features (feature_extraction.py), predict how
likely a human labeler would be to mark it "recommend" rather than
"reject."

Trained on drill and concept studies only. All 21 narrative-genre labels
were "reject", and narrative recommendations come from ChessBase games
instead (structured_search.select_narrative_game), so training on that
genre would only teach a pattern the model is never used for.

Classical models (logistic regression, gradient-boosted trees) because the
data is small: a couple of hundred examples labeled by one person (the
exact count is n_examples in models/quality_classifier.meta.json).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.recommendation.feature_extraction import FEATURE_NAMES, StudyFeatures, extract_features

# The genres the classifier scores (narrative excluded, see module docstring).
TRAINED_GENRES = frozenset({"drill", "concept"})

# Counts are right-skewed by a few huge studies, so they get log1p.
# comment_density (a ratio) and has_pre_game_comment (binary) don't.
_LOG_TRANSFORM_FEATURES = [
    "num_chapters",
    "total_plies",
    "num_annotated_plies",
    "total_comment_word_count",
    "avg_comment_word_count",
    "num_nag_annotations",
    "likes",
    "num_members",
    "title_length",
]
_LOG_TRANSFORM_INDICES = [FEATURE_NAMES.index(name) for name in _LOG_TRANSFORM_FEATURES]
_PASSTHROUGH_INDICES = [i for i in range(len(FEATURE_NAMES)) if i not in _LOG_TRANSFORM_INDICES]

# Minimums before a per-genre or per-phase metric is reported. Group size
# alone isn't enough: a 37-example group with 2 rejects passes a size check,
# but its ROC-AUC rests on those 2 examples. The minority class needs a
# minimum count too.
MIN_GROUP_SIZE_FOR_METRICS = 10
MIN_MINORITY_CLASS_SIZE_FOR_METRICS = 5

# The decision boundary, also imported by study_index's eligibility check.
DEFAULT_QUALITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class LabeledExample:
    features: StudyFeatures
    genre: str
    phase: str | None
    quality: int  # 1 = recommend, 0 = reject


def load_labeled_dataset(
    candidates_path: Path, labels_path: Path, *, genres: frozenset[str] = TRAINED_GENRES
) -> list[LabeledExample]:
    """Join collected candidates with human labels, keeping only real
    (non-skipped) labels whose genre is in `genres`.
    """
    candidates_by_id = {}
    with candidates_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                candidates_by_id[record["study_id"]] = record

    examples = []
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            label = json.loads(line)
            if label.get("skipped"):
                continue
            if label["genre"] not in genres:
                continue
            candidate = candidates_by_id.get(label["study_id"])
            if candidate is None:
                raise ValueError(
                    f"study_labels.jsonl references study_id "
                    f"{label['study_id']!r}, which isn't in {candidates_path}: "
                    "a stale labels file, a hand-edited candidates file, or the "
                    "wrong path was passed?"
                )
            examples.append(
                LabeledExample(
                    features=extract_features(candidate),
                    genre=label["genre"],
                    phase=label.get("phase"),
                    quality=1 if label["quality"] == "recommend" else 0,
                )
            )
    return examples


def build_feature_matrix(examples: list[LabeledExample]) -> tuple[np.ndarray, np.ndarray]:
    """X, y for scikit-learn: X in FEATURE_NAMES order (StudyFeatures.as_vector's
    contract), y as 0/1 quality labels.
    """
    x = np.array([example.features.as_vector() for example in examples])
    y = np.array([example.quality for example in examples])
    return x, y


def build_pipeline(model: ClassifierMixin) -> Pipeline:
    """Wrap a classifier with the shared preprocessing: log1p on the
    skewed count features, then standardize every feature. Used identically
    for every candidate model under comparison so differences in
    cross-validated performance reflect the model, not inconsistent
    preprocessing between them. The log1p step is a no-op for a tree-based
    model (monotonic transforms don't change split points) but doesn't
    hurt it either, so one shared pipeline definition covers both rather
    than maintaining two.
    """
    preprocessing = ColumnTransformer(
        [
            (
                "log1p",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                _LOG_TRANSFORM_INDICES,
            ),
            ("passthrough", "passthrough", _PASSTHROUGH_INDICES),
        ]
    )
    return Pipeline([("preprocess", preprocessing), ("scale", StandardScaler()), ("model", model)])


def cross_validated_predictions(
    pipeline: Pipeline, x: np.ndarray, y: np.ndarray, *, n_splits: int = 5, random_state: int = 0
) -> np.ndarray:
    """Out-of-fold P(recommend) for every example (stratified k-fold), so
    each prediction comes from a model that never saw that example.
    Per-genre and per-phase metrics are computed from these."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(pipeline, x, y, cv=cv, method="predict_proba")[:, 1]


@dataclass(frozen=True)
class Metrics:
    n: int
    n_positive: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, *, threshold: float = DEFAULT_QUALITY_THRESHOLD
) -> Metrics:
    y_pred = (y_proba >= threshold).astype(int)
    n_positive = int(y_true.sum())
    # ROC-AUC is undefined with only one class present in y_true.
    roc_auc = roc_auc_score(y_true, y_proba) if 0 < n_positive < len(y_true) else None
    return Metrics(
        n=len(y_true),
        n_positive=n_positive,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc) if roc_auc is not None else None,
    )


def breakdown_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups: Sequence[str | None],
    *,
    min_group_size: int = MIN_GROUP_SIZE_FOR_METRICS,
    min_minority_class_size: int = MIN_MINORITY_CLASS_SIZE_FOR_METRICS,
) -> dict[str, Metrics | None]:
    """Metrics for each value in `groups` (e.g. genre or phase), aligned
    with y_true/y_proba. A group too small overall, or with too few of its
    minority class, gets None rather than a misleading number."""
    labels = ["unknown" if g is None else g for g in groups]
    result: dict[str, Metrics | None] = {}
    for label in sorted(set(labels)):
        mask = np.array([lbl == label for lbl in labels])
        group_y = y_true[mask]
        minority_class_size = min(int(group_y.sum()), int((1 - group_y).sum()))
        if mask.sum() < min_group_size or minority_class_size < min_minority_class_size:
            result[label] = None
            continue
        result[label] = compute_metrics(y_true[mask], y_proba[mask])
    return result
