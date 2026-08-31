"""Quality classifier for the study recommendation feature (Step 5 of the
recommendation roadmap): given a candidate's structural features
(feature_extraction.py), predict how likely a human labeler would be to
mark it "recommend" rather than "reject."

Scope note: trained only on drill/concept-genre labels, not narrative.
Every one of the 21 narrative-genre labels in study_labels.jsonl came back
"reject" -- Lichess narrative studies turned out not to be worth
recommending at all, which is exactly why the narrative lane now draws a
plain-moves ChessBase game instead (see structured_search.select_narrative_
game). Training on an all-reject genre this classifier will never be asked
to score again would just teach it a pattern that has no bearing on its
actual job at inference time -- a textbook train/serve skew, avoided by
excluding it up front rather than filtering it out later.

Classical ML (logistic regression, gradient-boosted trees), not anything
deep: the labeled set here is small -- around 140 examples after excluding
narrative and skips -- hand-labeled by one person. Model capacity should
track the data available, not the data one might wish for; a small,
regularized classical model is the amount of capacity ~140 examples can
actually support without just memorizing them.
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

# Genres this classifier is trained on and will ever be asked to score.
# Narrative is excluded -- see module docstring.
TRAINED_GENRES = frozenset({"drill", "concept"})

# Count-like features are right-skewed (a handful of very long, very
# heavily-annotated studies dominate the raw scale) -- log1p first so a
# linear model's regularization penalty and a tree model's split thresholds
# both see a roughly symmetric distribution instead of a few outliers
# dominating the scale. comment_density is already a bounded ratio and
# has_pre_game_comment is already binary, so both skip the transform.
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

# Minimums before a genre/phase-broken-out metric is reported at all,
# mirroring CLAUDE.md's rule for trend synthesis: don't claim a pattern
# exists from a handful of examples. Two guards, not one: total group size
# alone isn't enough -- a 37-example group with only 2 examples of one
# class still passes a size-10 threshold, but its ROC-AUC is then computed
# from just those 2 examples' relative ranking and is nearly pure noise
# (this happened for real: the "unknown"-phase group, all labels collected
# before the phase field existed, is 35 recommend / 2 reject). Requiring a
# minimum count of the minority class specifically catches that case a
# total-size check alone misses.
MIN_GROUP_SIZE_FOR_METRICS = 10
MIN_MINORITY_CLASS_SIZE_FOR_METRICS = 5


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
            candidate = candidates_by_id[label["study_id"]]
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
    """Out-of-fold predicted P(recommend) for every example, via stratified
    k-fold: each example's prediction comes from a fold that never saw it
    during fitting, so aggregating these across all folds gives an honest,
    full-dataset estimate of generalization -- the standard way to get
    per-subgroup (genre/phase) metrics out of cross-validation, rather than
    trying to average per-fold-per-subgroup numbers computed on much
    smaller slices.
    """
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


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, *, threshold: float = 0.5) -> Metrics:
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
    """Metrics per distinct value in `groups` (e.g. genre or phase),
    aligned index-for-index with y_true/y_proba. Two conditions gate
    whether a group gets a real Metrics or None: total size, and minority
    class size specifically -- a group can clear the size bar while one
    class is still only 2 examples, which makes ROC-AUC in particular
    almost pure noise (see MIN_MINORITY_CLASS_SIZE_FOR_METRICS). Reporting
    a number in either case would overstate confidence the data doesn't
    support -- the same reasoning CLAUDE.md already applies to trend
    synthesis: require a minimum count before claiming a pattern.
    """
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
