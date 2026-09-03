"""Train and evaluate the study quality classifier.

Compares two candidate models via 5-fold cross-validation on the labeled
dataset (drill/concept genres only -- see quality_classifier.py's module
docstring for why narrative is excluded), reports overall and per-genre/
per-phase metrics for each, picks the higher cross-validated ROC-AUC,
refits the winner on the full labeled set, and saves it.

ROC-AUC (ranking quality, independent of any specific decision threshold)
is the selection criterion because this classifier's output feeds a later
cascade stage that ranks/filters candidates by predicted quality -- what
matters is whether it orders good studies above bad ones, not whether a
threshold of exactly 0.5 happens to split them well.

Run as a module from the repo root:
    .venv/bin/python -m scripts.train_quality_classifier
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV

from src.recommendation.feature_extraction import FEATURE_NAMES
from src.recommendation.quality_classifier import (
    Metrics,
    breakdown_metrics,
    build_feature_matrix,
    build_pipeline,
    compute_metrics,
    cross_validated_predictions,
    load_labeled_dataset,
)

logger = logging.getLogger(__name__)

CANDIDATES_PATH = Path("data/processed/study_candidates.jsonl")
LABELS_PATH = Path("data/processed/study_labels.jsonl")
MODEL_OUTPUT_PATH = Path("models/quality_classifier.joblib")
# A sibling JSON file, not a docstring or commit message, so a reader (or
# build_study_index.py itself, see its own check) can get this
# programmatically instead of digging through git log -- what the
# committed .joblib actually is: when it was trained, on how many
# examples, which model won and at what cross-validated ROC-AUC, and the
# exact FEATURE_NAMES order it was fit against. That last field is what
# lets build_study_index.py catch a future FEATURE_NAMES reorder before it
# silently corrupts every prediction, instead of after.
MODEL_METADATA_PATH = Path("models/quality_classifier.meta.json")
RANDOM_STATE = 0


def _build_candidate_models() -> dict[str, ClassifierMixin]:
    """Two different tuning strategies, deliberately: LogisticRegressionCV
    tunes its own regularization strength via an efficient internal CV
    search, which scikit-learn supports natively for linear models. There's
    no equivalent built-in for gradient boosting, and hand-rolling a nested
    CV grid search over HistGradientBoostingClassifier's hyperparameters on
    only ~140 examples risks overfitting the *search* to the CV folds
    themselves -- more tuning knobs turned on this little data is a bigger
    risk than a slightly worse fixed configuration. So its hyperparameters
    are fixed instead, deliberately conservative (shallow trees, few
    iterations, real L2) to match how little data it has to avoid
    memorizing.
    """
    return {
        "logistic_regression": LogisticRegressionCV(
            cv=5,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            # roc_auc, not accuracy: this module's own docstring picks
            # ROC-AUC over the outer two candidate models specifically
            # because ranking quality, independent of any one threshold,
            # is what the downstream cascade needs -- the same reasoning
            # applies one level down to how this model's own regularization
            # strength gets tuned. Scoring the inner search by accuracy
            # while scoring the outer choice by ROC-AUC was a real mismatch
            # between what's stated and what's implemented.
            scoring="roc_auc",
            l1_ratios=[0.0],
            use_legacy_attributes=False,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=50,
            learning_rate=0.1,
            min_samples_leaf=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def _format_metrics(label: str, metrics: Metrics | None) -> str:
    if metrics is None:
        return f"{label}: (n < minimum, not reported)"
    auc = f"{metrics.roc_auc:.3f}" if metrics.roc_auc is not None else "n/a (single class)"
    return (
        f"{label}: n={metrics.n} (positive={metrics.n_positive}) "
        f"acc={metrics.accuracy:.3f} prec={metrics.precision:.3f} "
        f"rec={metrics.recall:.3f} f1={metrics.f1:.3f} roc_auc={auc}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    examples = load_labeled_dataset(CANDIDATES_PATH, LABELS_PATH)
    logger.info("Loaded %d labeled examples (drill/concept only)", len(examples))
    x, y = build_feature_matrix(examples)
    genres = [e.genre for e in examples]
    phases = [e.phase for e in examples]

    results = {}
    for name, model in _build_candidate_models().items():
        pipeline = build_pipeline(model)
        y_proba = cross_validated_predictions(pipeline, x, y, random_state=RANDOM_STATE)
        results[name] = y_proba

        print(f"\n=== {name} ===")
        overall = compute_metrics(y, y_proba)
        print(_format_metrics("overall", overall))
        print(" by genre:")
        for genre, metrics in breakdown_metrics(y, y_proba, genres).items():
            print(f"  {_format_metrics(genre, metrics)}")
        print(" by phase:")
        for phase, metrics in breakdown_metrics(y, y_proba, phases).items():
            print(f"  {_format_metrics(phase, metrics)}")

    scores = {name: compute_metrics(y, proba).roc_auc or 0.0 for name, proba in results.items()}
    winner_name = max(scores, key=lambda name: scores[name])
    print(f"\nSelected: {winner_name} (cross-validated ROC-AUC {scores[winner_name]:.3f})")

    winner_model = _build_candidate_models()[winner_name]
    winner_pipeline = build_pipeline(winner_model)
    winner_pipeline.fit(x, y)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_pipeline, MODEL_OUTPUT_PATH)
    MODEL_METADATA_PATH.write_text(
        json.dumps(
            {
                "trained_at": datetime.now(UTC).isoformat(),
                "model_name": winner_name,
                "cv_roc_auc": scores[winner_name],
                "n_examples": len(examples),
                "feature_names": FEATURE_NAMES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Refit %s on all %d examples and saved to %s (metadata: %s)",
        winner_name,
        len(examples),
        MODEL_OUTPUT_PATH,
        MODEL_METADATA_PATH,
    )


if __name__ == "__main__":
    main()
