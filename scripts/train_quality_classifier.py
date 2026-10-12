"""Train and evaluate the study quality classifier.

Compares two models with 5-fold cross-validation on the labeled data
(drill and concept genres only, see quality_classifier.py), reports overall
and per-genre and per-phase metrics, picks the higher ROC-AUC, refits the
winner on all the data and saves it.

ROC-AUC because the scores are used to rank and filter studies: what
matters is ordering good studies above bad ones, not where one threshold
falls.

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
# Metadata saved next to the model: when it was trained, on how many
# examples, which model won at what ROC-AUC, and the FEATURE_NAMES order it
# was fit on, which build_study_index.py checks before scoring.
MODEL_METADATA_PATH = Path("models/quality_classifier.meta.json")
RANDOM_STATE = 0


def _build_candidate_models() -> dict[str, ClassifierMixin]:
    """The two candidates. LogisticRegressionCV tunes its own regularization
    by internal cross-validation. Gradient boosting has no such built-in,
    and searching its hyperparameters on ~140 examples would overfit the
    search itself, so it uses fixed, conservative settings (shallow trees,
    few iterations, L2).
    """
    return {
        "logistic_regression": LogisticRegressionCV(
            cv=5,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            # Tuned on ROC-AUC, the same measure the model choice uses.
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
