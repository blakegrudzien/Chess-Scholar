"""Build the searchable pool of recommendable Lichess studies: score every
collected candidate with the trained quality classifier, keep the ones
eligible for recommendation (study_index.is_eligible_for_recommendation --
both quality and a likes floor), embed their content via Voyage, and sync
the result into lichess_study_cache on Neon.

A full sync, not an append: candidates that no longer clear eligibility
(e.g. after retraining the classifier on more labels) are deleted from the
cache, not left stale, and existing rows are upserted rather than
duplicated. Safe to re-run any time the candidate pool, labels, or trained
model change -- there's no notion of "already processed" to get out of
sync with, unlike voyage_embedder.py's chunks backfill (which only ever
adds rows and never revisits one once embedded).

Run as a module from the repo root:
    .venv/bin/python -m scripts.build_study_index
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from src.embeddings.voyage_embedder import BATCH_SIZE, get_voyage_client
from src.ingestion.db_loader import get_connection
from src.recommendation.feature_extraction import FEATURE_NAMES, extract_features
from src.recommendation.study_index import build_embedding_text, is_eligible_for_recommendation

logger = logging.getLogger(__name__)

CANDIDATES_PATH = Path("data/processed/study_candidates.jsonl")
MODEL_PATH = Path("models/quality_classifier.joblib")
MODEL_METADATA_PATH = Path("models/quality_classifier.meta.json")


def _load_candidates() -> list[dict]:
    candidates = []
    with CANDIDATES_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates


def _check_feature_order_matches_training(metadata_path: Path = MODEL_METADATA_PATH) -> None:
    """The fitted ColumnTransformer inside the saved pipeline bakes in
    feature *positions*, not names -- ColumnTransformer.transform has no
    way to know FEATURE_NAMES has changed since fit time, so a future
    reorder (or insertion/removal) of a feature would silently score every
    candidate against the wrong columns instead of raising. Comparing
    against feature_names in the model's own metadata (written by
    scripts/train_quality_classifier.py alongside the .joblib) catches
    that mismatch loudly, before it can corrupt a whole index build,
    rather than not at all.

    Missing metadata (a model file present without its sidecar -- an old
    commit predating this check, or a hand-copied .joblib) is treated the
    same as a mismatch: there's nothing to verify consistency against, so
    trusting the model blindly isn't safe either.
    """
    if not metadata_path.exists():
        raise RuntimeError(
            f"{metadata_path} is missing -- can't verify {MODEL_PATH} was trained against "
            "the current FEATURE_NAMES order. Retrain via scripts.train_quality_classifier "
            "(it writes this file alongside the model), or restore the metadata file."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trained_feature_names = metadata["feature_names"]
    if trained_feature_names != FEATURE_NAMES:
        raise RuntimeError(
            f"FEATURE_NAMES has changed since {MODEL_PATH} was trained "
            f"(trained on {trained_feature_names}, current is {FEATURE_NAMES}) -- "
            "scoring candidates against it now would silently use the wrong "
            "columns. Retrain via scripts.train_quality_classifier first."
        )


def _select_eligible(candidates: list[dict]) -> list[tuple[dict, float]]:
    if not MODEL_PATH.exists():
        # label_studies_app.py, one script over in this same pipeline,
        # already gives this exact courtesy for its own missing-input
        # case (CANDIDATES_PATH) -- a bare FileNotFoundError from
        # joblib.load with no guidance was the gap here specifically.
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run `python -m scripts.train_quality_classifier` first."
        )
    _check_feature_order_matches_training()
    pipeline = joblib.load(MODEL_PATH)
    eligible = []
    for candidate in candidates:
        features = extract_features(candidate)
        proba = float(pipeline.predict_proba(np.array([features.as_vector()]))[0, 1])
        if is_eligible_for_recommendation(candidate["likes"], proba):
            eligible.append((candidate, proba))
    return eligible


def _embed_texts(client, texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        result = client.embed(batch, model="voyage-4", input_type="document")
        embeddings.extend(result.embeddings)
    return embeddings


def build_index() -> None:
    candidates = _load_candidates()
    logger.info("Loaded %d candidates", len(candidates))

    eligible = _select_eligible(candidates)
    logger.info("%d of %d candidates eligible for recommendation", len(eligible), len(candidates))

    client = get_voyage_client()
    texts = [build_embedding_text(candidate) for candidate, _ in eligible]
    embeddings = _embed_texts(client, texts)
    logger.info("Embedded %d candidates via Voyage", len(embeddings))

    rows = [
        (candidate["study_id"], candidate["title"], candidate["likes"], proba, embedding)
        for (candidate, proba), embedding in zip(eligible, embeddings, strict=True)
    ]
    eligible_ids = [row[0] for row in rows]

    conn = get_connection()
    register_vector(conn)
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO lichess_study_cache
                    (study_id, title, likes, quality_probability, embedding, updated_at)
                VALUES %s
                ON CONFLICT (study_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    likes = EXCLUDED.likes,
                    quality_probability = EXCLUDED.quality_probability,
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                """,
                rows,
                template="(%s, %s, %s, %s, %s::vector, now())",
                page_size=len(rows) if rows else 1,
            )
            cur.execute(
                "DELETE FROM lichess_study_cache WHERE NOT (study_id = ANY(%s))",
                (eligible_ids,),
            )
            deleted = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    logger.info("Synced lichess_study_cache: %d upserted, %d removed", len(rows), deleted)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_index()
