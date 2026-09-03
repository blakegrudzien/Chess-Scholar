import json
from unittest.mock import MagicMock, patch

import numpy as np
import psycopg2
import pytest
from pgvector.psycopg2 import register_vector

from scripts.build_study_index import (
    _check_feature_order_matches_training,
    _embed_texts,
    _select_eligible,
    build_index,
)
from src.recommendation.feature_extraction import FEATURE_NAMES

TEST_DB = "chess_rag_build_study_index_test"


def _postgres_available() -> bool:
    try:
        psycopg2.connect(dbname="postgres").close()
        return True
    except psycopg2.OperationalError:
        return False


def _pgvector_available() -> bool:
    try:
        conn = psycopg2.connect(dbname="postgres")
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.close()
        return True
    except psycopg2.Error:
        return False


pytestmark = pytest.mark.skipif(
    not (_postgres_available() and _pgvector_available()),
    reason="requires a local Postgres with the pgvector extension",
)


@pytest.fixture
def conn():
    admin = psycopg2.connect(dbname="postgres")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        cur.execute(f"CREATE DATABASE {TEST_DB}")
    admin.close()

    test_conn = psycopg2.connect(dbname=TEST_DB)
    with test_conn.cursor() as cur:
        cur.execute("CREATE EXTENSION vector")
        cur.execute("""
            CREATE TABLE lichess_study_cache (
                study_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                likes INTEGER NOT NULL,
                quality_probability REAL NOT NULL,
                embedding VECTOR(3) NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
    test_conn.commit()
    register_vector(test_conn)

    yield test_conn

    test_conn.close()
    admin = psycopg2.connect(dbname="postgres")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
    admin.close()


def _candidate(study_id: str, likes: int = 100) -> dict:
    return {
        "study_id": study_id,
        "title": f"Study {study_id}",
        "likes": likes,
        "chapter_titles": ["Intro"],
        "member_usernames": [],
        "pgn": '[Event "E"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n1. e4 e5 *\n',
    }


def test_check_feature_order_matches_training_passes_when_consistent(tmp_path):
    metadata_path = tmp_path / "meta.json"
    metadata_path.write_text(json.dumps({"feature_names": FEATURE_NAMES}), encoding="utf-8")

    _check_feature_order_matches_training(metadata_path)  # must not raise


def test_check_feature_order_matches_training_raises_on_a_reorder(tmp_path):
    """Regression test: the fitted ColumnTransformer inside the saved
    pipeline bakes in feature *positions*, not names -- a future reorder
    (or insertion/removal) of a FEATURE_NAMES entry without retraining
    would silently score every candidate against the wrong columns
    instead of raising, since nothing else checks this at all.
    """
    metadata_path = tmp_path / "meta.json"
    reordered = [*FEATURE_NAMES[1:], FEATURE_NAMES[0]]
    metadata_path.write_text(json.dumps({"feature_names": reordered}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="FEATURE_NAMES has changed"):
        _check_feature_order_matches_training(metadata_path)


def test_check_feature_order_matches_training_raises_when_metadata_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(RuntimeError, match="is missing"):
        _check_feature_order_matches_training(missing_path)


def test_select_eligible_raises_a_clear_error_when_the_model_is_missing():
    """Regression test: joblib.load(MODEL_PATH) on a fresh clone (or a
    models/ directory accidentally cleaned) used to raise a bare
    FileNotFoundError with no guidance -- label_studies_app.py, one script
    over in this same pipeline, already gives this exact courtesy for its
    own missing-input case.
    """
    with patch("scripts.build_study_index.MODEL_PATH") as mock_path:
        mock_path.exists.return_value = False
        with pytest.raises(RuntimeError, match="train_quality_classifier"):
            _select_eligible([_candidate("c1", likes=100)])


def test_select_eligible_filters_by_quality_and_likes(monkeypatch, tmp_path):
    candidates = [
        _candidate("high_quality_high_likes", likes=100),
        _candidate("high_quality_low_likes", likes=1),
        _candidate("low_quality_high_likes", likes=100),
    ]
    probas = iter([0.9, 0.9, 0.1])  # matches candidates order

    fake_pipeline = MagicMock()
    # sklearn's real predict_proba always returns a numpy ndarray, which
    # supports the [0, 1] tuple indexing _select_eligible relies on -- a
    # plain list of lists doesn't, so the fake must match the real type.
    fake_pipeline.predict_proba.side_effect = lambda x: np.array(
        [[1 - p, p] for p in [next(probas)]]
    )

    with (
        patch("scripts.build_study_index.joblib.load", return_value=fake_pipeline),
        # Isolated from the real models/quality_classifier.meta.json on
        # disk deliberately -- this test shouldn't start failing (or
        # passing only by coincidence) because someone retrains the real
        # model later; see the dedicated tests for
        # _check_feature_order_matches_training's own behavior.
        patch("scripts.build_study_index._check_feature_order_matches_training"),
    ):
        eligible = _select_eligible(candidates)

    assert [c["study_id"] for c, _ in eligible] == ["high_quality_high_likes"]


def test_select_eligible_against_a_real_fitted_pipeline(tmp_path):
    # A mocked predict_proba can accept input shapes the real thing
    # rejects -- this exercises the actual sklearn Pipeline (including the
    # log1p/scaling preprocessing step) to catch that class of mismatch,
    # the same one that slipped past the mocked version above.
    from sklearn.linear_model import LogisticRegression

    from src.recommendation.feature_extraction import extract_features
    from src.recommendation.quality_classifier import build_pipeline

    training_candidates = [_candidate(f"t{i}", likes=100) for i in range(10)]
    x = np.array([extract_features(c).as_vector() for c in training_candidates])
    y = np.array([1, 0] * 5)

    real_pipeline = build_pipeline(LogisticRegression())
    real_pipeline.fit(x, y)

    with (
        patch("scripts.build_study_index.joblib.load", return_value=real_pipeline),
        patch("scripts.build_study_index._check_feature_order_matches_training"),
    ):
        eligible = _select_eligible([_candidate("new1", likes=100)])

    assert len(eligible) in (0, 1)  # doesn't raise; either outcome is valid for random data


def test_embed_texts_batches_at_batch_size(monkeypatch):
    monkeypatch.setattr("scripts.build_study_index.BATCH_SIZE", 2)
    texts = ["a", "b", "c", "d", "e"]
    call_sizes = []

    fake_client = MagicMock()

    def fake_embed(batch, **kwargs):
        call_sizes.append(len(batch))
        result = MagicMock()
        result.embeddings = [[0.0, 0.0, 0.0] for _ in batch]
        return result

    fake_client.embed.side_effect = fake_embed

    embeddings = _embed_texts(fake_client, texts)

    assert call_sizes == [2, 2, 1]
    assert len(embeddings) == 5


def test_build_index_upserts_and_removes_ineligible(conn, tmp_path):
    candidates_path = tmp_path / "candidates.jsonl"
    with candidates_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_candidate("keep1", likes=100)) + "\n")
        f.write(json.dumps(_candidate("keep2", likes=100)) + "\n")

    # Pre-seed the cache with a study that will no longer be eligible this
    # run -- the real scenario a full sync (not append) needs to handle.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO lichess_study_cache "
            "(study_id, title, likes, quality_probability, embedding) "
            "VALUES ('stale1', 'Stale', 5, 0.9, %s)",
            ([0.1, 0.1, 0.1],),
        )
    conn.commit()

    fake_pipeline = MagicMock()
    fake_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])

    fake_client = MagicMock()

    def fake_embed(batch, **kwargs):
        result = MagicMock()
        result.embeddings = [[0.5, 0.5, 0.5] for _ in batch]
        return result

    fake_client.embed.side_effect = fake_embed

    with (
        patch("scripts.build_study_index.CANDIDATES_PATH", candidates_path),
        patch("scripts.build_study_index.joblib.load", return_value=fake_pipeline),
        patch("scripts.build_study_index.get_voyage_client", return_value=fake_client),
        patch("scripts.build_study_index.get_connection", return_value=conn),
    ):
        build_index()

    # build_index() closes the connection it was handed when it's done, the
    # same resource-ownership contract every other DB-writing script in
    # this project follows -- a fresh connection is needed to inspect the
    # result afterward, not a reason to change that contract.
    verify_conn = psycopg2.connect(dbname=TEST_DB)
    with verify_conn.cursor() as cur:
        cur.execute("SELECT study_id FROM lichess_study_cache ORDER BY study_id")
        remaining = [row[0] for row in cur.fetchall()]
    verify_conn.close()

    assert remaining == ["keep1", "keep2"]
    assert "stale1" not in remaining
