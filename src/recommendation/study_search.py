"""Query-time similarity search over the recommendable study pool
(lichess_study_cache). Mirrors src/rag/vector_search.py's read/write
split: that module's write side is src/embeddings/voyage_embedder.py,
this one's is scripts/build_study_index.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg2.extensions
import voyageai
from pgvector.psycopg2 import register_vector


@dataclass
class StudyResult:
    study_id: str
    title: str
    likes: int
    quality_probability: float
    distance: float


def search_studies(
    conn: psycopg2.extensions.connection,
    client: voyageai.Client,
    query: str,
    model: str = "voyage-4",
    limit: int = 5,
) -> list[StudyResult]:
    """Return the `limit` recommendable studies most semantically similar
    to `query`, nearest first.
    """
    register_vector(conn)
    result = client.embed([query], model=model, input_type="query")
    query_vector = result.embeddings[0]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT study_id, title, likes, quality_probability,
                   embedding <=> %s::vector AS distance
            FROM lichess_study_cache
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, limit),
        )
        rows = cur.fetchall()

    return [
        StudyResult(
            study_id=study_id,
            title=title,
            likes=likes,
            quality_probability=quality_probability,
            distance=distance,
        )
        for study_id, title, likes, quality_probability, distance in rows
    ]
