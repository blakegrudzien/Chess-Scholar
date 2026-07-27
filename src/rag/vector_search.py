"""Layer 2 query-time semantic search: embed a natural-language query via
Voyage AI and rank chunks by pgvector cosine distance. The write-side of
this pipeline (populating chunks.embedding) lives in
src/embeddings/voyage_embedder.py -- this module is read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg2.extensions
import voyageai
from pgvector.psycopg2 import register_vector


@dataclass
class ChunkResult:
    text: str
    source_type: str
    game_id: str | None
    source_title: str | None
    author: str | None
    year: int | None
    eco_code: str | None
    distance: float


def search_chunks(
    conn: psycopg2.extensions.connection,
    client: voyageai.Client,
    query: str,
    model: str = "voyage-4",
    limit: int = 5,
) -> list[ChunkResult]:
    """Return the `limit` chunks most semantically similar to `query`."""
    register_vector(conn)
    result = client.embed([query], model=model, input_type="query")
    query_vector = result.embeddings[0]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT text, source_type, game_id, source_title, author, year, eco_code,
                   embedding <=> %s::vector AS distance
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, limit),
        )
        rows = cur.fetchall()

    return [
        ChunkResult(
            text=text,
            source_type=source_type,
            game_id=game_id,
            source_title=source_title,
            author=author,
            year=year,
            eco_code=eco_code,
            distance=distance,
        )
        for text, source_type, game_id, source_title, author, year, eco_code, distance in rows
    ]
