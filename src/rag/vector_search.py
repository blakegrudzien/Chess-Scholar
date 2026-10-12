"""Layer 2 search: embed a question with Voyage AI and rank chunks by
pgvector cosine distance. Embeddings are written by
src/embeddings/voyage_embedder.py.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import psycopg2.extensions
import voyageai
from pgvector.psycopg2 import register_vector

# Connections that already know the vector type. register_vector queries the
# database, and pooled connections are reused, so it runs once per connection.
_vector_registered: weakref.WeakSet = weakref.WeakSet()


def ensure_vector_type(conn: psycopg2.extensions.connection) -> None:
    if conn not in _vector_registered:
        register_vector(conn)
        _vector_registered.add(conn)


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
    ensure_vector_type(conn)
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
