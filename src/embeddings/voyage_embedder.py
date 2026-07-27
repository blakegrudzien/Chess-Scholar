"""Layer 2 embeddings -- Voyage AI, Anthropic's recommended embeddings partner
(Anthropic has no first-party embedding model).

Only embeds chunks where chunks.embedding IS NULL, so re-running after a
partial run or a fresh ingestion batch never re-pays for already-embedded
rows -- the same idempotency principle as db_loader's ON CONFLICT inserts.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

import psycopg2
import psycopg2.extensions
import voyageai
import voyageai.error
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from src.ingestion.db_loader import get_connection

BATCH_SIZE = 128  # Voyage API max texts per embed() call
RETRY_BACKOFF_SECONDS = 5

Connect = Callable[[], psycopg2.extensions.connection]

# Transient Voyage errors worth retrying (as opposed to AuthenticationError,
# InvalidRequestError, MalformedRequestError, which won't fix themselves).
# Observed in practice: a 10-minute Timeout killed an otherwise-healthy run
# because only psycopg2.OperationalError was being retried.
VOYAGE_RETRYABLE_ERRORS = (
    voyageai.error.Timeout,
    voyageai.error.APIConnectionError,
    voyageai.error.ServerError,
    voyageai.error.ServiceUnavailableError,
    voyageai.error.TryAgain,
    voyageai.error.RateLimitError,
)


def get_voyage_client() -> voyageai.Client:
    load_dotenv()
    return voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])


def _fetch_unembedded_batch(
    conn: psycopg2.extensions.connection, limit: int
) -> list[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, text FROM chunks WHERE embedding IS NULL LIMIT %s",
            (limit,),
        )
        return cur.fetchall()


def embed_pending_chunks(
    connect: Connect,
    client: voyageai.Client,
    model: str = "voyage-4",
    max_consecutive_failures: int = 5,
) -> int:
    """Embed every chunk with a NULL embedding, batching up to BATCH_SIZE
    texts per Voyage API call. Returns the number of chunks embedded.

    Holds one connection across the whole job (reconnecting per batch was
    tried and made every batch ~3x slower -- Neon's per-connection setup
    cost dominates). On a psycopg2.OperationalError (observed in practice:
    Neon closed a long-lived idle connection mid-job), reconnect and retry.
    On a transient Voyage API error (observed in practice: a 10-minute read
    timeout), back off briefly and retry without touching the DB connection.
    Either way the failed batch was never committed, so it's safe to just
    loop again. Gives up after `max_consecutive_failures` in a row so a
    truly broken connection or API outage fails loudly instead of spinning
    forever.
    """
    total_embedded = 0
    consecutive_failures = 0
    conn = connect()
    register_vector(conn)

    try:
        while True:
            try:
                fetch_start = time.monotonic()
                batch = _fetch_unembedded_batch(conn, BATCH_SIZE)
                fetch_elapsed = time.monotonic() - fetch_start
                if not batch:
                    break

                chunk_ids = [chunk_id for chunk_id, _ in batch]
                texts = [text for _, text in batch]

                embed_start = time.monotonic()
                result = client.embed(texts, model=model, input_type="document")
                embed_elapsed = time.monotonic() - embed_start

                write_start = time.monotonic()
                rows = list(zip(chunk_ids, result.embeddings, strict=True))
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        UPDATE chunks AS c
                        SET embedding = data.embedding
                        FROM (VALUES %s) AS data(chunk_id, embedding)
                        WHERE c.chunk_id = data.chunk_id
                        """,
                        rows,
                        template="(%s, %s::vector)",
                        page_size=len(rows),
                    )
                conn.commit()
                write_elapsed = time.monotonic() - write_start

                total_embedded += len(batch)
                consecutive_failures = 0
                print(
                    f"batch={len(batch)} fetch={fetch_elapsed:.2f}s "
                    f"embed={embed_elapsed:.2f}s write={write_elapsed:.2f}s "
                    f"total={total_embedded}",
                    flush=True,
                )
            except psycopg2.OperationalError as exc:
                consecutive_failures += 1
                print(f"OperationalError (consecutive={consecutive_failures}): {exc}", flush=True)
                if consecutive_failures > max_consecutive_failures:
                    raise
                conn.close()
                conn = connect()
                register_vector(conn)
            except VOYAGE_RETRYABLE_ERRORS as exc:
                consecutive_failures += 1
                print(f"Voyage API error (consecutive={consecutive_failures}): {exc}", flush=True)
                if consecutive_failures > max_consecutive_failures:
                    raise
                time.sleep(RETRY_BACKOFF_SECONDS)
    finally:
        conn.close()

    return total_embedded


def _main() -> None:
    parser = argparse.ArgumentParser(description="Embed pending chunks via Voyage AI.")
    parser.add_argument("--model", default="voyage-4", help="Voyage embedding model")
    args = parser.parse_args()

    client = get_voyage_client()
    embedded = embed_pending_chunks(get_connection, client, model=args.model)
    print(f"Embedded {embedded} chunk(s) using {args.model}")


if __name__ == "__main__":
    _main()
