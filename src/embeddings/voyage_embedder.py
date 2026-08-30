"""Layer 2 embeddings -- Voyage AI, Anthropic's recommended embeddings partner
(Anthropic has no first-party embedding model).

Only embeds chunks where chunks.embedding IS NULL, so re-running after a
partial run or a fresh ingestion batch never re-pays for already-embedded
rows -- the same idempotency principle as db_loader's ON CONFLICT inserts.
"""

from __future__ import annotations

import argparse
import logging
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

logger = logging.getLogger(__name__)

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
    # override=True: see get_connection in db_loader.py for why .env must
    # win over whatever's already in os.environ (Streamlit's auto-loaded
    # .streamlit/secrets.toml, in particular).
    load_dotenv(override=True)
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
    cost dominates). On a dropped connection (observed in practice: Neon
    closed a long-lived idle connection mid-job, raised as either
    OperationalError or InterfaceError depending on which the client
    detected it), reconnect and retry. On a transient Voyage API error
    (observed in practice: a 10-minute read timeout), back off briefly and
    retry without touching the DB connection.
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
                logger.info(
                    "batch=%d fetch=%.2fs embed=%.2fs write=%.2fs total=%d",
                    len(batch),
                    fetch_elapsed,
                    embed_elapsed,
                    write_elapsed,
                    total_embedded,
                )
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                consecutive_failures += 1
                logger.warning(
                    "%s (consecutive=%d): %s", type(exc).__name__, consecutive_failures, exc
                )
                if consecutive_failures > max_consecutive_failures:
                    raise
                conn.close()
                conn = connect()
                register_vector(conn)
            except VOYAGE_RETRYABLE_ERRORS as exc:
                consecutive_failures += 1
                logger.warning("Voyage API error (consecutive=%d): %s", consecutive_failures, exc)
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
    logger.info("Embedded %d chunk(s) using %s", embedded, args.model)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _main()
