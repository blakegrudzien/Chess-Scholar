"""Cached, process-wide handles to the external resources the UI's tool
calls need -- one DB pool, one engine pool, one Voyage client, one Anthropic
client, one Lichess HTTP client/pacer for the whole deployment, not one per
request. st.cache_resource caches per Python process, not per Streamlit
session -- every concurrent user of a given deployment shares these same
handles (ENGINE_POOL_SIZE's own comment already assumes this: it's sized to
total deployment throughput, not per-user).
"""

from __future__ import annotations

import anthropic
import httpx
import psycopg2.pool
import streamlit as st

from src.embeddings.voyage_embedder import get_voyage_client
from src.engine.engine_pool import EnginePool
from src.engine.stockfish_eval import get_engine_path
from src.ingestion.db_loader import get_connection_pool
from src.recommendation.lichess_client import RequestPacer, default_http_client

# CPU-bound: each concurrent evaluation pins a core for the search duration,
# so this is sized to the deployment's compute, not to how many users we'd
# like to serve. Bump alongside the hosting tier, not in isolation.
ENGINE_POOL_SIZE = 2


@st.cache_resource
def get_db_pool() -> psycopg2.pool.ThreadedConnectionPool:
    return get_connection_pool()


@st.cache_resource
def get_engine_pool() -> EnginePool:
    return EnginePool(get_engine_path(), size=ENGINE_POOL_SIZE)


@st.cache_resource
def get_voyage():
    return get_voyage_client()


@st.cache_resource
def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@st.cache_resource
def get_lichess_http_client() -> httpx.Client:
    return default_http_client()


@st.cache_resource
def get_lichess_pacer() -> RequestPacer:
    # One pacer shared by every live recommend_resources() call across
    # every session, not one per call -- otherwise REQUEST_PACING_SECONDS
    # is enforced within a single call's own back-to-back
    # get_lichess_study_chapters invocations at best, and not at all
    # across two different users' concurrent requests, which is exactly
    # the courtesy this pacer exists to provide (see RequestPacer's own
    # docstring: "regardless of which caller issues them").
    return RequestPacer()
