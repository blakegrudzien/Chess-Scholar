"""Cached, session-wide handles to the external resources the UI's tool
calls need -- one DB pool, one engine pool, one Voyage client, one Anthropic
client per Streamlit session, not one per request.
"""

from __future__ import annotations

import anthropic
import psycopg2.pool
import streamlit as st

from src.embeddings.voyage_embedder import get_voyage_client
from src.engine.engine_pool import EnginePool
from src.engine.stockfish_eval import get_engine_path
from src.ingestion.db_loader import get_connection_pool

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
