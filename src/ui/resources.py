"""Cached, process-wide handles to the external resources the UI's tool
calls need -- one DB pool, one engine pool, one Voyage client, one Anthropic
client, one Lichess HTTP client/pacer for the whole deployment, not one per
request. st.cache_resource caches per Python process, not per Streamlit
session -- every concurrent user of a given deployment shares these same
handles (ENGINE_POOL_SIZE's own comment already assumes this: it's sized to
total deployment throughput, not per-user).
"""

from __future__ import annotations

import logging
import threading

import anthropic
import httpx
import psycopg2.pool
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from src.embeddings.voyage_embedder import get_voyage_client
from src.engine.engine_pool import EnginePool
from src.engine.stockfish_eval import get_engine_path
from src.ingestion.db_loader import get_connection_pool, ping_database
from src.recommendation.lichess_client import RequestPacer, default_http_client

logger = logging.getLogger(__name__)

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


# Neon suspends after 5 minutes without queries. Pinging at most this long
# apart keeps it awake for as long as someone is using the app, so nobody
# waits on a cold start mid-session.
KEEP_WARM_PING_INTERVAL_SECONDS = 270

# Bounds the cost of a tab left open: past this long without an interaction,
# pings stop and Neon is allowed to suspend.
KEEP_WARM_INACTIVITY_LIMIT_SECONDS = 30 * 60


def session_inactive(now: float, last_interaction: float) -> bool:
    return now - last_interaction > KEEP_WARM_INACTIVITY_LIMIT_SECONDS


def keep_warm_ping_due(now: float, last_ping: float | None, last_interaction: float) -> bool:
    """Whether a session should ping the database now. A session's first
    check always pings, which starts waking Neon as soon as the page loads.
    """
    if session_inactive(now, last_interaction):
        return False
    return last_ping is None or now - last_ping >= KEEP_WARM_PING_INTERVAL_SECONDS


def ping_database_in_background() -> None:
    """Ping on a daemon thread so the page never waits on Neon waking up.
    Creating the pool happens on that thread too when it doesn't exist yet,
    since opening its first connections is itself a wait on Neon.
    """
    thread = threading.Thread(target=_ping_database, name="db-keep-warm", daemon=True)
    # Lets st.cache_resource inside get_db_pool() see the session it serves;
    # without it, every ping logs a "missing ScriptRunContext" warning.
    add_script_run_ctx(thread, get_script_run_ctx())
    thread.start()


def _ping_database() -> None:
    try:
        ping_database(get_db_pool())
    except Exception:
        # A failed ping only costs the warm start it was meant to provide;
        # the next real query retries on its own. Logged, never raised,
        # since nothing on this background thread could handle it.
        logger.warning("Background database ping failed", exc_info=True)
