"""Resource recommendations: decide whether a Lichess study chapter or a
ChessBase master game is worth showing alongside a chat answer, and which.

A small agent of its own, separate from the chat agent and built the same
way: the model decides through tool calls. It supplies ECO codes to
find_chessbase_game from its own opening knowledge, as it does for
get_eco_summary in the chat agent. Recommending nothing is a normal
outcome; a fixed quota would surface weaker matches.

The recommend_* tools are how the model reports its picks: each call is
recorded in a shared list rather than parsed out of free text.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import anthropic
import httpx
import psycopg2.pool
import voyageai
from anthropic import beta_tool

from src.config import CLAUDE_MODEL
from src.ingestion.db_loader import query_with_retry
from src.recommendation.lichess_client import RequestPacer, study_chapter_embed_url
from src.recommendation.lichess_scraper import fetch_study_chapters
from src.recommendation.study_search import search_studies
from src.search.structured_search import (
    NarrativeGameCandidate,
    game_moves_as_pgn,
    select_narrative_game,
)

logger = logging.getLogger(__name__)

MAX_TOKENS = 2048

# Caps tool-calling turns per lookup (each turn is a paid call). The
# shortest useful lookup is five turns (search, list chapters, recommend,
# find a game, recommend), and the model usually lists chapters for several
# studies first, so the cap leaves room for that; running out returns
# nothing at all.
MAX_AGENT_TURNS = 14

SYSTEM_PROMPT = """You help decide whether any external resources are worth \
recommending alongside an answer to a chess question.

Tools:
- search_lichess_studies: semantic search over a pool of quality-checked, \
community-vetted Lichess studies. Use this first for most questions.
- get_lichess_study_chapters: once a study from search_lichess_studies looks \
worth recommending, call this to see its chapters and pick the single most \
relevant one.
- find_chessbase_game: finds one real grandmaster game (moves only, no \
annotations) from the local corpus, matched by ECO opening code. Supply the \
ECO code(s) yourself from your own opening theory knowledge, the same way \
you would for a structured-search tool. Use this when a concrete illustrative \
game would help, not for every question.
- recommend_lichess_study / recommend_chessbase_game: call one of these for \
each resource you've decided is genuinely worth showing, with a one or two \
sentence blurb explaining why it's relevant to the question.

Recommending nothing is a valid and often correct outcome. Only recommend a \
resource that is specifically relevant to this question; do not call \
recommend_* just to fill a quota. Never recommend a study or game you have \
not looked up with a tool in this turn.
"""


@dataclass(frozen=True)
class LichessStudyRecommendation:
    study_id: str
    study_title: str
    chapter_id: str
    chapter_name: str
    embed_url: str
    blurb: str


@dataclass(frozen=True)
class ChessbaseGameRecommendation:
    game_id: str
    white: str
    black: str
    event: str
    pgn: str
    blurb: str


Recommendation = LichessStudyRecommendation | ChessbaseGameRecommendation


@dataclass
class _SessionState:
    """What the lookup's tools have fetched, keyed by id, so a recommend_*
    call only needs an id and a blurb."""

    studies_by_id: dict[str, str] = field(default_factory=dict)  # id -> title
    # study_id -> {chapter_id: name}
    chapters_by_study_id: dict[str, dict[str, str]] = field(default_factory=dict)
    games_by_id: dict[str, NarrativeGameCandidate] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)


def build_tools(
    db_pool: psycopg2.pool.ThreadedConnectionPool,
    voyage_client: voyageai.Client,
    state: _SessionState,
    http_client: httpx.Client | None = None,
    pacer: RequestPacer | None = None,
) -> list[Callable]:
    @beta_tool
    def search_lichess_studies(query: str) -> str:
        """Search the pool of quality-checked Lichess studies for ones
        semantically relevant to a topic.

        Args:
            query: Natural-language description of what to search for.
        """
        results = query_with_retry(db_pool, search_studies, voyage_client, query, limit=5)
        if not results:
            return "No matching studies found."
        for r in results:
            state.studies_by_id[r.study_id] = r.title
        return "\n".join(
            f'{r.study_id}: "{r.title}" ({r.likes} likes, '
            f"quality score {r.quality_probability:.2f})"
            for r in results
        )

    @beta_tool
    def get_lichess_study_chapters(study_id: str) -> str:
        """List the chapters of a Lichess study, so a specific one can be
        picked to recommend.

        Args:
            study_id: A study id from search_lichess_studies results.
        """
        # The shared client and pacer from recommend_resources, so requests
        # to Lichess are spaced out across every visitor's lookups.
        chapters = fetch_study_chapters(study_id, http_client=http_client, pacer=pacer)
        if not chapters:
            return f"No chapters found for study {study_id}."
        state.chapters_by_study_id[study_id] = {c.chapter_id: c.name for c in chapters}
        return "\n".join(f'{c.chapter_id}: "{c.name}"' for c in chapters)

    @beta_tool
    def find_chessbase_game(eco_codes: list[str]) -> str:
        """Find one real grandmaster game from the local ChessBase corpus
        matching a chess opening, to recommend as moves only (no
        annotations). Prefers a game the corpus originally annotated
        heavily, as a notability signal.

        Args:
            eco_codes: One or more ECO opening codes to match, e.g. ["B70", "B71"].
        """
        candidate = query_with_retry(db_pool, select_narrative_game, eco_codes)
        if candidate is None:
            return f"No chessbase game found for ECO code(s) {eco_codes}."
        state.games_by_id[candidate.game_id] = candidate
        return (
            f"{candidate.game_id}: {candidate.white} vs {candidate.black} "
            f"({candidate.event}, {candidate.year}, {candidate.eco_code}, "
            f"{candidate.result}), originally annotated with "
            f"{candidate.annotation_chunk_count} comments."
        )

    @beta_tool
    def recommend_lichess_study(study_id: str, chapter_id: str, blurb: str) -> str:
        """Recommend one Lichess study chapter to the user.

        Args:
            study_id: A study id already looked up with search_lichess_studies.
            chapter_id: A chapter id already looked up with get_lichess_study_chapters
                for this exact study_id.
            blurb: One or two sentences explaining why this is relevant to the question.
        """
        title = state.studies_by_id.get(study_id)
        chapter_name = state.chapters_by_study_id.get(study_id, {}).get(chapter_id)
        if title is None or chapter_name is None:
            return (
                "Unknown study_id/chapter_id combination. Look it up with "
                "search_lichess_studies and get_lichess_study_chapters first."
            )
        state.recommendations.append(
            LichessStudyRecommendation(
                study_id=study_id,
                study_title=title,
                chapter_id=chapter_id,
                chapter_name=chapter_name,
                embed_url=study_chapter_embed_url(study_id, chapter_id),
                blurb=blurb,
            )
        )
        return "Recorded."

    @beta_tool
    def recommend_chessbase_game(game_id: str, blurb: str) -> str:
        """Recommend one ChessBase master game to the user.

        Args:
            game_id: A game id already looked up with find_chessbase_game.
            blurb: One or two sentences explaining why this is relevant to the question.
        """
        candidate = state.games_by_id.get(game_id)
        if candidate is None:
            return "Unknown game_id. Look it up with find_chessbase_game first."
        pgn = query_with_retry(db_pool, game_moves_as_pgn, game_id)
        if pgn is None:
            return f"Could not load moves for game {game_id}."
        state.recommendations.append(
            ChessbaseGameRecommendation(
                game_id=game_id,
                white=candidate.white or "?",
                black=candidate.black or "?",
                event=candidate.event or "?",
                pgn=pgn,
                blurb=blurb,
            )
        )
        return "Recorded."

    return [
        search_lichess_studies,
        get_lichess_study_chapters,
        find_chessbase_game,
        recommend_lichess_study,
        recommend_chessbase_game,
    ]


def recommend_resources(
    question: str,
    db_pool: psycopg2.pool.ThreadedConnectionPool,
    voyage_client: voyageai.Client,
    client: anthropic.Anthropic | None = None,
    http_client: httpx.Client | None = None,
    pacer: RequestPacer | None = None,
) -> list[Recommendation]:
    """Decide which resources, if any, to recommend alongside an answer to
    `question`. An empty list means nothing was relevant enough.

    The clients and pacer should be the caller's long-lived shared instances
    (see src.ui.resources), so Lichess request pacing holds across calls.
    """
    client = client or anthropic.Anthropic()
    state = _SessionState()
    tools = build_tools(db_pool, voyage_client, state, http_client=http_client, pacer=pacer)

    runner = client.beta.messages.tool_runner(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=[{"role": "user", "content": question}],
        max_iterations=MAX_AGENT_TURNS,
    )
    # The recommendations are collected in `state` by the tools themselves,
    # so the loop only needs to drive the runner to completion.
    turns = 0
    for _ in runner:
        turns += 1

    # An empty result usually means no close match, but it is also what a
    # lookup cut off by the turn cap returns. Logged to tell the two apart.
    if turns >= MAX_AGENT_TURNS and not state.recommendations:
        logger.warning(
            "Recommendation lookup hit the %d-turn ceiling with nothing recorded: "
            "the model ran out of turns while searching rather than finding no match.",
            MAX_AGENT_TURNS,
        )
    logger.info(
        "recommend_resources: %d turn(s), %d recommendation(s)",
        turns,
        len(state.recommendations),
    )

    return state.recommendations
