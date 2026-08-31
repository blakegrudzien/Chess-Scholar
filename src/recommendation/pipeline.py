"""Study recommendation pipeline: given a chat question, decide whether any
external resources (a Lichess study chapter, a ChessBase master game) are
worth surfacing alongside the chat answer, and if so, which ones.

A separate small agent from src/agent/chess_agent.py, not a branch inside
it, and built the same way: native tool-calling decides everything, not
hand-rolled procedural logic. Concretely, this means no explicit "infer the
relevant ECO code" step written in Python -- the find_chessbase_game tool
takes eco_codes as a parameter and the model supplies them from its own
opening theory knowledge, exactly how get_eco_summary already works in the
main chat agent. Recommending nothing at all is a valid, expected outcome,
not an error case: forcing a fixed quota of resources per question would
produce worse results than trusting relevance, the same reasoning that
motivated dropping the earlier drill/concept genre-routing design in favor
of pure similarity search.

The two recommend_* tools are the loop's forced structured output: unlike
every other tool here, their return value isn't meant to inform a further
turn, it's how the orchestrating code learns what the model decided. Each
call is recorded via a closure-captured list rather than parsed from the
model's final text, the same reason the rest of this project prefers
native tool schemas over parsing free text.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import anthropic
import psycopg2.pool
import voyageai
from anthropic import beta_tool

from src.ingestion.db_loader import get_connection_with_timeout
from src.recommendation.lichess_client import study_chapter_embed_url
from src.recommendation.lichess_scraper import fetch_study_chapters
from src.recommendation.study_search import search_studies
from src.search.structured_search import (
    NarrativeGameCandidate,
    game_moves_as_pgn,
    select_narrative_game,
)

MODEL = "claude-sonnet-5"
MAX_TOKENS = 2048

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
resource that is specifically relevant to this question -- do not call \
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
    """Metadata gathered during one recommendation turn, keyed by id, so
    the final recommend_* tool calls only need to pass an id and a blurb --
    everything else needed to build the final Recommendation objects was
    already fetched by an earlier tool call in the same turn and doesn't
    need to be re-requested from the model or re-fetched from the DB/API.
    """

    studies_by_id: dict[str, str] = field(default_factory=dict)  # id -> title
    # study_id -> {chapter_id: name}
    chapters_by_study_id: dict[str, dict[str, str]] = field(default_factory=dict)
    games_by_id: dict[str, NarrativeGameCandidate] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)


def build_tools(
    db_pool: psycopg2.pool.ThreadedConnectionPool,
    voyage_client: voyageai.Client,
    state: _SessionState,
) -> list[Callable]:
    @beta_tool
    def search_lichess_studies(query: str) -> str:
        """Search the pool of quality-checked Lichess studies for ones
        semantically relevant to a topic.

        Args:
            query: Natural-language description of what to search for.
        """
        conn = get_connection_with_timeout(db_pool)
        try:
            results = search_studies(conn, voyage_client, query, limit=5)
        finally:
            db_pool.putconn(conn)
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
        chapters = fetch_study_chapters(study_id)
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
        conn = get_connection_with_timeout(db_pool)
        try:
            candidate = select_narrative_game(conn, eco_codes)
        finally:
            db_pool.putconn(conn)
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
                "Unknown study_id/chapter_id combination -- look it up with "
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
            return "Unknown game_id -- look it up with find_chessbase_game first."
        conn = get_connection_with_timeout(db_pool)
        try:
            pgn = game_moves_as_pgn(conn, game_id)
        finally:
            db_pool.putconn(conn)
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
) -> list[Recommendation]:
    """Decide which, if any, external resources to recommend alongside an
    answer to `question`. Returns an empty list if nothing was relevant
    enough to recommend -- a valid, expected outcome, not a failure.
    """
    client = client or anthropic.Anthropic()
    state = _SessionState()
    tools = build_tools(db_pool, voyage_client, state)

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=[{"role": "user", "content": question}],
    )
    for _ in runner:
        pass

    return state.recommendations
