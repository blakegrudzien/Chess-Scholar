"""Builds the searchable pool of recommendable Lichess studies: which
candidates are eligible to ever be surfaced, and what text represents each
one for embedding similarity search.

Genre routing (originally planned as a separate heuristic step) was
dropped in favor of this simpler design: rather than pre-classify every
candidate as drill/concept/narrative with hand-tuned thresholds on noisy
structural signals (real overlap exists -- e.g. 18 of 156 labeled "concept"
studies are also single-chapter, the same shape a "narrative" study has),
this reuses the project's existing Layer 2 pattern -- embed content, do
similarity search against a question, let the LLM read what actually came
back. The prose itself carries the genre distinction far more reliably
than any cheap proxy a heuristic could compute from it.

Eligibility is a separate concern from the quality classifier's prediction
-- deliberately. A candidate can score well on predicted quality and still
be unfit to recommend if almost nobody on Lichess has seen it. This module
is the one place a candidate can fail to be recommendable for reasons
other than "the classifier didn't like it."
"""

from __future__ import annotations

from src.ingestion.annotation_extractor import extract_chapter_comment_text
from src.recommendation.study_pgn import iter_study_chapters

# A hard floor independent of the quality classifier's own judgment: even a
# candidate the model scores well is an unvetted, essentially unknown
# resource below this many likes -- almost nobody has seen it, so
# recommending it is a different kind of risk than the classifier's
# accuracy addresses. Set at 20 knowing this excludes 4 studies already
# hand-labeled "recommend" in the training data (10, 12, 14, 16 likes) --
# a deliberate choice: what's good enough to keep in *training*
# (more labeled examples, including low-like ones, made the classifier
# more honest) doesn't have to be good enough to *serve*, since serving
# carries a different risk than training does.
MIN_LIKES_FOR_RECOMMENDATION = 20

# The classifier's own default decision boundary (matches how
# quality_classifier.compute_metrics evaluates precision/recall/F1), not
# independently tuned -- revisit together with those metrics if this ever
# needs to move.
DEFAULT_QUALITY_THRESHOLD = 0.5

# Annotation text is embedded up to this many characters -- long enough to
# capture what a study actually teaches, short enough to keep embedding
# calls cheap and avoid diluting the vector with a full multi-chapter essay
# dominated by whichever chapter happens to be longest.
MAX_ANNOTATION_CHARS_FOR_EMBEDDING = 2000


def is_eligible_for_recommendation(
    likes: int,
    quality_probability: float,
    *,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> bool:
    """Whether a candidate may ever be surfaced as a recommendation --
    both the likes floor and the quality bar must clear, independently.
    """
    return likes >= MIN_LIKES_FOR_RECOMMENDATION and quality_probability >= quality_threshold


def build_embedding_text(candidate: dict) -> str:
    """Text representing one candidate for embedding similarity search:
    title, chapter titles (cheap, always available, often the clearest
    single signal of topic), and a bounded sample of the actual annotation
    prose (what the study actually explains, not just names).

    candidate is one record as written by collect_study_candidates.py:
    {study_id, title, chapter_titles, pgn, ...}.
    """
    parts = [candidate["title"], *candidate["chapter_titles"]]

    annotation_chars: list[str] = []
    remaining = MAX_ANNOTATION_CHARS_FOR_EMBEDDING
    for game in iter_study_chapters(candidate["pgn"]):
        if remaining <= 0:
            break
        comment = extract_chapter_comment_text(game)
        if not comment:
            continue
        take = comment[:remaining]
        annotation_chars.append(take)
        remaining -= len(take)

    if annotation_chars:
        parts.append(" ".join(annotation_chars))

    return "\n".join(parts)
