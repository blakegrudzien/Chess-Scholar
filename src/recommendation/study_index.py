"""The searchable pool of Lichess studies: which candidates may be
recommended, and the text embedded for each.

Studies are matched to questions by embedding similarity, as in Layer 2,
rather than sorted into genres by hand-tuned rules: structural signals
overlap too much (18 of 156 "concept" studies are single-chapter, like a
narrative one), and the text itself says what a study is about.

Eligibility is checked here, separately from the quality classifier: a
well-scored study can still be too obscure to recommend.
"""

from __future__ import annotations

from src.ingestion.annotation_extractor import extract_chapter_comment_text
from src.recommendation.quality_classifier import DEFAULT_QUALITY_THRESHOLD
from src.recommendation.study_pgn import iter_study_chapters

# A likes floor independent of the classifier: below it, almost nobody has
# vetted a study, however well it scores. It excludes 4 studies labeled
# "recommend" in training (10-16 likes); they still help train the model,
# but serving one is a different risk.
MIN_LIKES_FOR_RECOMMENDATION = 20

# Enough annotation text to capture what a study teaches, without one long
# chapter dominating the embedding.
MAX_ANNOTATION_CHARS_FOR_EMBEDDING = 2000


def is_eligible_for_recommendation(
    likes: int,
    quality_probability: float,
    *,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> bool:
    """Whether a candidate may be recommended: it must clear both the likes
    floor and the quality threshold."""
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
