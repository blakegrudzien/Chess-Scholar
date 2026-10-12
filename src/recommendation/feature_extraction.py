"""Turn one collected study candidate into a fixed-size numeric feature
vector for the quality classifier.

Only structural and text-statistics features from the study's PGN and
listing data: a study's quality is scored once, independent of any
question, while relevance to a question is judged later by embedding
similarity.

A handful of interpretable numbers rather than TF-IDF or embeddings,
because a couple of hundred hand-labeled examples can't support a
high-dimensional model without it memorizing noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion.annotation_extractor import iter_plies_with_comments, strip_computer_glyphs
from src.recommendation.study_pgn import iter_study_chapters

_WORD_RE = re.compile(r"\S+")

# The column order training and inference share; everything that builds a
# feature vector follows this list.
FEATURE_NAMES = [
    "num_chapters",
    "total_plies",
    "num_annotated_plies",
    "comment_density",
    "total_comment_word_count",
    "avg_comment_word_count",
    "has_pre_game_comment",
    "num_nag_annotations",
    "likes",
    "num_members",
    "title_length",
]


@dataclass(frozen=True)
class StudyFeatures:
    study_id: str
    num_chapters: int
    total_plies: int
    num_annotated_plies: int
    comment_density: float
    total_comment_word_count: int
    avg_comment_word_count: float
    has_pre_game_comment: bool
    num_nag_annotations: int
    likes: int
    num_members: int
    title_length: int

    def as_vector(self) -> list[float]:
        """Feature values as floats, in FEATURE_NAMES order."""
        return [
            float(self.num_chapters),
            float(self.total_plies),
            float(self.num_annotated_plies),
            self.comment_density,
            float(self.total_comment_word_count),
            self.avg_comment_word_count,
            float(self.has_pre_game_comment),
            float(self.num_nag_annotations),
            float(self.likes),
            float(self.num_members),
            float(self.title_length),
        ]


def extract_features(candidate: dict) -> StudyFeatures:
    """candidate is one record as written by collect_study_candidates.py:
    {study_id, title, author, likes, updated_at, chapter_titles,
    member_usernames, source_sort, pgn, ...}.
    """
    num_chapters = 0
    total_plies = 0
    num_annotated_plies = 0
    total_comment_words = 0
    has_pre_game_comment = False
    num_nag_annotations = 0

    for game in iter_study_chapters(candidate["pgn"]):
        num_chapters += 1

        pre_game_comment = strip_computer_glyphs(game.comment)
        if pre_game_comment:
            has_pre_game_comment = True
            total_comment_words += len(_WORD_RE.findall(pre_game_comment))

        for _ply, _move_san, comment, nags in iter_plies_with_comments(game):
            total_plies += 1
            if comment:
                num_annotated_plies += 1
                total_comment_words += len(_WORD_RE.findall(comment))
            if nags:
                num_nag_annotations += 1

    comment_density = num_annotated_plies / total_plies if total_plies else 0.0
    avg_comment_word_count = (
        total_comment_words / num_annotated_plies if num_annotated_plies else 0.0
    )

    return StudyFeatures(
        study_id=candidate["study_id"],
        num_chapters=num_chapters,
        total_plies=total_plies,
        num_annotated_plies=num_annotated_plies,
        comment_density=comment_density,
        total_comment_word_count=total_comment_words,
        avg_comment_word_count=avg_comment_word_count,
        has_pre_game_comment=has_pre_game_comment,
        num_nag_annotations=num_nag_annotations,
        likes=candidate["likes"],
        num_members=len(candidate["member_usernames"]),
        title_length=len(candidate["title"]),
    )
