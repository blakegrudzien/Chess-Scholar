"""Streamlit review tool for hand-labeling the study candidates collected by
collect_study_candidates.py.

Reads data/processed/study_candidates.jsonl, shows one not-yet-labeled study
at a time with a readable preview of its annotations, and appends each
decision as one JSON line to data/processed/study_labels.jsonl. The two
files are kept separate on purpose: candidates are re-collectible network
data, labels are irreplaceable human judgment, and a rerun of the collector
should never risk the labeling work already done.

Labeling resumes correctly across restarts because "the next study to show"
is always recomputed as "first candidate whose study_id isn't already in the
labels file" -- there's no separate progress counter that could drift out of
sync with what's actually been recorded.

Run as a module so the repo root lands on sys.path (same reason
collect_study_candidates.py does):
    .venv/bin/streamlit run scripts/label_studies_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.ingestion.annotation_extractor import extract_chapter_comment_text  # noqa: E402
from src.recommendation.study_pgn import iter_study_chapters  # noqa: E402

CANDIDATES_PATH = Path("data/processed/study_candidates.jsonl")
LABELS_PATH = Path("data/processed/study_labels.jsonl")

GENRE_OPTIONS = ["drill", "concept", "narrative"]
# Captured alongside genre, not as a genre of its own: the 3-way
# drill/concept/narrative split stays the user-facing recommendation
# taxonomy (unchanged), but within it an opening repertoire and an endgame
# study can have very different structural shapes (chapter count, comment
# length) despite both being "concept." Without this, the quality
# classifier (Step 5) has no way to tell "terse because it's an endgame
# study" from "terse because it's low effort" -- it only sees the 11
# structural numbers in feature_extraction.py, nothing chess-semantic, so
# it can only learn whatever correlation actually shows up in the labels.
PHASE_OPTIONS = ["opening", "middlegame", "endgame", "general"]
QUALITY_OPTIONS = ["recommend", "reject"]


@st.cache_data
def _load_candidates(path: str, mtime: float) -> list[dict]:
    """mtime is part of the cache key purely so this reloads automatically
    if collect_study_candidates.py writes more rows after the app has
    already cached a shorter version of the file -- Streamlit's cache
    otherwise has no way to know the file on disk changed underneath it.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_labeled_ids(path: Path) -> set[str]:
    """Not cached, unlike _load_candidates: this file changes on every
    label submitted during the session, and correctness here depends on
    seeing that change on the very next rerun.
    """
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["study_id"])
    return ids


def _append_label(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


@st.cache_data
def _build_comment_preview(pgn_text: str) -> str:
    """Parse every chapter in a study's PGN export and join each chapter's
    prose annotations under a heading, so a labeler can judge quality
    without leaving the app. Cached on the PGN text itself (not study_id)
    since that's the actual input the parse depends on.

    iter_study_chapters silently stops at the first chapter it can't parse
    rather than raising, so a malformed chapter just means a shorter
    preview here, not a crashed page.
    """
    sections = []
    for game in iter_study_chapters(pgn_text):
        title = game.headers.get("ChapterName") or game.headers.get("Event") or "Untitled"
        text = extract_chapter_comment_text(game).strip()
        sections.append(f"## {title}\n{text}" if text else f"## {title}\n_(no comments)_")
    return "\n\n".join(sections)


def main() -> None:
    st.set_page_config(page_title="Label Study Candidates", layout="wide")
    st.title("Label study candidates")

    if not CANDIDATES_PATH.exists():
        st.error(
            f"{CANDIDATES_PATH} not found. Run `python -m scripts.collect_study_candidates` first."
        )
        return

    candidates = _load_candidates(str(CANDIDATES_PATH), CANDIDATES_PATH.stat().st_mtime)
    labeled_ids = _load_labeled_ids(LABELS_PATH)
    remaining = [c for c in candidates if c["study_id"] not in labeled_ids]

    st.caption(f"{len(labeled_ids)} labeled, {len(remaining)} remaining of {len(candidates)} total")
    if candidates:
        st.progress(len(labeled_ids) / len(candidates))

    if not remaining:
        st.success("All candidates labeled.")
        return

    study = remaining[0]
    study_id = study["study_id"]

    col_meta, col_preview = st.columns([1, 2])

    with col_meta:
        st.subheader(study["title"])
        st.markdown(f"[Open in Lichess ↗](https://lichess.org/study/{study_id})")
        st.write(f"**Author:** {study['author']}")
        st.write(f"**Likes:** {study['likes']}")
        st.write(f"**Listing:** {study['source_sort']}")
        st.write(f"**Chapters ({len(study['chapter_titles'])}):**")
        for chapter_title in study["chapter_titles"]:
            st.write(f"- {chapter_title}")

    with col_preview:
        st.markdown("**Annotation preview**")
        preview = _build_comment_preview(study["pgn"])
        st.markdown(preview if preview.strip() else "_(no annotations found)_")

    st.divider()
    with st.form(key=f"label_form_{study_id}"):
        genre = st.radio("Genre", GENRE_OPTIONS, index=None, horizontal=True)
        phase = st.radio("Game phase", PHASE_OPTIONS, index=None, horizontal=True)
        quality = st.radio("Quality", QUALITY_OPTIONS, index=None, horizontal=True)
        note = st.text_area("Note (optional)")
        submitted = st.form_submit_button("Submit label")

    if submitted:
        if genre is None or phase is None or quality is None:
            st.warning("Pick a genre, game phase, and quality before submitting.")
        else:
            _append_label(
                LABELS_PATH,
                {
                    "study_id": study_id,
                    "genre": genre,
                    "phase": phase,
                    "quality": quality,
                    "note": note,
                    "skipped": False,
                    "labeled_at": datetime.now(UTC).isoformat(),
                },
            )
            st.rerun()

    if st.button("Skip (unsure)"):
        _append_label(
            LABELS_PATH,
            {"study_id": study_id, "skipped": True, "labeled_at": datetime.now(UTC).isoformat()},
        )
        st.rerun()


main()
