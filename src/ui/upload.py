"""The Analyze Your Game tab: upload a PGN, find similar corpus games."""

from __future__ import annotations

import itertools
import os
import tempfile

import streamlit as st

from src.ingestion.pgn_parser import parse_pgn
from src.ui.chat import ask_with_status, render_position_thumbnail


def render_pgn_upload_tab() -> None:
    st.caption(
        "This comparison is **illustrative, not authoritative** -- it matches on "
        "exact opening moves, not true positional similarity."
    )
    uploaded = st.file_uploader("Upload a PGN of your own game", type=["pgn"])
    if uploaded is None:
        return

    with tempfile.NamedTemporaryFile(suffix=".pgn", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        # Only the first game is ever used below, so pull at most two from
        # the generator: the first to use, and a second only to learn
        # whether there's more than one, without fully parsing an upload
        # (untrusted input) that could contain a large number of games.
        first_two_games = list(itertools.islice(parse_pgn(tmp_path, source="user_upload"), 2))
    finally:
        os.unlink(tmp_path)

    if not first_two_games:
        st.error("Couldn't find a game in that file.")
        return

    if len(first_two_games) > 1:
        st.info("This file has more than one game. Only the first is analyzed.")

    game = first_two_games[0]
    move_sans = [m.move_san for m in game.moves]
    st.write(f"Parsed **{game.white or '?'} vs {game.black or '?'}** ({len(move_sans)} plies)")
    preview = " ".join(move_sans[:20]) + (" ..." if len(move_sans) > 20 else "")
    st.code(preview, language=None)

    if st.button("Find similar games in the corpus", type="primary"):
        question = (
            "Here is a game I played, as a list of moves in order: "
            f"{', '.join(move_sans)}. Find similar games in the corpus and give me "
            "an illustrative comparison."
        )
        _, touched_fens = ask_with_status(question)
        for fen, label in touched_fens[:4]:
            render_position_thumbnail(fen, label)
