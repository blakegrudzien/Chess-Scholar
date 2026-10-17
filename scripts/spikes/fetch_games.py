"""Throwaway spike code for Phase 0 of the game report roadmap:
download a player's recent games from
Lichess or Chess.com. Phase 1 replaces this with a production fetcher
(rate limits, retries, caching)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import chess
import chess.pgn
import httpx

# Mate scores become this many centipawns, so swings and averages stay finite.
MATE_CP = 10000

# Games shorter than this (aborted, or resigned at once) say nothing useful.
MIN_PLIES = 10

USER_AGENT = "chess-scholar-spike/0.1 (github.com/blakegrudzien/Chess-Scholar)"

_CHESSCOM_DRAWS = {
    "agreed",
    "repetition",
    "stalemate",
    "insufficient",
    "50move",
    "timevsinsufficient",
}


class PlayerNotFound(Exception):
    pass


@dataclass
class SpikeGame:
    game_id: str
    user_color: chess.Color
    result: str  # "win", "loss" or "draw", from the user's side
    sans: list[str]
    # Lichess's own analysis, one entry per ply, centipawns from White's side;
    # None when the game was never analyzed on Lichess.
    white_evals: list[int | None] | None


def _lichess_eval(entry: dict) -> int | None:
    if "eval" in entry:
        return entry["eval"]
    if "mate" in entry:
        return MATE_CP if entry["mate"] > 0 else -MATE_CP
    return None


def parse_lichess_game(data: dict, username: str) -> SpikeGame | None:
    if data.get("variant") != "standard" or "initialFen" in data:
        return None
    sans = data.get("moves", "").split()
    if len(sans) < MIN_PLIES:
        return None
    white_name = data["players"]["white"].get("user", {}).get("name", "")
    user_color = chess.WHITE if white_name.lower() == username.lower() else chess.BLACK
    winner = data.get("winner")
    if winner is None:
        result = "draw"
    else:
        result = "win" if (winner == "white") == (user_color == chess.WHITE) else "loss"
    analysis = data.get("analysis")
    return SpikeGame(
        game_id=data["id"],
        user_color=user_color,
        result=result,
        sans=sans,
        white_evals=[_lichess_eval(e) for e in analysis] if analysis else None,
    )


def parse_chesscom_game(data: dict, username: str) -> SpikeGame | None:
    if data.get("rules") != "chess":
        return None
    game = chess.pgn.read_game(io.StringIO(data.get("pgn", "")))
    if game is None or game.headers.get("SetUp") == "1":
        return None
    board = game.board()
    sans = []
    for move in game.mainline_moves():
        sans.append(board.san(move))
        board.push(move)
    if len(sans) < MIN_PLIES:
        return None
    is_white = data["white"]["username"].lower() == username.lower()
    side = data["white"] if is_white else data["black"]
    outcome = side.get("result")
    if outcome == "win":
        result = "win"
    elif outcome in _CHESSCOM_DRAWS:
        result = "draw"
    else:
        result = "loss"
    return SpikeGame(
        game_id=data["url"],
        user_color=chess.WHITE if is_white else chess.BLACK,
        result=result,
        sans=sans,
        white_evals=None,
    )


def fetch_lichess_games(
    client: httpx.Client, username: str, max_games: int, perf_type: str
) -> list[SpikeGame]:
    response = client.get(
        f"https://lichess.org/api/games/user/{username}",
        params={"max": max_games, "perfType": perf_type, "evals": "true", "moves": "true"},
        headers={"Accept": "application/x-ndjson", "User-Agent": USER_AGENT},
        timeout=120,
    )
    if response.status_code == 404:
        raise PlayerNotFound(username)
    response.raise_for_status()
    games = []
    for line in response.text.splitlines():
        if line.strip():
            game = parse_lichess_game(json.loads(line), username)
            if game is not None:
                games.append(game)
    return games


def fetch_chesscom_games(
    client: httpx.Client, username: str, max_games: int, time_class: str
) -> list[SpikeGame]:
    headers = {"User-Agent": USER_AGENT}
    response = client.get(
        f"https://api.chess.com/pub/player/{username.lower()}/games/archives",
        headers=headers,
    )
    if response.status_code == 404:
        raise PlayerNotFound(username)
    response.raise_for_status()
    games: list[SpikeGame] = []
    # Monthly archives, oldest first; read newest first.
    for archive_url in reversed(response.json()["archives"]):
        month = client.get(archive_url, headers=headers)
        month.raise_for_status()
        for data in reversed(month.json()["games"]):
            if data.get("time_class") != time_class:
                continue
            game = parse_chesscom_game(data, username)
            if game is not None:
                games.append(game)
            if len(games) >= max_games:
                return games
    return games


def fetch_games(site: str, username: str, max_games: int, time_control: str) -> list[SpikeGame]:
    with httpx.Client(timeout=60) as client:
        if site == "lichess":
            return fetch_lichess_games(client, username, max_games, time_control)
        if site == "chesscom":
            return fetch_chesscom_games(client, username, max_games, time_control)
    raise ValueError(f"Unknown site: {site}")


def fens_after_each_ply(sans: list[str]) -> list[str]:
    board = chess.Board()
    fens = []
    for san in sans:
        board.push_san(san)
        fens.append(board.fen())
    return fens
