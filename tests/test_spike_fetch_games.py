# tests/test_spike_fetch_games.py
import json

import chess
import httpx
import pytest

from scripts.spikes.fetch_games import (
    MATE_CP,
    USER_AGENT,
    PlayerNotFound,
    fens_after_each_ply,
    fetch_chesscom_games,
    fetch_lichess_games,
    parse_chesscom_game,
    parse_lichess_game,
)

TWELVE_PLIES = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5"


def _lichess(**overrides):
    data = {
        "id": "abc123",
        "variant": "standard",
        "players": {"white": {"user": {"name": "Alice"}}, "black": {"user": {"name": "Bob"}}},
        "winner": "white",
        "moves": TWELVE_PLIES,
    }
    data.update(overrides)
    return data


def test_lichess_game_from_the_users_side():
    game = parse_lichess_game(_lichess(), "alice")
    assert game.user_color == chess.WHITE
    assert game.result == "win"
    assert len(game.sans) == 12
    assert game.white_evals is None


def test_lichess_loss_and_draw():
    assert parse_lichess_game(_lichess(), "bob").result == "loss"
    assert parse_lichess_game(_lichess(winner=None), "bob").result == "draw"


def test_lichess_analysis_maps_mate_to_a_large_value():
    analysis = [{"eval": 20}, {"mate": 3}, {"mate": -2}, {}]
    game = parse_lichess_game(_lichess(analysis=analysis), "alice")
    assert game.white_evals == [20, MATE_CP, -MATE_CP, None]


def test_lichess_skips_variants_custom_starts_and_short_games():
    assert parse_lichess_game(_lichess(variant="chess960"), "alice") is None
    assert parse_lichess_game(_lichess(variant="fromPosition"), "alice") is None
    assert parse_lichess_game(_lichess(initialFen="8/8/8/8/8/8/8/K6k w - - 0 1"), "alice") is None
    assert parse_lichess_game(_lichess(moves="e4 e5 Qh5"), "alice") is None


def _pgn(moves=TWELVE_PLIES, setup=False):
    headers = '[White "Alice"]\n[Black "Bob"]\n'
    if setup:
        headers += '[SetUp "1"]\n[FEN "8/8/8/8/8/8/8/K6k w - - 0 1"]\n'
    numbered = []
    for i, san in enumerate(moves.split()):
        numbered.append(f"{i // 2 + 1}. {san}" if i % 2 == 0 else san)
    return headers + "\n" + " ".join(numbered) + " 1-0\n"


def _chesscom(**overrides):
    data = {
        "url": "https://www.chess.com/game/live/1",
        "rules": "chess",
        "time_class": "blitz",
        "pgn": _pgn(),
        "white": {"username": "Alice", "result": "win"},
        "black": {"username": "Bob", "result": "resigned"},
    }
    data.update(overrides)
    return data


def test_chesscom_game_from_the_users_side():
    game = parse_chesscom_game(_chesscom(), "bob")
    assert game.user_color == chess.BLACK
    assert game.result == "loss"
    assert game.sans[:2] == ["e4", "e5"]


def test_chesscom_draw_result():
    data = _chesscom(
        white={"username": "Alice", "result": "repetition"},
        black={"username": "Bob", "result": "repetition"},
    )
    assert parse_chesscom_game(data, "alice").result == "draw"


def test_chesscom_skips_variants_custom_starts_and_short_games():
    assert parse_chesscom_game(_chesscom(rules="chess960"), "alice") is None
    assert parse_chesscom_game(_chesscom(pgn=_pgn(setup=True)), "alice") is None
    assert parse_chesscom_game(_chesscom(pgn=_pgn(moves="e4 e5 Qh5")), "alice") is None


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_unknown_lichess_player_raises_player_not_found():
    client = _client(lambda request: httpx.Response(404))
    with pytest.raises(PlayerNotFound):
        fetch_lichess_games(client, "nobody", 10, "blitz")


def test_lichess_fetch_parses_each_ndjson_line():
    body = "\n".join(json.dumps(_lichess(id=str(i))) for i in range(3)) + "\n"
    client = _client(lambda request: httpx.Response(200, text=body))
    games = fetch_lichess_games(client, "alice", 10, "blitz")
    assert [g.game_id for g in games] == ["0", "1", "2"]


def test_lichess_fetch_sends_a_descriptive_user_agent():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text="")

    games = fetch_lichess_games(_client(handler), "alice", 10, "blitz")
    assert games == []
    assert seen[0].headers["User-Agent"] == USER_AGENT


def test_unknown_chesscom_player_raises_player_not_found():
    client = _client(lambda request: httpx.Response(404))
    with pytest.raises(PlayerNotFound):
        fetch_chesscom_games(client, "nobody", 10, "blitz")


def test_chesscom_fetch_filters_time_class_and_stops_at_max_games():
    archive_url = "https://api.chess.com/pub/player/alice/games/2026/10"
    month = {
        "games": [
            _chesscom(url="g1"),
            _chesscom(url="g2", time_class="rapid"),
            _chesscom(url="g3"),
            _chesscom(url="g4"),
        ]
    }

    def handler(request):
        if request.url.path.endswith("/archives"):
            return httpx.Response(200, json={"archives": [archive_url]})
        return httpx.Response(200, json=month)

    games = fetch_chesscom_games(_client(handler), "alice", 2, "blitz")
    # Newest games first, rapid skipped.
    assert [g.game_id for g in games] == ["g4", "g3"]


def test_fens_after_each_ply():
    fens = fens_after_each_ply(["e4", "e5"])
    assert len(fens) == 2
    assert fens[0].startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b")
