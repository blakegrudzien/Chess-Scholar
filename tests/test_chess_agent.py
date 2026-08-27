from unittest.mock import MagicMock, patch

import psycopg2

from src.agent.chess_agent import ask, build_tools
from src.engine.engine_pool import EngineBusyError
from src.engine.stockfish_eval import PositionEval
from src.personalization.similarity import SimilarGame
from src.rag.vector_search import ChunkResult
from src.search.structured_search import EcoSummary, MoveFrequency, SquareFrequency


def _build_tools():
    db_pool = MagicMock()
    conn = MagicMock()
    db_pool.getconn.return_value = conn
    engine_pool = MagicMock()
    engine = MagicMock()
    engine_pool.checkout.return_value.__enter__.return_value = engine
    voyage_client = MagicMock()
    tools = {t.name: t for t in build_tools(db_pool, engine_pool, voyage_client)}
    return tools, db_pool, conn, engine_pool, engine, voyage_client


def test_get_eco_summary_formats_stats():
    tools, db_pool, conn, _, _, _ = _build_tools()
    summary = EcoSummary(
        eco_code="C50", game_count=10, white_wins=4, black_wins=3, draws=3, avg_ply_count=42.5
    )
    with patch("src.agent.chess_agent.eco_summary", return_value=summary) as mock_fn:
        result = tools["get_eco_summary"]("C50")

    mock_fn.assert_called_once_with(conn, "C50")
    db_pool.putconn.assert_called_once_with(conn)
    assert "10 games" in result
    assert "White wins 4" in result
    assert "42.5 plies" in result


def test_get_eco_summary_reconnects_after_dropped_connection():
    """tool_runner swallows exceptions a tool raises (turns them into a
    tool_result error) rather than letting them propagate -- so a dropped
    connection has to be discarded and replaced inside the tool call itself,
    not by whatever calls ask(). This confirms the pool does that: the dead
    connection is returned with close=True and a fresh one is checked out
    for a retry within the same call.
    """
    db_pool = MagicMock()
    dead_conn = MagicMock()
    fresh_conn = MagicMock()
    db_pool.getconn.side_effect = [dead_conn, fresh_conn]
    engine_pool = MagicMock()
    voyage_client = MagicMock()
    tools = {t.name: t for t in build_tools(db_pool, engine_pool, voyage_client)}

    summary = EcoSummary(
        eco_code="C50", game_count=1, white_wins=1, black_wins=0, draws=0, avg_ply_count=40.0
    )
    with patch(
        "src.agent.chess_agent.eco_summary",
        side_effect=[psycopg2.InterfaceError("connection already closed"), summary],
    ) as mock_fn:
        result = tools["get_eco_summary"]("C50")

    assert mock_fn.call_args_list[0].args[0] is dead_conn
    assert mock_fn.call_args_list[1].args[0] is fresh_conn
    db_pool.putconn.assert_any_call(dead_conn, close=True)
    db_pool.putconn.assert_any_call(fresh_conn)
    assert "1 games" in result


def test_get_eco_summary_reports_no_games():
    tools, _, _, _, _, _ = _build_tools()
    empty = EcoSummary(
        eco_code="Z99", game_count=0, white_wins=0, black_wins=0, draws=0, avg_ply_count=None
    )
    with patch("src.agent.chess_agent.eco_summary", return_value=empty):
        result = tools["get_eco_summary"]("Z99")

    assert "No games found" in result


def test_get_piece_placement_formats_squares():
    tools, _, conn, _, _, _ = _build_tools()
    freqs = [SquareFrequency("f6", 357), SquareFrequency("f3", 352)]
    with patch("src.agent.chess_agent.piece_placement_frequency", return_value=freqs) as mock_fn:
        result = tools["get_piece_placement"]("D12", "N", color="both", max_ply=20)

    mock_fn.assert_called_once_with(conn, "D12", "N", color="both", max_ply=20)
    assert "f6 (357x)" in result
    assert "f3 (352x)" in result


def test_get_piece_placement_surfaces_invalid_input():
    tools, _, _, _, _, _ = _build_tools()
    with patch(
        "src.agent.chess_agent.piece_placement_frequency", side_effect=ValueError("bad piece")
    ):
        result = tools["get_piece_placement"]("D12", "X")

    assert "Invalid input" in result
    assert "bad piece" in result


def test_get_common_moves_at_ply_formats_moves():
    tools, _, conn, _, _, _ = _build_tools()
    freqs = [MoveFrequency("Nf3", 207), MoveFrequency("e3", 64)]
    with patch("src.agent.chess_agent.common_moves_at_ply", return_value=freqs) as mock_fn:
        result = tools["get_common_moves_at_ply"]("D12", 5, limit=5)

    mock_fn.assert_called_once_with(conn, "D12", 5, limit=5)
    assert "Nf3 (207x)" in result


def test_search_annotations_formats_bullets():
    tools, _, conn, _, _, voyage_client = _build_tools()
    chunks = [
        ChunkResult(
            text="A sharp pawn grab.",
            source_type="game_annotation",
            game_id="abc",
            source_title=None,
            author=None,
            year=2021,
            eco_code="C51",
            distance=0.1,
        )
    ]
    with patch("src.agent.chess_agent.search_chunks", return_value=chunks) as mock_fn:
        result = tools["search_annotations"]("gambit", limit=5)

    mock_fn.assert_called_once_with(conn, voyage_client, "gambit", limit=5)
    assert "A sharp pawn grab." in result


def test_search_annotations_reports_no_results():
    tools, _, _, _, _, _ = _build_tools()
    with patch("src.agent.chess_agent.search_chunks", return_value=[]):
        result = tools["search_annotations"]("nonexistent concept")

    assert "No relevant annotations found" in result


def test_evaluate_chess_position_reports_score():
    tools, _, _, _, engine, _ = _build_tools()
    position_eval = PositionEval(
        fen="startpos", score_cp=34, mate_in=None, best_move_san="Nh4", pv_san=["Nh4", "Bg6"]
    )
    with patch("src.agent.chess_agent.evaluate_position", return_value=position_eval) as mock_fn:
        result = tools["evaluate_chess_position"](
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", depth=16
        )

    assert mock_fn.call_args.args[0] is engine
    assert "34 centipawns" in result
    assert "Nh4" in result


def test_evaluate_chess_position_reports_mate():
    tools, _, _, _, _, _ = _build_tools()
    position_eval = PositionEval(
        fen="startpos", score_cp=None, mate_in=1, best_move_san="Re8#", pv_san=["Re8#"]
    )
    with patch("src.agent.chess_agent.evaluate_position", return_value=position_eval):
        result = tools["evaluate_chess_position"]("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")

    assert "Mate in 1" in result
    assert "Re8#" in result


def test_evaluate_chess_position_rejects_invalid_fen():
    tools, _, _, _, _, _ = _build_tools()
    result = tools["evaluate_chess_position"]("not a fen")

    assert "Invalid FEN" in result


def test_evaluate_chess_position_reports_busy_when_pool_exhausted():
    """When every pooled engine is checked out, the tool should return a
    clear message the model can relay -- not raise and not hang.
    """
    db_pool = MagicMock()
    engine_pool = MagicMock()
    busy_message = (
        "All 2 engine(s) are busy handling other requests right now. "
        "Please try again in a moment."
    )
    engine_pool.checkout.side_effect = EngineBusyError(busy_message)
    voyage_client = MagicMock()
    tools = {t.name: t for t in build_tools(db_pool, engine_pool, voyage_client)}

    result = tools["evaluate_chess_position"](
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )

    assert result == busy_message


def test_find_similar_corpus_games_formats_matches():
    tools, _, conn, _, _, _ = _build_tools()
    matches = [
        SimilarGame(
            game_id="g1",
            white="Adams, Michael",
            black="Kramnik, Vladimir",
            year=2014,
            eco_code="C67",
            result="1/2-1/2",
            matching_plies=10,
        )
    ]
    with patch("src.agent.chess_agent._find_similar_games", return_value=matches) as mock_fn:
        result = tools["find_similar_corpus_games"](["e4", "e5", "Nf3"], max_ply=20, limit=5)

    mock_fn.assert_called_once_with(conn, ["e4", "e5", "Nf3"], max_ply=20, limit=5)
    assert "10 matching plies" in result
    assert "Adams, Michael vs Kramnik, Vladimir" in result


def test_find_similar_corpus_games_surfaces_invalid_input():
    tools, _, _, _, _, _ = _build_tools()
    with patch("src.agent.chess_agent._find_similar_games", side_effect=ValueError("empty")):
        result = tools["find_similar_corpus_games"]([])

    assert "Invalid input" in result


def test_ask_returns_text_from_final_message_only():
    early_message = MagicMock()
    early_message.content = [MagicMock(type="text", text="Let me check...")]
    final_message = MagicMock()
    final_message.content = [MagicMock(type="text", text="The answer is 42.")]

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [early_message, final_message]

    result = ask("some question", MagicMock(), MagicMock(), MagicMock(), client=client)

    assert result == "The answer is 42."
    call_kwargs = client.beta.messages.tool_runner.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert len(call_kwargs["tools"]) == 6
    assert call_kwargs["messages"] == [{"role": "user", "content": "some question"}]


def test_ask_returns_empty_string_when_no_messages_yielded():
    client = MagicMock()
    client.beta.messages.tool_runner.return_value = []

    result = ask("some question", MagicMock(), MagicMock(), MagicMock(), client=client)

    assert result == ""
