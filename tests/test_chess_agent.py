import threading
import time
from unittest.mock import MagicMock, patch

import chess
import psycopg2
import pytest

from src.agent.chess_agent import TOOL_LABELS, ask, build_tools
from src.engine.engine_pool import EngineBusyError
from src.engine.stockfish_eval import PositionEval
from src.personalization.similarity import SimilarGame
from src.rag.vector_search import ChunkResult
from src.search.structured_search import EcoSummary, MoveFrequency, SquareFrequency


def _build_tools(on_position=None):
    db_pool = MagicMock()
    conn = MagicMock()
    db_pool.getconn.return_value = conn
    engine_pool = MagicMock()
    engine_pool.size = 2  # compare_candidate_moves needs a real int for ThreadPoolExecutor
    engine = MagicMock()
    engine_pool.checkout.return_value.__enter__.return_value = engine
    voyage_client = MagicMock()
    tools = {
        t.name: t for t in build_tools(db_pool, engine_pool, voyage_client, on_position=on_position)
    }
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


def test_get_eco_summary_discards_both_connections_after_repeated_failures():
    """If the retry also hits a dropped connection, that second connection
    must be discarded too, not returned to the pool as healthy for the next
    caller to fail on. This is the bug the attempt-tracking loop in _query
    fixes: the old single try/except/finally shape had no way to tell a
    connection that failed twice from one that succeeded.
    """
    db_pool = MagicMock()
    dead_conn_1 = MagicMock()
    dead_conn_2 = MagicMock()
    db_pool.getconn.side_effect = [dead_conn_1, dead_conn_2]
    engine_pool = MagicMock()
    voyage_client = MagicMock()
    tools = {t.name: t for t in build_tools(db_pool, engine_pool, voyage_client)}

    with (
        patch(
            "src.agent.chess_agent.eco_summary",
            side_effect=[
                psycopg2.InterfaceError("connection already closed"),
                psycopg2.OperationalError("connection also closed"),
            ],
        ),
        pytest.raises(psycopg2.OperationalError),
    ):
        tools["get_eco_summary"]("C50")

    db_pool.putconn.assert_any_call(dead_conn_1, close=True)
    db_pool.putconn.assert_any_call(dead_conn_2, close=True)


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
        "All 2 engine(s) are busy handling other requests right now. Please try again in a moment."
    )
    engine_pool.checkout.side_effect = EngineBusyError(busy_message)
    voyage_client = MagicMock()
    tools = {t.name: t for t in build_tools(db_pool, engine_pool, voyage_client)}

    result = tools["evaluate_chess_position"](
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )

    assert result == busy_message


def test_compare_candidate_moves_evaluates_all_candidates_concurrently():
    """The whole reason this tool exists over calling evaluate_chess_position
    once per candidate is that the candidates run in parallel, not one
    after another (see chess_agent.py's own reasoning: a "compare several
    lines" question that used to mean N sequential engine searches plus N
    full model round trips was observed taking several real minutes).
    Verified directly here, not just trusted from reading the code, by
    tracking how many evaluate_position calls are genuinely in flight at
    the same time.
    """
    tools, _, _, _, _, _ = _build_tools()  # engine_pool.size == 2
    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    def slow_eval(engine, board, depth=18):
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return PositionEval(
            fen=board.fen(), score_cp=0, mate_in=None, best_move_san="e4", pv_san=["e4"]
        )

    with patch("src.agent.chess_agent.evaluate_position", side_effect=slow_eval):
        result = tools["compare_candidate_moves"](chess.Board().fen(), ["e4", "d4", "c4"])

    assert state["peak"] >= 2, "candidates ran one at a time, not concurrently"
    assert "e4:" in result
    assert "d4:" in result
    assert "c4:" in result


def test_compare_candidate_moves_reports_an_illegal_candidate_without_failing_the_rest():
    tools, _, _, _, _, _ = _build_tools()
    position_eval = PositionEval(
        fen="startpos", score_cp=12, mate_in=None, best_move_san="Nf6", pv_san=["Nf6"]
    )
    with patch("src.agent.chess_agent.evaluate_position", return_value=position_eval):
        result = tools["compare_candidate_moves"](chess.Board().fen(), ["e4", "e9"])

    assert "e4: Evaluation: 12 centipawns" in result
    assert "e9: not legal in that position." in result


def test_compare_candidate_moves_rejects_invalid_starting_fen():
    tools, _, _, _, _, _ = _build_tools()
    result = tools["compare_candidate_moves"]("not a fen", ["e4"])

    assert "Invalid FEN" in result


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
            fen_after="r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",
        )
    ]
    with patch("src.agent.chess_agent._find_similar_games", return_value=matches) as mock_fn:
        result = tools["find_similar_corpus_games"](["e4", "e5", "Nf3"], max_ply=20, limit=5)

    mock_fn.assert_called_once_with(conn, ["e4", "e5", "Nf3"], max_ply=20, limit=5)
    assert "10 matching plies" in result
    assert "Adams, Michael vs Kramnik, Vladimir" in result


def test_find_similar_corpus_games_reports_the_top_match_fen_via_on_position():
    positions = []
    tools, _, _, _, _, _ = _build_tools(on_position=positions.append)
    matches = [
        SimilarGame(
            game_id="g1",
            white="Adams, Michael",
            black="Kramnik, Vladimir",
            year=2014,
            eco_code="C67",
            result="1/2-1/2",
            matching_plies=10,
            fen_after="r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",
        ),
        SimilarGame(
            game_id="g2",
            white="Carlsen, Magnus",
            black="Caruana, Fabiano",
            year=2018,
            eco_code="C67",
            result="1-0",
            matching_plies=8,
            fen_after="some-other-fen",
        ),
    ]
    with patch("src.agent.chess_agent._find_similar_games", return_value=matches):
        tools["find_similar_corpus_games"](["e4", "e5", "Nf3"])

    # Only the top-ranked match's fen is reported, not every result's.
    assert positions == ["r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"]


def test_find_similar_corpus_games_does_not_call_on_position_with_no_results():
    positions = []
    tools, _, _, _, _, _ = _build_tools(on_position=positions.append)
    with patch("src.agent.chess_agent._find_similar_games", return_value=[]):
        tools["find_similar_corpus_games"](["e4", "e5", "Nf3"])

    assert positions == []


def test_find_similar_corpus_games_does_not_call_on_position_when_top_match_has_no_fen():
    positions = []
    tools, _, _, _, _, _ = _build_tools(on_position=positions.append)
    matches = [
        SimilarGame(
            game_id="g1",
            white="Adams, Michael",
            black="Kramnik, Vladimir",
            year=2014,
            eco_code="C67",
            result="1/2-1/2",
            matching_plies=0,
            fen_after=None,
        )
    ]
    with patch("src.agent.chess_agent._find_similar_games", return_value=matches):
        tools["find_similar_corpus_games"](["e4", "e5", "Nf3"])

    assert positions == []


def test_find_similar_corpus_games_surfaces_invalid_input():
    tools, _, _, _, _, _ = _build_tools()
    with patch("src.agent.chess_agent._find_similar_games", side_effect=ValueError("empty")):
        result = tools["find_similar_corpus_games"]([])

    assert "Invalid input" in result


def test_show_opening_line_reports_the_fen_and_label_via_on_position():
    on_position = MagicMock()
    tools, _, _, _, _, _ = _build_tools(on_position=on_position)

    result = tools["show_opening_line"](["e4", "e5", "Nf3"], "Main line")

    board = chess.Board()
    for san in ["e4", "e5", "Nf3"]:
        board.push_san(san)
    on_position.assert_called_once_with(board.fen(), label="Main line", update_board=False)
    assert "Main line" in result
    assert "e4 e5 Nf3" in result


def test_show_opening_line_rejects_an_illegal_move_without_calling_on_position():
    on_position = MagicMock()
    tools, _, _, _, _, _ = _build_tools(on_position=on_position)

    result = tools["show_opening_line"](["e4", "e5", "Qh5", "Ke7", "Qxe7"], "bogus")

    # e5's king can't legally reach e7 in one move -- the sequence should
    # fail at that step, not silently continue or show a wrong position.
    on_position.assert_not_called()
    assert "not legal" in result or "isn't legal" in result


def test_show_opening_line_called_twice_reports_two_distinct_positions():
    on_position = MagicMock()
    tools, _, _, _, _, _ = _build_tools(on_position=on_position)

    tools["show_opening_line"](["e4", "e5", "Nf3"], "Main line")
    tools["show_opening_line"](["e4", "e5", "Nf3", "Nc6", "Bb5"], "Ruy Lopez")

    assert on_position.call_count == 2
    first_call, second_call = on_position.call_args_list
    assert first_call.kwargs["label"] == "Main line"
    assert second_call.kwargs["label"] == "Ruy Lopez"
    assert first_call.args[0] != second_call.args[0]


def _tool_use_block(name: str, input: dict | None = None) -> MagicMock:
    # MagicMock(name=...) is reserved by unittest.mock for the mock's own
    # repr, not for setting a real .name attribute -- has to be set after
    # construction, or block.name silently returns the internal mock name.
    block = MagicMock(type="tool_use")
    block.name = name
    block.input = input or {}
    return block


def _stream_turn(content_blocks: list, chunks: list[str] | None = None) -> MagicMock:
    """Build a mock stream-shaped turn matching what ask() consumes with
    stream=True: text_stream yields chunks as they arrive, get_final_message
    returns the complete message once the stream is drained.

    Defaults chunks to the joined text of content_blocks as one chunk, for
    tests that only care about the final assembled text, not incremental
    delivery.
    """
    if chunks is None:
        chunks = ["".join(block.text for block in content_blocks if block.type == "text")]
    turn = MagicMock()
    turn.text_stream = iter(chunks)
    message = MagicMock()
    message.content = content_blocks
    turn.get_final_message.return_value = message
    return turn


def test_ask_returns_text_from_final_message_only():
    early_turn = _stream_turn([MagicMock(type="text", text="Let me check...")])
    final_turn = _stream_turn([MagicMock(type="text", text="The answer is 42.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [early_turn, final_turn]

    result = ask("some question", MagicMock(), MagicMock(), MagicMock(), client=client)

    assert result == "The answer is 42."
    call_kwargs = client.beta.messages.tool_runner.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert call_kwargs["stream"] is True
    assert len(call_kwargs["tools"]) == 8
    assert call_kwargs["messages"] == [{"role": "user", "content": "some question"}]


def test_ask_prepends_history_before_the_new_question():
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])
    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [final_turn]
    prior_turns = [
        {"role": "user", "content": "What's the Ruy Lopez?"},
        {"role": "assistant", "content": "1.e4 e5 2.Nf3 Nc6 3.Bb5, a classical opening."},
    ]

    ask(
        "What about the Berlin Defense?",
        MagicMock(),
        MagicMock(),
        MagicMock(),
        client=client,
        history=prior_turns,
    )

    call_kwargs = client.beta.messages.tool_runner.call_args.kwargs
    assert call_kwargs["messages"] == [
        *prior_turns,
        {"role": "user", "content": "What about the Berlin Defense?"},
    ]


def test_ask_without_history_sends_only_the_new_question():
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])
    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [final_turn]

    ask("a fresh question", MagicMock(), MagicMock(), MagicMock(), client=client)

    call_kwargs = client.beta.messages.tool_runner.call_args.kwargs
    assert call_kwargs["messages"] == [{"role": "user", "content": "a fresh question"}]


def test_ask_returns_empty_string_when_no_messages_yielded():
    client = MagicMock()
    client.beta.messages.tool_runner.return_value = []

    result = ask("some question", MagicMock(), MagicMock(), MagicMock(), client=client)

    assert result == ""


def test_ask_recovers_when_the_loop_ends_on_a_tool_call_with_no_synthesis():
    """Reproduces the real, observed failure: the tool-calling loop ends on
    a turn whose only content is a tool_use block (no text at all), instead
    of a clean text-only synthesis turn. ask() should force a real answer
    rather than silently returning empty.
    """
    dangling_turn = _stream_turn([_tool_use_block("search_annotations", {"query": "x"})])

    client = MagicMock()
    # A real BaseSyncToolRunner exposes _params["messages"] (see
    # _recover_synthesis's docstring for why this is read at all) -- a bare
    # MagicMock() return_value already has an auto-speccing ._params
    # attribute, so set it explicitly to a realistic accumulated
    # conversation ending on the same dangling tool_use block.
    runner = MagicMock()
    runner.__iter__.return_value = iter([dangling_turn])
    runner._params = {
        "messages": [
            {"role": "user", "content": "some question"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "search_annotations", "input": {}}
                ],
            },
        ]
    }
    client.beta.messages.tool_runner.return_value = runner
    recovery_response = MagicMock()
    recovery_response.content = [MagicMock(type="text", text="Here is the real answer.")]
    client.beta.messages.create.return_value = recovery_response

    steps = []
    result = ask(
        "some question", MagicMock(), MagicMock(), MagicMock(), client=client, on_step=steps.append
    )

    assert result == "Here is the real answer."
    # The dangling tool_use-only assistant message must not be sent as-is
    # (the API requires a tool_result immediately after any tool_use) --
    # confirm it was dropped rather than forwarded verbatim.
    recovery_messages = client.beta.messages.create.call_args.kwargs["messages"]
    assert all(
        not (
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
        )
        for m in recovery_messages
        if isinstance(m.get("content"), list)
    )
    assert recovery_messages[-1]["role"] == "user"
    assert "complete answer now" in recovery_messages[-1]["content"]
    assert any("Recovering" in s for s in steps)


def test_ask_reports_model_rationale_via_on_step():
    tool_turn = _stream_turn(
        [
            MagicMock(type="text", text="Checking Layer 2 for strategic ideas."),
            _tool_use_block("search_annotations"),
        ]
    )
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [tool_turn, final_turn]

    steps = []
    result = ask("q", MagicMock(), MagicMock(), MagicMock(), client=client, on_step=steps.append)

    assert result == "Final answer."
    assert steps == ["Checking Layer 2 for strategic ideas."]


def test_ask_falls_back_to_tool_label_when_model_gives_no_rationale():
    tool_turn = _stream_turn([_tool_use_block("evaluate_chess_position")])
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [tool_turn, final_turn]

    steps = []
    ask("q", MagicMock(), MagicMock(), MagicMock(), client=client, on_step=steps.append)

    assert steps == [TOOL_LABELS["evaluate_chess_position"]]


def test_ask_joins_fallback_labels_for_multiple_tools_with_no_rationale():
    tool_turn = _stream_turn(
        [
            _tool_use_block("get_eco_summary"),
            _tool_use_block("search_annotations"),
        ]
    )
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [tool_turn, final_turn]

    steps = []
    ask("q", MagicMock(), MagicMock(), MagicMock(), client=client, on_step=steps.append)

    assert steps == [f"{TOOL_LABELS['get_eco_summary']} / {TOOL_LABELS['search_annotations']}"]


def test_ask_does_not_call_on_step_for_a_text_only_turn():
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [final_turn]

    steps = []
    ask("q", MagicMock(), MagicMock(), MagicMock(), client=client, on_step=steps.append)

    assert steps == []


def test_ask_without_on_step_does_not_error_on_a_tool_call_turn():
    tool_turn = _stream_turn([_tool_use_block("search_annotations")])
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [tool_turn, final_turn]

    result = ask("q", MagicMock(), MagicMock(), MagicMock(), client=client)

    assert result == "Final answer."


def test_ask_reports_position_via_on_position():
    tool_turn = _stream_turn(
        [_tool_use_block("evaluate_chess_position", {"fen": "8/8/8/8/8/8/8/K6k w - - 0 1"})]
    )
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [tool_turn, final_turn]

    positions = []
    ask("q", MagicMock(), MagicMock(), MagicMock(), client=client, on_position=positions.append)

    assert positions == ["8/8/8/8/8/8/8/K6k w - - 0 1"]


def test_ask_does_not_crash_on_an_evaluate_chess_position_call_missing_fen():
    """Regression test for a real, reproduced KeyError crash: a turn cut
    off mid-generation (see MAX_TOKENS's comment) can produce an
    evaluate_chess_position tool_use block whose input dict never got as
    far as including "fen" at all. This must not crash the app -- just
    skip reporting a position update for that turn.
    """
    tool_turn = _stream_turn([_tool_use_block("evaluate_chess_position", {})])
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [tool_turn, final_turn]

    positions = []
    result = ask(
        "q", MagicMock(), MagicMock(), MagicMock(), client=client, on_position=positions.append
    )

    assert result == "Final answer."
    assert positions == []


def test_ask_does_not_call_on_position_for_other_tools():
    tool_turn = _stream_turn([_tool_use_block("search_annotations", {"query": "isolated pawns"})])
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [tool_turn, final_turn]

    positions = []
    ask("q", MagicMock(), MagicMock(), MagicMock(), client=client, on_position=positions.append)

    assert positions == []


def test_ask_calls_on_position_for_every_evaluate_call_across_turns():
    first_turn = _stream_turn([_tool_use_block("evaluate_chess_position", {"fen": "fen-one"})])
    second_turn = _stream_turn([_tool_use_block("evaluate_chess_position", {"fen": "fen-two"})])
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [first_turn, second_turn, final_turn]

    positions = []
    ask("q", MagicMock(), MagicMock(), MagicMock(), client=client, on_position=positions.append)

    assert positions == ["fen-one", "fen-two"]


def test_ask_without_on_position_does_not_error_on_an_evaluate_call_turn():
    tool_turn = _stream_turn([_tool_use_block("evaluate_chess_position", {"fen": "some-fen"})])
    final_turn = _stream_turn([MagicMock(type="text", text="Final answer.")])

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [tool_turn, final_turn]

    result = ask("q", MagicMock(), MagicMock(), MagicMock(), client=client)

    assert result == "Final answer."


def test_ask_calls_on_chunk_for_every_turn_in_order():
    tool_turn = _stream_turn(
        [MagicMock(type="text", text="Checking Layer 2."), _tool_use_block("search_annotations")],
        chunks=["Checking ", "Layer 2."],
    )
    final_turn = _stream_turn(
        [MagicMock(type="text", text="The answer.")],
        chunks=["The ", "answer."],
    )

    client = MagicMock()
    client.beta.messages.tool_runner.return_value = [tool_turn, final_turn]

    chunks = []
    result = ask("q", MagicMock(), MagicMock(), MagicMock(), client=client, on_chunk=chunks.append)

    assert result == "The answer."
    assert chunks == ["Checking ", "Layer 2.", "The ", "answer."]


def test_tool_labels_cover_every_tool():
    tools = build_tools(MagicMock(), MagicMock(), MagicMock())
    tool_names = {t.name for t in tools}

    assert tool_names == set(TOOL_LABELS)
