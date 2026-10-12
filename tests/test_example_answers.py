from streamlit.testing.v1 import AppTest

_CACHED_APP = """
from concurrent.futures import Future
from unittest.mock import MagicMock, patch
import streamlit as st
from src.ui import chat
from src.ui.example_answers import CachedAnswer
from src.recommendation.pipeline import LichessStudyRecommendation

st.session_state.chat_history = []
rec = LichessStudyRecommendation("s1", "Sicilian", "c1", "Open", "https://lichess.org/x", "Why.")
cached = CachedAnswer(
    answer="Cached answer.\\n[[diagram: Main line]]\\nMore.",
    touched_fens=[("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Main line")],
    recommendations=[rec],
    timeline={{"status_lines": [], "answer_start_seconds": 0.1, "total_seconds": 0.2}},
)
shared = MagicMock()
with (
    patch("src.ui.chat.cached_example_answer", return_value={cached_value}),
    patch("src.ui.chat.ask_agent", return_value="Live answer.") as ask,
    patch("src.ui.chat.get_shared_question_limit", return_value=shared),
    patch("src.ui.chat._start_resource_lookup") as start_lookup,
    patch("src.ui.chat.log_conversation_best_effort") as log,
    patch("src.ui.chat.get_db_pool"),
):
    chat._submit_question({question!r})
st.session_state["_asked"] = ask.called
st.session_state["_lookup_started"] = start_lookup.called
st.session_state["_shared_counted"] = shared.try_acquire.called
st.session_state["_logged"] = log.called
st.session_state["_request_times"] = len(st.session_state.get("request_times", []))
lookup = st.session_state.resource_lookup
st.session_state["_recs"] = lookup.future.result() if lookup is not None else None
"""


def _node_types(at: AppTest) -> list[str]:
    types = []

    def walk(node) -> None:
        types.append(node.type)
        for child in (getattr(node, "children", None) or {}).values():
            walk(child)

    for node in at.main.children.values():
        walk(node)
    return types


def _run(question: str, cached_value: str) -> AppTest:
    at = AppTest.from_string(_CACHED_APP.format(question=question, cached_value=cached_value))
    at.run()
    assert not at.exception
    return at


def test_a_cached_example_is_answered_without_the_agent_or_the_limits():
    at = _run("Evaluate 1. e4 e5 2. Qh5 for White", "cached")
    assert at.session_state["_asked"] is False
    assert at.session_state.chat_history[-1][1].startswith("Cached answer.")
    # Its diagram renders at the marker, and its recommendations are ready.
    assert "iframe" in _node_types(at)
    assert [r.study_id for r in at.session_state["_recs"]] == ["s1"]
    assert at.session_state["_lookup_started"] is False
    # Free questions don't use up either limit, but are still logged.
    assert at.session_state["_shared_counted"] is False
    assert at.session_state["_request_times"] == 0
    assert at.session_state["_logged"] is True


def test_a_question_without_a_cached_answer_goes_to_the_agent():
    at = _run("Is the Najdorf sound?", "None")
    assert at.session_state["_asked"] is True
    assert at.session_state.chat_history[-1][1] == "Live answer."
    assert at.session_state["_lookup_started"] is True
    assert at.session_state["_shared_counted"] is True


def test_only_example_questions_asked_without_a_position_are_looked_up():
    from unittest.mock import patch

    from src.ui.chat import EXAMPLE_QUESTIONS
    from src.ui.example_answers import cached_example_answer

    with patch("src.ui.example_answers._load_all_cached") as load:
        assert cached_example_answer("Is the Najdorf sound?", fen_context=None) is None
        assert (
            cached_example_answer(EXAMPLE_QUESTIONS[0], fen_context="8/8/8/8/8/8/8/4K2k w - - 0 1")
            is None
        )
        load.assert_not_called()


def test_a_cache_that_cannot_be_read_falls_back_to_the_agent():
    from unittest.mock import patch

    from src.ui.chat import EXAMPLE_QUESTIONS
    from src.ui.example_answers import cached_example_answer

    with patch("src.ui.example_answers._load_all_cached", side_effect=RuntimeError("db down")):
        assert cached_example_answer(EXAMPLE_QUESTIONS[0], fen_context=None) is None


def test_replay_emits_status_lines_then_the_answer_at_a_scaled_pace():
    from src.ui.example_answers import CachedAnswer, replay_cached_answer

    cached = CachedAnswer(
        answer="One two three four.",
        touched_fens=[("8/8/8/8/8/8/8/4K2k w - - 0 1", "Main line")],
        recommendations=[],
        timeline={
            "status_lines": [{"seconds": 10.0, "text": "Checking Layer 1."}],
            "answer_start_seconds": 20.0,
            "total_seconds": 40.0,
        },
    )
    clock = [0.0]
    events = []

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    answer = replay_cached_answer(
        cached,
        on_step=lambda text: events.append(("step", round(clock[0], 2), text)),
        on_chunk=lambda text: events.append(("chunk", round(clock[0], 2), text)),
        on_position=lambda fen, label=None: events.append(("position", fen, label)),
        speed=0.5,
        sleep=sleep,
        clock=lambda: clock[0],
    )

    assert answer == "One two three four."
    # The status line arrives at 10s x 0.5 = 5s, then becomes a status step.
    assert events[0] == ("chunk", 5.0, "Checking Layer 1.")
    assert events[1] == ("step", 5.0, "Checking Layer 1.")
    answer_chunks = [e for e in events if e[0] == "chunk"][1:]
    assert "".join(c[2] for c in answer_chunks) == "One two three four."
    assert answer_chunks[0][1] == 10.0  # answer starts at 20s x 0.5
    assert clock[0] == 20.0  # and ends at 40s x 0.5
    assert events[-1] == ("position", "8/8/8/8/8/8/8/4K2k w - - 0 1", "Main line")


def test_a_saved_answer_without_a_timeline_falls_back_to_the_agent():
    from unittest.mock import patch

    from src.ui.chat import EXAMPLE_QUESTIONS
    from src.ui.example_answers import CachedAnswer, cached_example_answer

    untimed = CachedAnswer(answer="Old.", touched_fens=[], recommendations=[], timeline=None)
    with patch(
        "src.ui.example_answers._load_all_cached", return_value={EXAMPLE_QUESTIONS[0]: untimed}
    ):
        assert cached_example_answer(EXAMPLE_QUESTIONS[0], fen_context=None) is None
