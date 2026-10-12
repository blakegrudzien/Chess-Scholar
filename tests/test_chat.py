"""Tests for src/ui/chat.py. AppTest (streamlit.testing.v1) runs the real
Streamlit element tree without a browser, so tests can check what renders
and in what order, which matters most for diagram placement.
"""

from streamlit.testing.v1 import AppTest

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OTHER_FEN = "8/8/8/8/8/8/8/4K2k w - - 0 1"


def _leaf_nodes(at: AppTest) -> list:
    """Every markdown and iframe element in document order, however deeply
    nested, so tests can check that diagrams land between the right pieces
    of text."""
    leaves = []

    def walk(node) -> None:
        if node.type in ("markdown", "iframe"):
            leaves.append(node)
        children = getattr(node, "children", None)
        if children:
            for child in children.values():
                walk(child)

    for node in at.main.children.values():
        walk(node)
    return leaves


def _node_types(at: AppTest) -> list[str]:
    return [node.type for node in _leaf_nodes(at)]


def _markdown_values(at: AppTest) -> list[str]:
    return [node.value for node in _leaf_nodes(at) if node.type == "markdown"]


def test_render_answer_content_places_diagram_at_its_marker():
    at = AppTest.from_string(f"""
from src.ui import chat
chat._render_answer_content(
    "Before the line.\\n[[diagram: Main line]]\\nAfter the line.",
    [("{START_FEN}", "Main line")],
)
""")
    at.run()
    assert not at.exception
    assert _node_types(at) == ["markdown", "iframe", "markdown"]
    values = _markdown_values(at)
    assert "Before the line." in values[0]
    assert "After the line." in values[1]


def test_render_answer_content_strips_unmatched_marker_and_falls_back_to_the_end():
    """A marker that matches no diagram is removed from the text, and a
    diagram with no marker still appears, after the text."""
    at = AppTest.from_string(f"""
from src.ui import chat
chat._render_answer_content(
    "Some prose. [[diagram: Nonexistent]] more prose.",
    [("{OTHER_FEN}", "Main line")],
)
""")
    at.run()
    assert not at.exception
    assert _node_types(at) == ["markdown", "markdown", "iframe"]
    combined = " ".join(_markdown_values(at))
    assert "[[diagram:" not in combined
    assert "Nonexistent" not in combined


def test_render_answer_content_puts_unlabeled_diagrams_at_the_end():
    """Unlabeled positions (engine evaluations, similar games) have no
    marker to match, so they always follow the answer text."""
    at = AppTest.from_string(f"""
from src.ui import chat
chat._render_answer_content(
    "The position after this line is roughly equal.",
    [("{OTHER_FEN}", None)],
)
""")
    at.run()
    assert not at.exception
    assert _node_types(at) == ["markdown", "iframe"]


def test_render_answer_content_caps_total_diagrams_shown():
    labels = [f"Line {n}" for n in range(1, 6)]  # one more than MAX_INLINE_DIAGRAMS
    markers = "\\n".join(f"[[diagram: {label}]]" for label in labels)
    fens_literal = ", ".join(f'("{START_FEN}", "{label}")' for label in labels)
    at = AppTest.from_string(f"""
from src.ui import chat
chat._render_answer_content("{markers}", [{fens_literal}])
""")
    at.run()
    assert not at.exception
    assert _node_types(at).count("iframe") == 4  # MAX_INLINE_DIAGRAMS


def test_to_model_text_prepends_fen_context_only_when_given():
    from src.ui.chat import _to_model_text

    assert _to_model_text("What's the plan here?", "8/8/8/8/8/8/8/4K2k w - - 0 1") == (
        "Current board position: 8/8/8/8/8/8/8/4K2k w - - 0 1\n\nWhat's the plan here?"
    )
    assert _to_model_text("What's the plan here?", None) == "What's the plan here?"


def test_build_message_history_replays_prior_turns_with_fen_context():
    at = AppTest.from_string("""
import streamlit as st
from src.ui.chat import _build_message_history

st.session_state.chat_history = [
    ("user", "What's the plan in the KID?", None, []),
    ("assistant", "Black plays for ...e5 or ...c5.", None, []),
    ("user", "What about this position?", "8/8/8/8/8/8/8/4K2k w - - 0 1", []),
]
history = _build_message_history()
st.session_state["_history_out"] = history
""")
    at.run()
    assert not at.exception
    history = at.session_state["_history_out"]
    assert history[0] == {"role": "user", "content": "What's the plan in the KID?"}
    assert history[1] == {"role": "assistant", "content": "Black plays for ...e5 or ...c5."}
    assert history[2] == {
        "role": "user",
        "content": (
            "Current board position: 8/8/8/8/8/8/8/4K2k w - - 0 1\n\nWhat about this position?"
        ),
    }


def test_build_message_history_caps_at_max_history_messages():
    at = AppTest.from_string("""
import streamlit as st
from src.ui.chat import MAX_HISTORY_MESSAGES, _build_message_history

st.session_state.chat_history = [
    ("user", f"question {i}", None, []) for i in range(MAX_HISTORY_MESSAGES + 4)
]
history = _build_message_history()
st.session_state["_history_len"] = len(history)
st.session_state["_history_first"] = history[0]["content"]
""")
    at.run()
    assert not at.exception
    from src.ui.chat import MAX_HISTORY_MESSAGES

    assert at.session_state["_history_len"] == MAX_HISTORY_MESSAGES
    assert at.session_state["_history_first"] == "question 4"


def test_ask_with_status_shows_a_fallback_message_when_the_answer_is_empty():
    """An empty answer from ask() shows an explanation, not an empty
    bubble."""
    at = AppTest.from_string("""
from unittest.mock import patch
from src.ui import chat

with patch("src.ui.chat.ask_agent", return_value=""):
    answer, touched = chat.ask_with_status("What's the best response to 1. e4?")

import streamlit as st
st.session_state["_answer"] = answer
st.session_state["_touched"] = touched
""")
    at.run()
    assert not at.exception
    assert "interrupted" in at.session_state["_answer"].lower()
    assert at.session_state["_touched"] == []
    markdown_values = [n.value for n in at.main.children.values() if n.type == "markdown"]
    assert any("interrupted" in value.lower() for value in markdown_values)


def test_ask_with_status_shows_a_fallback_message_when_the_api_call_fails():
    """An API error from the SDK itself (rate limit, dropped connection,
    5xx) isn't a tool exception, so tool_runner doesn't catch it. It must
    produce the fallback message, not crash the page."""
    at = AppTest.from_string("""
from unittest.mock import patch
import httpx
import anthropic
from src.ui import chat

error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
with patch("src.ui.chat.ask_agent", side_effect=error):
    answer, touched = chat.ask_with_status("What's the best response to 1. e4?")

import streamlit as st
st.session_state["_answer"] = answer
st.session_state["_touched"] = touched
""")
    at.run()
    assert not at.exception
    assert "interrupted" in at.session_state["_answer"].lower()
    assert at.session_state["_touched"] == []


def test_ask_with_status_places_a_diagram_at_its_marker_end_to_end():
    """A labeled diagram lands at its [[diagram: ...]] marker when going
    through the whole ask_with_status path, streaming included."""
    at = AppTest.from_string(f"""
from unittest.mock import patch
from src.ui import chat

def fake_ask_agent(question, on_step=None, on_chunk=None, on_position=None, history=None):
    on_position("{START_FEN}", label="Main line")
    return "Here is the idea.\\n[[diagram: Main line]]\\nThat's the point."

with patch("src.ui.chat.ask_agent", fake_ask_agent):
    answer, touched = chat.ask_with_status("What's the main line here?")

import streamlit as st
st.session_state["_touched"] = touched
""")
    at.run()
    assert not at.exception
    assert at.session_state["_touched"] == [(START_FEN, "Main line")]
    assert _node_types(at) == ["markdown", "iframe", "markdown"]
    combined = " ".join(_markdown_values(at))
    assert "[[diagram:" not in combined


def test_ask_with_status_collapses_tool_lines_into_done_and_removes_the_bar():
    at = AppTest.from_string("""
from unittest.mock import patch
from src.ui import chat

def fake_ask_agent(question, on_step=None, on_chunk=None, on_position=None, history=None):
    on_step("Layer 1 (structured search): pulling opening statistics.")
    on_step("Layer 3 (Stockfish): evaluating the position.")
    return "The answer."

with patch("src.ui.chat.ask_agent", fake_ask_agent):
    chat.ask_with_status("What's the best response to 1. e4?")
""")
    at.run()
    assert not at.exception
    nodes = []

    def walk(node) -> None:
        nodes.append(node)
        for child in (getattr(node, "children", None) or {}).values():
            walk(child)

    for node in at.main.children.values():
        walk(node)
    done = [n for n in nodes if n.type == "status" and n.label == "Done"]
    assert len(done) == 1
    step_text = " ".join(c.value for c in done[0].children.values() if c.type == "markdown")
    assert "Layer 1" in step_text and "Layer 3" in step_text
    # The bar is gone once the answer is in.
    assert not [n for n in nodes if n.type == "html"]


def test_game_path_from_pgn_returns_every_ply_fen():
    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    at = AppTest.from_string(f"""
from src.ui.chat import _game_path_from_pgn
import streamlit as st
st.session_state["_path"] = _game_path_from_pgn({pgn!r})
""")
    at.run()
    assert not at.exception
    path = at.session_state["_path"]
    assert len(path) == 5  # starting position + 4 plies
    assert path[0] == START_FEN

    import chess

    expected_final = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6"]:
        expected_final.push_san(san)
    assert path[-1] == expected_final.fen()


def test_game_path_from_pgn_returns_none_for_unparseable_pgn():
    at = AppTest.from_string("""
from src.ui.chat import _game_path_from_pgn
import streamlit as st
st.session_state["_path"] = _game_path_from_pgn("")
""")
    at.run()
    assert not at.exception
    assert at.session_state["_path"] is None


def test_a_tool_reporting_a_position_never_touches_the_board():
    """Tools report the positions they touch so the answer can show them as
    diagrams; they must never replace the board. Replacing it with a fresh
    chess.Board(fen) erased the move history even when the position was the
    same one (Evaluate disabled Undo), and a typed question about another
    line wiped the game in progress. During replay the board is the
    free-play game sitting behind the visible replay, so it must survive
    there too."""
    for replaying in (False, True):
        at = AppTest.from_string(f"""
from unittest.mock import patch
import chess
import streamlit as st
from src.ui import chat

if {replaying}:
    st.session_state.game_path = ["{START_FEN}", "{OTHER_FEN}"]
    st.session_state.game_path_index = 0
else:
    st.session_state.game_path = None
original_board = chess.Board()
for san in ("e4", "e5"):
    original_board.push_san(san)
st.session_state.board = original_board

def fake_ask_agent(question, on_step=None, on_chunk=None, on_position=None, history=None):
    on_position(original_board.fen())
    on_position("{OTHER_FEN}")
    return "Evaluation done."

with patch("src.ui.chat.ask_agent", fake_ask_agent):
    _, touched = chat.ask_with_status("Evaluate this position")

st.session_state["_same_object"] = st.session_state.board is original_board
st.session_state["_history"] = len(st.session_state.board.move_stack)
st.session_state["_touched"] = touched
""")
        at.run()
        assert not at.exception
        assert at.session_state["_same_object"] is True
        assert at.session_state["_history"] == 2  # Undo still has moves to undo
        # The positions still reach the answer as diagrams.
        assert [fen for fen, _ in at.session_state["_touched"]][-1] == OTHER_FEN


def test_example_prompts_render_as_buttons():
    at = AppTest.from_string("""
from src.ui.chat import _render_example_prompts
_render_example_prompts()
""")
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "How should White meet the Sicilian Defense?" in labels
    assert len(labels) == 4  # one example per layer/feature


def test_example_prompts_are_disabled_while_an_answer_generates():
    """Clicking an example mid-answer would cancel the answer being
    generated, since any widget interaction cancels the running script."""
    at = AppTest.from_string("""
from src.ui.chat import _render_example_prompts
_render_example_prompts(disabled=True)
""")
    at.run()
    assert not at.exception
    assert len(at.button) == 4
    assert all(b.disabled for b in at.button)


def test_clicking_an_example_prompt_stashes_it_as_a_pending_question():
    """Clicking an example only stashes it as the pending question. The
    examples render before the transcript, so submitting from the click
    would draw the new exchange twice in the same run."""
    at = AppTest.from_string("""
import streamlit as st
from src.ui.chat import _render_example_prompts

st.session_state.chat_history = []
_render_example_prompts()
""")
    at.run()
    assert not at.exception
    example_button = next(b for b in at.button if "Sicilian" in b.label)

    example_button.click().run()

    assert not at.exception
    # Stashed, not submitted: render_main_screen submits it on the next run.
    assert at.session_state.chat_history == []
    assert at.session_state.pending_question == (
        "How should White meet the Sicilian Defense?",
        None,
    )


_LOOKUP_APP = """
from concurrent.futures import Future
import streamlit as st
from src.ui import chat
from src.recommendation.pipeline import LichessStudyRecommendation

future = Future()
{setup}
if "resource_lookup" not in st.session_state:
    st.session_state.chat_history = [
        ("user", "How should White meet the Sicilian?", None, []),
        ("assistant", "The Open Sicilian is the main try.", None, []),
    ]
    st.session_state.resource_lookup = chat._ResourceLookup(future)
chat._render_resource_recommendations()
"""

_STUDY = (
    "LichessStudyRecommendation(study_id='s1', study_title='All about the Sicilian', "
    "chapter_id='c1', chapter_name='Open Sicilian', "
    "embed_url='https://lichess.org/study/embed/s1/c1', blurb='Covers the main line.')"
)


def _find_button(at: AppTest):
    return next(b for b in at.button if b.key == "find_resources")


def test_resources_button_is_disabled_while_the_lookup_runs():
    at = AppTest.from_string(_LOOKUP_APP.format(setup=""))
    at.run()
    assert not at.exception
    button = _find_button(at)
    assert button.disabled
    assert button.label == "Finding related resources..."


def test_resources_button_reveals_results_once_the_lookup_finishes():
    at = AppTest.from_string(_LOOKUP_APP.format(setup=f"future.set_result([{_STUDY}])"))
    at.run()
    button = _find_button(at)
    assert not button.disabled
    assert button.label == "Show related resources"
    assert not [n for n in at.main.children.values() if n.type == "html"]

    button.click().run()

    assert not at.exception
    assert any("All about the Sicilian" in str(n.proto) for n in at.get("html"))


def test_resources_button_says_so_when_nothing_was_found():
    at = AppTest.from_string(_LOOKUP_APP.format(setup="future.set_result([])"))
    at.run()
    button = _find_button(at)
    assert button.disabled
    assert button.label == "No related resources found"


def test_resources_button_shows_a_failed_lookup_instead_of_crashing():
    """The lookup chains Anthropic, Voyage and live Lichess calls; any of
    them can fail, and that must show on the button, not crash the page."""
    at = AppTest.from_string(
        _LOOKUP_APP.format(setup="future.set_exception(RuntimeError('Lichess timed out'))")
    )
    at.run()
    assert not at.exception
    button = _find_button(at)
    assert button.disabled
    assert button.label == "Couldn't load related resources"


def test_a_new_answer_replaces_the_previous_lookup():
    at = AppTest.from_string("""
from unittest.mock import MagicMock, patch
import streamlit as st
from src.ui import chat

st.session_state.chat_history = [
    ("user", "How should White meet the Sicilian?", None, []),
    ("assistant", "The Open Sicilian is the main try.", None, []),
]
stale = object()
st.session_state.resource_lookup = stale
with (
    patch("src.ui.chat.ask_with_status", return_value=("Play the Ruy Lopez.", [])),
    patch("src.ui.chat.log_conversation_best_effort"),
    patch("src.ui.chat.get_db_pool", return_value=MagicMock()),
    patch("src.ui.chat.get_voyage", return_value=MagicMock()),
    patch("src.ui.chat.get_anthropic_client", return_value=MagicMock()),
    patch("src.ui.chat.get_lichess_http_client", return_value=MagicMock()),
    patch("src.ui.chat.get_lichess_pacer", return_value=MagicMock()),
    patch("src.ui.chat.recommend_resources", return_value=[]) as lookup,
):
    chat._submit_question("What about 1. e4 e5?")
    result = st.session_state.resource_lookup.future.result(timeout=5)
st.session_state["_replaced"] = st.session_state.resource_lookup is not stale
st.session_state["_result"] = result
st.session_state["_context"] = lookup.call_args.args[0]
""")
    at.run()
    assert not at.exception
    assert at.session_state["_replaced"]
    assert at.session_state["_result"] == []
    # Looked up for the new exchange, answer included (see _lookup_context).
    assert "What about 1. e4 e5?" in at.session_state["_context"]
    assert "Ruy Lopez" in at.session_state["_context"]


def test_describe_uploaded_game_shows_a_friendly_error_for_a_header_delimiter_collision():
    """A header containing the game-ID delimiter ("|") makes
    compute_game_id raise ValueError. An upload like that gets the same
    "couldn't find a game" message as an unreadable file, returned for the
    caller to show."""
    at = AppTest.from_string('''
class _FakeUpload:
    def __init__(self, text):
        self._text = text

    def getvalue(self):
        return self._text.encode("utf-8")

pgn = """[Event "Test"]
[White "Weird|Name"]
[Black "Bob"]
[Date "2021.05.01"]
[Result "1-0"]

1. e4 e5 1-0
"""

from src.ui.chat import _describe_uploaded_game
import streamlit as st
st.session_state["_result"] = _describe_uploaded_game(_FakeUpload(pgn), "")
''')
    at.run()
    assert not at.exception
    question, notices = at.session_state["_result"]
    assert question is None
    assert any(
        level == "error" and "couldn't find a game" in message.lower() for level, message in notices
    )


def test_describe_uploaded_game_rejects_a_file_with_no_moves():
    """python-chess parses prose into a game with zero moves; that must be
    rejected, not sent to the model as an empty move list."""
    at = AppTest.from_string("""
class _FakeUpload:
    def __init__(self, text):
        self._text = text

    def getvalue(self):
        return self._text.encode("utf-8")

from src.ui.chat import _describe_uploaded_game
import streamlit as st
st.session_state["_result"] = _describe_uploaded_game(
    _FakeUpload("this is not a chess game, just some prose"), ""
)
""")
    at.run()
    assert not at.exception
    question, notices = at.session_state["_result"]
    assert question is None
    assert any("couldn't find a game" in message.lower() for _, message in notices)


def test_describe_uploaded_game_warns_when_moves_were_truncated():
    """A movetext error stops the parse there, so the moves that survive are
    a prefix of the real game. Analyzing part of someone's game without
    saying so is the same silent data loss the multiple-games case already
    warns about.
    """
    at = AppTest.from_string('''
class _FakeUpload:
    def __init__(self, text):
        self._text = text

    def getvalue(self):
        return self._text.encode("utf-8")

pgn = """[Event "Test"]
[White "A"]
[Black "B"]

1. e4 e5 2. Ke2 Zz9 3. Qh5
"""

from src.ui.chat import _describe_uploaded_game
import streamlit as st
st.session_state["_result"] = _describe_uploaded_game(_FakeUpload(pgn), "")
''')
    at.run()
    assert not at.exception
    question, notices = at.session_state["_result"]
    assert question is not None  # the valid prefix is still worth analyzing
    assert any(
        level == "warning" and "couldn't be read" in message.lower() for level, message in notices
    )


def test_lookup_context_carries_the_board_position_and_the_answer():
    """A question asked from the board says little on its own; the lookup
    context includes its position and the start of the answer, where the
    opening gets named."""
    from src.ui.chat import _lookup_context

    fen = "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"
    history = [
        ("user", "How can white play in this position?", fen, []),
        ("assistant", "This position (1.d4 f5) is the Dutch Defense, ECO A80.", None, []),
    ]

    lookup = _lookup_context(history)

    assert "How can white play in this position?" in lookup
    assert fen in lookup  # the board position the question was about
    assert "Dutch Defense" in lookup  # the topic, which only the answer names


def test_lookup_context_handles_a_question_with_no_board_position():
    from src.ui.chat import _lookup_context

    history = [
        ("user", "What is the Sicilian Defense?", None, []),
        ("assistant", "A reply to 1.e4 with 1...c5.", None, []),
    ]

    lookup = _lookup_context(history)

    assert "What is the Sicilian Defense?" in lookup
    assert "Current board position" not in lookup
    assert "A reply to 1.e4 with 1...c5." in lookup


def test_submit_question_logs_the_real_conversation_for_later_eval_review():
    """Every question is logged with its question, position and answer.

    get_db_pool is mocked too: it's evaluated as an argument before the
    mocked logger is called, and would otherwise need a real database.
    """
    at = AppTest.from_string("""
from unittest.mock import MagicMock, patch
import streamlit as st
from src.ui import chat

st.session_state.chat_history = []
mock_log = MagicMock()
with (
    patch("src.ui.chat.ask_agent", return_value="Play 3.d4."),
    patch("src.ui.chat.get_db_pool", return_value=MagicMock()),
    patch("src.ui.chat.log_conversation_best_effort", mock_log),
    # Stubbed so this test doesn't start a real background resource lookup.
    patch("src.ui.chat._start_resource_lookup"),
):
    chat._submit_question(
        "How should White meet the Sicilian?", fen_context="8/8/8/8/8/8/8/4K2k w - - 0 1"
    )

st.session_state["_call_args"] = mock_log.call_args
""")
    at.run()
    assert not at.exception
    call = at.session_state["_call_args"]
    assert call.args[1] == "How should White meet the Sicilian?"
    assert call.args[2] == "8/8/8/8/8/8/8/4K2k w - - 0 1"
    assert call.args[3] == "Play 3.d4."


def test_real_status_lines_are_not_mistaken_for_an_answer():
    from src.ui.chat import _looks_like_answer

    # Real tool-call status lines from the app's logs, the longest ~215 chars.
    for status_line in (
        "Checking multiple layers at once: Layer 1 stats on the Sicilian family and "
        "common second moves, Layer 2 strategic commentary on anti-Sicilian systems, "
        "and Layer 3 engine comparison of White's main options after 1.e4 c5.",
        "Retrying the Layer 1 ECO summary call that failed due to a connection error, "
        "and adding stats for the two main Sicilian branches (Open Sicilian and Closed "
        "Sicilian) for comparison.",
        "I'll check both the engine evaluation of the resulting position and how the "
        "database classifies/handles this",
    ):
        assert not _looks_like_answer(status_line)


def test_long_text_or_a_line_break_counts_as_an_answer():
    from src.ui.chat import _looks_like_answer

    assert _looks_like_answer("**2. Qh5?! Evaluation**\n")
    assert _looks_like_answer("word " * 70)
