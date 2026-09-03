"""src/ui/chat.py has real branching logic (diagram placement, message
history construction, the empty-answer fallback) but, before this file,
zero automated coverage -- a change to any of it was only ever checked by
hand in a browser or a throwaway script. AppTest (streamlit.testing.v1)
drives the real Streamlit element tree without a browser, so it can assert
on render *order*, not just "did this raise" -- essential for
_render_answer_content, whose whole job is putting things in the right
order.
"""

from streamlit.testing.v1 import AppTest

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OTHER_FEN = "8/8/8/8/8/8/8/4K2k w - - 0 1"


def _leaf_nodes(at: AppTest) -> list:
    """Every markdown/iframe leaf under the app root, in document order,
    however deeply nested inside container blocks (st.status, st.empty,
    a placeholder's .container() -- ask_with_status uses all three). What
    matters for _render_answer_content is that diagrams land *between* the
    right text segments, not just that they're present somewhere, so this
    walks the real tree instead of only checking top-level children.
    """
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
    """A marker whose label doesn't match any diagram must not leak its raw
    "[[diagram: ...]]" syntax into the chat -- and the diagram it *should*
    have matched (a different label the model never wrote a marker for)
    still has to show up somewhere, so it falls back to the end.
    """
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
    """evaluate_chess_position and find_similar_corpus_games report FENs
    with label=None (see ask_with_status's docstring) -- there's no label
    for the model to write a marker for, so these always land after the
    full answer text, regardless of markers elsewhere in it.
    """
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
    """ask() can return an empty string when the tool-calling loop ends
    without a clean synthesis turn (see chess_agent.ask's docstring) -- the
    chat bubble must show something explaining that, not render nothing.
    """
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
    """Regression test: ask_agent (chess_agent.ask -> the Anthropic SDK's
    own tool_runner) can raise anthropic.APIError for a real, transient
    failure -- a rate limit, a dropped connection, a 5xx from Anthropic --
    none of which is a tool exception, so tool_runner's own "catches every
    exception a tool raises" guarantee doesn't cover it. Before this fix,
    that exception propagated all the way up and crashed the whole
    Streamlit script instead of showing the same graceful fallback already
    used for an empty answer.
    """
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
    """The actual regression this redesign fixes: a show_opening_line-style
    diagram (a labeled FEN reported through on_position) has to land at its
    [[diagram: ...]] marker in the rendered answer, not bunched at the end
    -- exercised through the real ask_with_status path (streaming +
    final replace), not just _render_answer_content in isolation.
    """
    at = AppTest.from_string(f"""
from unittest.mock import patch
from src.ui import chat

def fake_ask_agent(question, on_step=None, on_chunk=None, on_position=None, history=None):
    on_position("{START_FEN}", label="Main line", update_board=False)
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


def test_ask_with_status_does_not_touch_the_free_play_board_during_replay():
    """Real bug found during planning for game-replay: on_position used to
    unconditionally overwrite st.session_state.board whenever a tool call
    reported a position (update_board=True by default). During replay, that
    board is the free-play board sitting *behind* the visible replay
    position, not a copy -- evaluating a replay ply would silently corrupt
    it with no visible symptom until the user exited replay. Must stay
    completely untouched (same object, not just an equal-looking one) while
    st.session_state.game_path is not None.
    """
    at = AppTest.from_string(f"""
from unittest.mock import patch
import chess
import streamlit as st
from src.ui import chat

st.session_state.game_path = ["{START_FEN}", "{OTHER_FEN}"]
st.session_state.game_path_index = 0
original_board = chess.Board("{OTHER_FEN}")
st.session_state.board = original_board

def fake_ask_agent(question, on_step=None, on_chunk=None, on_position=None, history=None):
    on_position("{START_FEN}", update_board=True)
    return "Evaluation done."

with patch("src.ui.chat.ask_agent", fake_ask_agent):
    chat.ask_with_status("Evaluate this position")

st.session_state["_board_is_same_object"] = st.session_state.board is original_board
st.session_state["_board_fen"] = st.session_state.board.fen()
""")
    at.run()
    assert not at.exception
    assert at.session_state["_board_is_same_object"] is True
    assert at.session_state["_board_fen"] == OTHER_FEN


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


def test_clicking_an_example_prompt_submits_it_like_typing_and_enter():
    at = AppTest.from_string("""
from unittest.mock import patch
import streamlit as st
from src.ui import chat

st.session_state.chat_history = []
with patch("src.ui.chat.ask_agent", return_value="An answer."):
    chat._render_example_prompts()
""")
    at.run()
    assert not at.exception
    example_button = next(b for b in at.button if "Sicilian" in b.label)

    example_button.click().run()

    assert not at.exception
    roles = [entry[0] for entry in at.session_state.chat_history]
    assert roles == ["user", "assistant"]
    assert at.session_state.chat_history[0][1] == "How should White meet the Sicilian Defense?"


def test_find_related_resources_shows_a_fallback_instead_of_crashing_on_api_failure():
    """Regression test: recommend_resources chains Anthropic + Voyage + live
    Lichess HTTP behind one button click, and nothing caught a failure in
    any of them -- an anthropic.APIError (rate limit, dropped connection,
    a 5xx) used to crash the whole script. resource_recommendations must
    stay None (not an empty list) on failure, so the button reappears for
    a retry instead of wrongly claiming nothing relevant was found.
    """
    at = AppTest.from_string("""
from unittest.mock import patch
import httpx
import anthropic
import streamlit as st
from src.ui import chat

st.session_state.chat_history = [
    ("user", "How should White meet the Sicilian?", None, []),
    ("assistant", "The Open Sicilian is the main try.", None, []),
]
st.session_state.resource_recommendations = None

error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
with patch("src.ui.chat.recommend_resources", side_effect=error):
    chat._render_resource_recommendations()
""")
    at.run()
    assert not at.exception
    find_button = next(b for b in at.button if b.label == "Find related resources")

    find_button.click().run()

    assert not at.exception
    assert at.session_state.resource_recommendations is None


def test_describe_uploaded_game_shows_a_friendly_error_for_a_header_delimiter_collision():
    """Regression test: compute_game_id (parse_pgn -> parse_game ->
    compute_game_id) deliberately raises ValueError if a PGN header
    contains hash_utils.ID_DELIMITER ("|") -- untrusted user input this
    code is directly reachable from via the chat's PGN attach, and until
    this fix nothing caught it, so a player name containing "|" crashed
    the whole script instead of showing the same honest message a
    genuinely unparseable file already gets.
    """
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
    assert at.session_state["_result"] is None
    assert any("couldn't find a game" in e.value.lower() for e in at.error)
