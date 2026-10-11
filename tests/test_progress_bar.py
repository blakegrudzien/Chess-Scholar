import pytest

from src.ui.progress_bar import (
    CREEP_SECONDS,
    CREEP_TARGET,
    FILL_FRACTION,
    FULL_BAR_SECONDS,
    finishing_bar_html,
    progress_fraction,
    running_bar_html,
)


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (0, 0.0),
        (FULL_BAR_SECONDS / 2, FILL_FRACTION / 2),
        (FULL_BAR_SECONDS, FILL_FRACTION),
        (FULL_BAR_SECONDS + CREEP_SECONDS / 2, (FILL_FRACTION + CREEP_TARGET) / 2),
        (FULL_BAR_SECONDS + CREEP_SECONDS, CREEP_TARGET),
        (10_000, CREEP_TARGET),
    ],
)
def test_progress_fraction_fills_then_creeps_without_ever_reaching_full(elapsed, expected):
    assert progress_fraction(elapsed) == pytest.approx(expected)


def test_progress_never_reaches_full_before_the_answer_arrives():
    assert progress_fraction(10_000) < 1.0


def test_running_bar_keyframes_match_the_python_formula():
    """The browser animates the bar and Python computes where it is at the
    finish; both must follow the same curve."""
    html = running_bar_html()
    fill_point = 100 * FULL_BAR_SECONDS / (FULL_BAR_SECONDS + CREEP_SECONDS)
    assert f"{fill_point:.2f}%" in html
    assert f"{100 * FILL_FRACTION:.2f}%" in html
    assert f"{100 * CREEP_TARGET:.2f}%" in html
    assert f"{FULL_BAR_SECONDS + CREEP_SECONDS:g}s" in html


def test_finishing_bar_starts_from_the_current_position():
    html = finishing_bar_html(elapsed=FULL_BAR_SECONDS / 2)
    assert f"{100 * FILL_FRACTION / 2:.2f}%" in html
    assert "100%" in html
