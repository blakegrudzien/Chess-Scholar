"""check_sync() itself needs a regression test, not just the repo state it
happens to check today -- otherwise a bug in the comparison logic could
silently pass in CI right alongside the drift it exists to catch.
"""

from unittest.mock import patch

from scripts.check_requirements_sync import check_sync


def test_check_sync_passes_against_the_repo_files_as_they_actually_are():
    """The real regression test: run it against this repo's actual
    pyproject.toml/requirements.txt, not fixtures -- if someone edits one
    without the other, this is the test that should fail.
    """
    assert check_sync() == []


def test_check_sync_reports_a_dependency_missing_from_requirements_txt():
    with (
        patch(
            "scripts.check_requirements_sync._pyproject_dependencies",
            return_value={"anthropic>=0.40", "httpx>=0.27"},
        ),
        patch(
            "scripts.check_requirements_sync._requirements_txt_dependencies",
            return_value={"anthropic>=0.40"},
        ),
    ):
        problems = check_sync()
    assert len(problems) == 1
    assert "httpx>=0.27" in problems[0]
    assert "missing from requirements.txt" in problems[0]


def test_check_sync_reports_a_dependency_missing_from_pyproject_toml():
    with (
        patch(
            "scripts.check_requirements_sync._pyproject_dependencies",
            return_value={"anthropic>=0.40"},
        ),
        patch(
            "scripts.check_requirements_sync._requirements_txt_dependencies",
            return_value={"anthropic>=0.40", "httpx>=0.27"},
        ),
    ):
        problems = check_sync()
    assert len(problems) == 1
    assert "httpx>=0.27" in problems[0]
    assert "missing from pyproject.toml" in problems[0]
