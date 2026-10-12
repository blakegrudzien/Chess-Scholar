"""Tests check_sync() itself, so a bug in its comparison can't pass
unnoticed along with the drift it should catch.
"""

from unittest.mock import patch

from scripts.check_requirements_sync import check_sync


def test_check_sync_passes_against_the_repo_files_as_they_actually_are():
    """Runs against the repo's real pyproject.toml and requirements.txt, so
    editing one without the other fails here."""
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
