"""Fail loudly if pyproject.toml's [project.dependencies] and the root
requirements.txt have drifted apart.

Two copies of the same dependency list exist only because Streamlit
Community Cloud reads requirements.txt specifically and can't be pointed at
pyproject.toml (see both files' own comments) -- pyproject.toml stays the
source of truth for local dev. "Kept in sync by hand" has no enforcement
without this: nothing previously caught the two silently drifting apart
after a dependency change that only touched one file.

Run directly, or via CI (see .github/workflows/ci.yml):
    python -m scripts.check_requirements_sync
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


def _pyproject_dependencies() -> set[str]:
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)
    return set(data["project"]["dependencies"])


def _requirements_txt_dependencies() -> set[str]:
    lines = REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def check_sync() -> list[str]:
    """Returns a list of human-readable mismatch descriptions -- empty if
    the two files declare exactly the same dependency set.
    """
    pyproject_deps = _pyproject_dependencies()
    requirements_deps = _requirements_txt_dependencies()

    problems = []
    only_in_pyproject = pyproject_deps - requirements_deps
    only_in_requirements = requirements_deps - pyproject_deps
    if only_in_pyproject:
        problems.append(
            f"In pyproject.toml but missing from requirements.txt: {sorted(only_in_pyproject)}"
        )
    if only_in_requirements:
        problems.append(
            f"In requirements.txt but missing from pyproject.toml: {sorted(only_in_requirements)}"
        )
    return problems


def main() -> None:
    problems = check_sync()
    if problems:
        print("pyproject.toml and requirements.txt have drifted apart:")
        for problem in problems:
            print(f"  - {problem}")
        print("Update both -- see the comment at the top of either file.")
        sys.exit(1)
    print("pyproject.toml and requirements.txt agree.")


if __name__ == "__main__":
    main()
