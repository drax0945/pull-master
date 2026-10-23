import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def git_identity(monkeypatch):
    """Commits made in tests shouldn't depend on the machine's git config."""
    for key, value in {
        "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.com",
    }.items():
        monkeypatch.setenv(key, value)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


SAMPLE_FILES = {
    "calc.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n",
    "tests/test_calc.py": "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n",
}


@pytest.fixture
def bare_remote(tmp_path) -> Path:
    """A bare git repo standing in for GitHub, holding a small project with a bug in add()."""
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "init", "-q", "-b", "main")
    for name, text in SAMPLE_FILES.items():
        path = seed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(seed, "add", "-A")
    git(seed, "commit", "-q", "-m", "initial")
    bare = tmp_path / "remote.git"
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(bare))
    return bare


def read_remote_file(bare: Path, branch: str, path: str) -> str:
    return git(bare, "show", f"{branch}:{path}")


os.environ.setdefault("NO_COLOR", "1")
