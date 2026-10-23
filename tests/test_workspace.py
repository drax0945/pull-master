import asyncio
import json

import pytest

from conftest import git, read_remote_file
from pull_master.workspace import Workspace, detect_project


def test_detect_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    info = detect_project(tmp_path)
    assert info.kind == "python" and info.python and info.test_cmd == "python -m pytest -q"
    assert any("pip install" in c for c in info.setup_cmds)


def test_detect_node_with_test_script(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))
    (tmp_path / "package-lock.json").write_text("{}")
    info = detect_project(tmp_path)
    assert (info.kind, info.test_cmd, info.setup_cmds) == ("node", "npm test", ["npm ci"])


def test_detect_node_placeholder_test_script_is_ignored(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": 'echo "Error: no test specified" && exit 1'}}))
    assert detect_project(tmp_path).test_cmd is None


@pytest.mark.parametrize("marker, cmd", [("go.mod", "go test ./..."), ("Cargo.toml", "cargo test")])
def test_detect_other_languages(tmp_path, marker, cmd):
    (tmp_path / marker).write_text("")
    assert detect_project(tmp_path).test_cmd == cmd


def test_detect_unknown(tmp_path):
    (tmp_path / "README.md").write_text("hi")
    assert detect_project(tmp_path).test_cmd is None


def test_sandbox_env_drops_secrets(tmp_path, monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "MY_SERVICE_PASSWORD", "AWS_PROFILE", "SSH_AUTH_SOCK"):
        monkeypatch.setenv(key, "x")
    monkeypatch.setenv("HARMLESS", "1")
    env = Workspace(tmp_path, "https://github.com/a/b.git", "tok").sandbox_env()
    for key in ("ANTHROPIC_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "MY_SERVICE_PASSWORD", "AWS_PROFILE", "SSH_AUTH_SOCK"):
        assert key not in env
    assert env["HARMLESS"] == "1" and "PATH" in env
    assert env["GIT_CONFIG_KEY_0"] == "credential.helper" and env["GIT_CONFIG_VALUE_0"] == ""
    assert env["GH_CONFIG_DIR"].startswith(str(tmp_path))


def test_clone_branch_commit_push(tmp_path, bare_remote):
    ws = Workspace(tmp_path / "ws", str(bare_remote), None)
    ws.prepare()
    assert ws.remote_default_branch() == "main"
    ws.start_branch("fix/one", "main")
    assert not ws.has_changes("main")

    (ws.repo_dir / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (ws.repo_dir / "__pycache__").mkdir()
    (ws.repo_dir / "__pycache__" / "junk.pyc").write_text("x")
    assert ws.has_changes("main")
    ws.commit_all("Fix add")
    assert "calc.py" in ws.diff("main", stat=True)
    assert "__pycache__" not in ws.diff("main", stat=True)  # local excludes keep junk out

    ws.push("fix/one")
    assert "a + b" in read_remote_file(bare_remote, "fix/one", "calc.py")
    ws.reset_to("main")
    assert "a - b" in (ws.repo_dir / "calc.py").read_text()


def test_git_config_edits_are_undone(tmp_path, bare_remote):
    ws = Workspace(tmp_path / "ws", str(bare_remote), None)
    ws.prepare()
    ws.start_branch("fix/two", "main")
    marker = tmp_path / "pwned"
    # An agent (or a prompt-injected issue) adds a filter that would run code on `git add`.
    git(ws.repo_dir, "config", "filter.evil.clean", f"touch {marker}; cat")
    (ws.repo_dir / ".gitattributes").write_text("*.py filter=evil\n")
    (ws.repo_dir / "calc.py").write_text("changed\n")
    ws.commit_all("change")
    assert not marker.exists()


def test_run_captures_output_and_timeout(tmp_path):
    (tmp_path / "ws" / "repo").mkdir(parents=True)
    ws = Workspace(tmp_path / "ws", "unused", None)
    ok = asyncio.run(ws.run("echo hello; echo err >&2", 10))
    assert ok.ok and "hello" in ok.output and "err" in ok.output
    slow = asyncio.run(ws.run("echo start; sleep 5", 1))
    assert slow.timed_out and not slow.ok and "start" in slow.output
