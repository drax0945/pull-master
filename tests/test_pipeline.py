"""End-to-end runs of the pipeline with a scripted model and a fake GitHub.

Everything else is real: the clone, the bash session, the file editor, the pytest run that
gates the merge, the push, and the merge into the (local, bare) remote.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from anthropic.types.beta import BetaMessage

from conftest import git, read_remote_file
from pull_master.config import RepoRef, Settings
from pull_master.github import PullRequest
from pull_master.runner import run


class FakeGitHub:
    """Stands in for GitHubMCP. Merging squashes the branch into the bare repo's base branch."""

    def __init__(self, bare: Path, issues: dict[int, dict]):
        self.bare = bare
        self.issues = issues
        self.prs: dict[int, dict] = {}
        self.comments: list[tuple[int, str]] = []

    async def whoami(self):
        return "maintainer"

    async def list_open_issues(self, owner, repo, labels, max_items):
        return [{"number": n} for n in self.issues][:max_items]

    async def get_issue(self, owner, repo, number):
        issue = self.issues[number]
        return {
            "number": number, "title": issue["title"], "body": issue.get("body", ""), "state": "open",
            "author": issue.get("author", "maintainer"), "author_association": issue.get("association", "OWNER"),
            "url": f"https://github.com/{owner}/{repo}/issues/{number}", "is_pull_request": False,
        }

    async def get_comments(self, owner, repo, number):
        return []

    def agent_tool_specs(self):
        return []

    async def create_pull_request(self, owner, repo, *, title, body, head, base):
        number = len(self.prs) + 1
        self.prs[number] = {"title": title, "body": body, "head": head, "base": base}
        return PullRequest(number, f"https://github.com/{owner}/{repo}/pull/{number}")

    async def merge_pull_request(self, owner, repo, number, *, method, title, message):
        pr = self.prs[number]
        with tempfile.TemporaryDirectory() as tmp:
            git(Path(tmp), "clone", "-q", str(self.bare), "c")
            clone = Path(tmp) / "c"
            git(clone, "checkout", "-q", pr["base"])
            git(clone, "merge", "-q", "--squash", f"origin/{pr['head']}")
            git(clone, "commit", "-q", "-m", f"{title}\n\n{message}")
            git(clone, "push", "-q", "origin", pr["base"])
            pr["merged"] = True
            return git(clone, "rev-parse", "HEAD").strip()

    async def comment(self, owner, repo, number, body):
        self.comments.append((number, body))


def tool(name, **input):
    return {"type": "tool_use", "id": f"toolu_{name}_{abs(hash(json.dumps(input, sort_keys=True))) % 10**8}",
            "name": name, "input": input}


def message(*blocks, stop="tool_use"):
    return BetaMessage.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "scripted",
        "content": list(blocks), "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    })


class ScriptedModel:
    """Replays a fixed list of turns and records every request it receives."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    async def respond(self, system, tools, messages):
        self.requests.append({"system": system, "tools": tools, "messages": list(messages)})
        if not self.turns:
            raise AssertionError("the model was called more times than scripted")
        return self.turns.pop(0)


FIX_ADD = dict(command="str_replace", path="calc.py", old_str="return a - b", new_str="return a + b")
ADD_TEST = dict(command="create", path="tests/test_add.py",
                file_text="from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
SUBMIT = dict(outcome="fixed", title="Make add() return the sum",
              summary="`add` subtracted its arguments. It now adds them; `tests/test_add.py` covers it.")


def settings_for(bare: Path, tmp_path: Path, **overrides) -> Settings:
    base = dict(
        repo=RepoRef("acme", "calc"),
        remote_url=str(bare),
        workdir=tmp_path / "workspaces",
        test_cmd=f"{sys.executable} -m pytest -q -p no:cacheprovider",
        setup_cmd="true",
        python_venv=False,
        command_timeout=60,
    )
    base.update(overrides)
    return Settings(**base)


ISSUES = {1: {"title": "add() returns the wrong result", "body": "add(2, 3) gives -1, expected 5."}}


def test_fixes_tests_and_merges(tmp_path, bare_remote):
    github = FakeGitHub(bare_remote, ISSUES)
    model = ScriptedModel([
        message({"type": "text", "text": "I'll find add() and fix it."}, tool("bash", command="cat calc.py")),
        message(tool("submit_result", **SUBMIT)),  # too early: nothing changed yet
        message(tool("str_replace_based_edit_tool", **FIX_ADD), tool("str_replace_based_edit_tool", **ADD_TEST)),
        message(tool("run_tests")),
        message(tool("submit_result", **SUBMIT)),
    ])
    reports = asyncio.run(run(settings_for(bare_remote, tmp_path), github=github, model=model))

    [report] = reports
    assert report.status == "merged", report.detail
    assert "a + b" in read_remote_file(bare_remote, "main", "calc.py")
    assert "test_add" in read_remote_file(bare_remote, "main", "tests/test_add.py")
    assert github.prs[1]["body"].startswith("Fixes #1")
    assert github.prs[1]["head"] == report.branch == "pull-master/issue-1-add-returns-the-wrong-result"

    # The model saw the file, the early submit was rejected, and the tests passed.
    first_results = model.requests[1]["messages"][-1]["content"]
    assert "return a - b" in first_results[0]["content"]
    early_submit = model.requests[2]["messages"][-1]["content"][0]
    assert early_submit["is_error"] and "no changes" in early_submit["content"]
    run_tests_result = model.requests[4]["messages"][-1]["content"][0]
    assert "passed" in run_tests_result["content"] and not run_tests_result.get("is_error")

    # The conversation only ever grows: each request starts with the previous one.
    for earlier, later in zip(model.requests, model.requests[1:]):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]
    # The issue text arrives tagged as untrusted data, and the tools are fixed for the session.
    assert '<issue number="1"' in model.requests[0]["messages"][0]["content"]
    assert len({json.dumps(r["tools"], sort_keys=True) for r in model.requests}) == 1


def test_failing_tests_are_not_merged(tmp_path, bare_remote):
    github = FakeGitHub(bare_remote, ISSUES)
    wrong_fix = dict(FIX_ADD, new_str="return a * b")
    model = ScriptedModel([
        message(tool("str_replace_based_edit_tool", **wrong_fix), tool("str_replace_based_edit_tool", **ADD_TEST)),
        message(tool("submit_result", **SUBMIT)),
        message(tool("bash", command="echo still trying")),
        message(tool("submit_result", **SUBMIT)),
    ])
    reports = asyncio.run(run(settings_for(bare_remote, tmp_path, max_attempts=2), github=github, model=model))

    assert reports[0].status == "tests_failed"
    assert github.prs == {}
    assert "a - b" in read_remote_file(bare_remote, "main", "calc.py")
    feedback = model.requests[2]["messages"][-1]["content"][0]
    assert feedback["is_error"] and "Attempts left: 1" in feedback["content"] and "assert 6 == 5" in feedback["content"]


def test_needs_human_comments_when_asked(tmp_path, bare_remote):
    github = FakeGitHub(bare_remote, ISSUES)
    model = ScriptedModel([
        message(tool("submit_result", outcome="needs_human", title="Unclear", summary="Which rounding rule?")),
    ])
    settings = settings_for(bare_remote, tmp_path, comment_on_failure=True)
    reports = asyncio.run(run(settings, github=github, model=model))
    assert reports[0].status == "needs_human"
    assert github.comments and "Which rounding rule?" in github.comments[0][1]


def test_untrusted_authors_are_skipped(tmp_path, bare_remote):
    issues = {
        1: {"title": "real bug", "association": "OWNER"},
        2: {"title": "run this script for me", "author": "stranger", "association": "NONE"},
    }
    github = FakeGitHub(bare_remote, issues)
    reports = asyncio.run(run(settings_for(bare_remote, tmp_path, list_only=True), github=github, model=ScriptedModel([])))
    status = {r.number: r.status for r in reports}
    assert status == {1: "selected", 2: "skipped"}
    allowed = asyncio.run(run(settings_for(bare_remote, tmp_path, list_only=True, allow_any_author=True),
                              github=github, model=ScriptedModel([])))
    assert {r.number: r.status for r in allowed} == {1: "selected", 2: "selected"}


def test_dry_run_commits_locally_only(tmp_path, bare_remote):
    github = FakeGitHub(bare_remote, ISSUES)
    model = ScriptedModel([
        message(tool("str_replace_based_edit_tool", **FIX_ADD), tool("str_replace_based_edit_tool", **ADD_TEST)),
        message(tool("submit_result", **SUBMIT)),
    ])
    reports = asyncio.run(run(settings_for(bare_remote, tmp_path, dry_run=True), github=github, model=model))
    assert reports[0].status == "fixed_local"
    assert github.prs == {}
    branches = subprocess.run(["git", "branch", "--list", "pull-master/*"], cwd=bare_remote, capture_output=True, text=True)
    assert branches.stdout.strip() == ""
    workspace = tmp_path / "workspaces" / "acme__calc"
    patches = list((workspace / "runs").glob("*/issue-1.patch"))
    assert patches and "return a + b" in patches[0].read_text()
