"""The pipeline: pick issues -> prepare the workspace -> fix each issue -> test -> PR -> merge."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from pull_master import console, prompts
from pull_master.agent import ClaudeModel, IssueFixer, Model
from pull_master.config import TRUSTED_ASSOCIATIONS, Settings, check_anthropic_credentials, resolve_github_token
from pull_master.github import GitHubError, GitHubMCP
from pull_master.tools import FileEditor, ShellSession
from pull_master.workspace import CommandResult, Workspace, detect_project

# Statuses that count as a failed run for the exit code.
FAILED_STATUSES = {"error", "merge_failed"}


@dataclass
class IssueReport:
    number: int
    title: str
    url: str = ""
    status: str = "pending"
    detail: str = ""
    agent_summary: str = ""
    branch: str | None = None
    pr_url: str | None = None
    merge_sha: str | None = None
    tests: str | None = None
    diffstat: str = ""
    attempts: int = 0
    turns: int = 0
    usage: dict = field(default_factory=dict)


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "fix"


async def run(settings: Settings, *, github: Any | None = None, model: Model | None = None) -> list[IssueReport]:
    """Entry point. `github` and `model` can be injected (the tests pass fakes)."""
    token = resolve_github_token() if github is None or settings.remote_url is None else None
    if github is not None:
        return await _run(settings, github, model, token)
    console.step(f"Connecting to the GitHub MCP server ({settings.mcp_command or settings.mcp_url})")
    async with GitHubMCP(token or "", url=settings.mcp_url, command=settings.mcp_command) as gh:
        console.ok(f"Connected; {len(gh.tools)} tools available")
        return await _run(settings, gh, model, token)


async def _select_issues(settings: Settings, gh: Any, login: str) -> tuple[list[dict], list[IssueReport]]:
    owner, name = settings.repo.owner, settings.repo.name
    explicit = list(settings.issues) or ([settings.repo.issue] if settings.repo.issue else [])
    if explicit:
        candidates = [{"number": n} for n in explicit]
    else:
        # Fetch extra so that skipping untrusted issues still leaves `limit` to work on.
        fetch = settings.limit if settings.allow_any_author else settings.limit * 4
        candidates = await gh.list_open_issues(owner, name, settings.labels, fetch)

    selected: list[dict] = []
    skipped: list[IssueReport] = []
    for candidate in candidates:
        if len(selected) >= settings.limit and not explicit:
            break
        issue = await gh.get_issue(owner, name, int(candidate["number"]))
        reason = None
        if issue["is_pull_request"]:
            reason = "is a pull request"
        elif issue["state"] != "open":
            reason = f"is {issue['state']}"
        elif not settings.allow_any_author and issue["author_association"] not in TRUSTED_ASSOCIATIONS \
                and issue["author"].lower() != login.lower():
            reason = (f"opened by {issue['author']} ({issue['author_association'].lower()}), not an owner, "
                      "member or collaborator; pass --allow-any-author to include it")
        if reason:
            skipped.append(IssueReport(issue["number"], issue["title"], issue["url"], "skipped", reason))
        else:
            selected.append(issue)
    return selected, skipped


async def _run(settings: Settings, gh: Any, model: Model | None, token: str | None) -> list[IssueReport]:
    repo = settings.repo
    try:
        login = await gh.whoami()
    except GitHubError:
        login = ""
    console.step(f"Checking open issues in {repo.slug}" + (f" as {login}" if login else ""))
    selected, reports = await _select_issues(settings, gh, login)
    for r in reports:
        console.info(f"skip #{r.number} {r.title!r}: {r.detail}")
    for issue in selected:
        console.info(f"todo #{issue['number']} {issue['title']!r} (by {issue['author']})")
    if not selected:
        console.warn("No issues to work on.")
        return reports
    if settings.list_only:
        reports += [IssueReport(i["number"], i["title"], i["url"], "selected") for i in selected]
        return reports

    if model is None:
        check_anthropic_credentials()
        model = ClaudeModel(settings.model, settings.effort, settings.fallbacks)

    console.step("Preparing the local sandbox")
    workspace = Workspace(settings.workdir / f"{repo.owner}__{repo.name}", settings.clone_url, token)
    workspace.prepare()
    base = settings.base or workspace.remote_default_branch()
    workspace.reset_to(base)
    console.info(f"Workspace: {workspace.repo_dir} (base branch {base})")

    project = detect_project(workspace.repo_dir)
    test_cmd = settings.test_cmd or project.test_cmd
    console.info(f"Project type: {project.kind}; test command: {test_cmd or 'not detected (the agent will look)'}")
    if project.python and settings.python_venv:
        workspace.ensure_python_env()
    setup_cmds = [settings.setup_cmd] if settings.setup_cmd else project.setup_cmds
    for cmd in setup_cmds:
        console.info(f"setup: {cmd}")
        result = await workspace.run(cmd, settings.command_timeout)
        if not result.ok:
            console.warn(f"setup step {result.describe()}; the agent can fix the environment\n{result.tail(1500)}")

    baseline: CommandResult | None = None
    if test_cmd:
        console.info(f"Running the tests on {base} first: {test_cmd}")
        baseline = await workspace.run(test_cmd, settings.command_timeout)
        (console.ok if baseline.ok else console.warn)(f"baseline tests {baseline.describe()}")

    run_dir = workspace.root / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    for issue in selected:
        report = IssueReport(issue["number"], issue["title"], issue["url"])
        reports.append(report)
        try:
            await _fix_issue(settings, gh, model, workspace, base, test_cmd, baseline, issue, report, run_dir)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            report.status, report.detail = "error", "Anthropic API credentials were rejected"
            raise
        except Exception as err:
            report.status, report.detail = "error", f"{type(err).__name__}: {err}"
            console.error(f"#{issue['number']}: {report.detail}")
        finally:
            try:
                workspace.reset_to(base)
            except Exception as err:
                console.warn(f"Could not reset the workspace: {err}")

    _write_reports(run_dir, settings, base, test_cmd, reports)
    _print_summary(reports, run_dir)
    return reports


async def _fix_issue(
    settings: Settings,
    gh: Any,
    model: Model,
    workspace: Workspace,
    base: str,
    test_cmd: str | None,
    baseline: CommandResult | None,
    issue: dict,
    report: IssueReport,
    run_dir: Path,
) -> None:
    repo = settings.repo
    number = issue["number"]
    console.step(f"Issue #{number}: {issue['title']}")
    comments = await gh.get_comments(repo.owner, repo.name, number)
    branch = f"pull-master/issue-{number}-{slugify(issue['title'])}"
    workspace.start_branch(branch, base)
    report.branch = branch

    shell = ShellSession(
        workspace.repo_dir, workspace.sandbox_env(), workspace.root / "shell",
        settings.command_timeout, settings.approve_commands,
    )
    fixer = IssueFixer(
        model=model,
        workspace=workspace,
        shell=shell,
        editor=FileEditor(workspace.repo_dir),
        github=gh,
        base=base,
        test_cmd=test_cmd,
        max_attempts=settings.max_attempts,
        max_turns=settings.max_turns,
        command_timeout=settings.command_timeout,
        transcript=run_dir / f"issue-{number}.transcript.jsonl",
    )
    kickoff = prompts.issue_prompt(
        repo_slug=repo.slug, base=base, repo_dir=str(workspace.repo_dir), issue=issue,
        comments=comments, test_cmd=test_cmd, baseline=baseline,
    )
    try:
        result = await fixer.run(kickoff)
    finally:
        await shell.close()

    report.attempts, report.turns, report.usage = result.attempts, result.turns, result.usage
    if result.test_result is not None:
        report.tests = f"`{result.test_result.command}` {result.test_result.describe()}"

    report.agent_summary = result.summary
    if result.outcome != "fixed":
        report.status = result.outcome
        report.detail = {
            "needs_human": "The agent says a person should look at this (see agent_summary).",
            "tests_failed": f"Tests still failed after {result.attempts} attempt(s); nothing was merged.",
            "refused": result.summary,
        }.get(result.outcome, f"Stopped without a passing fix. {result.summary[:200]}")
        console.warn(f"#{number}: {report.detail}\n{result.summary[:400]}")
        if settings.comment_on_failure and not settings.dry_run:
            await gh.comment(repo.owner, repo.name, number, _failure_comment(result.outcome, result.summary, settings))
        return

    commit_message = f"{result.title}\n\nFixes #{number}\n\n{result.summary}"
    workspace.commit_all(commit_message)
    report.diffstat = workspace.diff(base, stat=True).rstrip()
    (run_dir / f"issue-{number}.patch").write_text(workspace.diff(base))
    console.ok(f"Fix committed on {branch}\n{report.diffstat}")

    if settings.dry_run:
        report.status = "fixed_local"
        report.detail = f"Dry run: committed on local branch {branch} in {workspace.repo_dir}; nothing pushed."
        return

    workspace.push(branch)
    pr = await gh.create_pull_request(
        repo.owner, repo.name, title=result.title, head=branch, base=base,
        body=_pr_body(number, result.summary, report, settings),
    )
    report.pr_url = pr.url
    console.ok(f"Opened pull request #{pr.number} {pr.url}")
    if not settings.merge:
        report.status, report.detail = "pr_open", "Left open for review (--no-merge)."
        return

    try:
        sha = await gh.merge_pull_request(
            repo.owner, repo.name, pr.number, method=settings.merge_method,
            title=f"{result.title} (#{pr.number})", message=f"Fixes #{number}",
        )
    except GitHubError as err:
        report.status, report.detail = "merge_failed", f"Pull request is open but could not be merged: {err}"
        console.error(report.detail)
        return
    report.status, report.merge_sha = "merged", sha
    report.detail = f"Merged into {base}."
    console.ok(f"Merged #{pr.number} into {base}")
    if settings.delete_branch:
        workspace.delete_remote_branch(branch)


def _pr_body(number: int, summary: str, report: IssueReport, settings: Settings) -> str:
    return (
        f"Fixes #{number}\n\n{summary}\n\n---\n"
        f"**Tests:** {report.tests or 'not run'} (local run, attempt {report.attempts}).\n\n"
        f"<details><summary>Files changed</summary>\n\n```\n{report.diffstat}\n```\n</details>\n\n"
        f"_Automated fix by pull-master using {settings.model}._"
    )


def _failure_comment(outcome: str, summary: str, settings: Settings) -> str:
    headline = {
        "needs_human": "pull-master looked at this issue and thinks it needs a person.",
        "tests_failed": "pull-master tried to fix this issue, but the tests still failed, so nothing was merged.",
    }.get(outcome, "pull-master could not resolve this issue automatically.")
    return f"{headline}\n\n{summary}\n\n_Automated note from pull-master using {settings.model}._"


def _write_reports(run_dir: Path, settings: Settings, base: str, test_cmd: str | None, reports: list[IssueReport]) -> None:
    data = {
        "repo": settings.repo.slug,
        "base": base,
        "model": settings.model,
        "effort": settings.effort,
        "test_command": test_cmd,
        "dry_run": settings.dry_run,
        "issues": [asdict(r) for r in reports],
    }
    (run_dir / "report.json").write_text(json.dumps(data, indent=2))
    lines = [f"# pull-master run on {settings.repo.slug}", "", f"Base `{base}`, model `{settings.model}`.", "",
             "| Issue | Status | Details |", "|---|---|---|"]
    for r in reports:
        detail = (r.pr_url or r.detail or "").replace("\n", " ").replace("|", "\\|")[:200]
        lines.append(f"| [#{r.number}]({r.url}) {r.title.replace('|', '/')} | {r.status} | {detail} |")
    for r in reports:
        if r.agent_summary:
            lines += ["", f"## #{r.number} {r.title}", "", f"Status: **{r.status}**. Tests: {r.tests or 'not run'}.",
                      "", r.agent_summary]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _print_summary(reports: list[IssueReport], run_dir: Path) -> None:
    console.step("Summary")
    for r in reports:
        extra = r.pr_url or r.detail.split("\n")[0][:120]
        console.info(f"#{r.number:<5} {r.status:<13} {extra}")
    tokens = {}
    for r in reports:
        for k, v in r.usage.items():
            tokens[k] = tokens.get(k, 0) + v
    if tokens:
        console.info("tokens: " + ", ".join(f"{k.replace('_input_tokens', '').replace('_tokens', '')}={v:,}" for k, v in tokens.items()))
    console.info(f"Report and transcripts: {run_dir}")
