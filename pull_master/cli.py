"""Command line: `pull-master https://github.com/owner/repo [options]`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import anthropic

from pull_master import console
from pull_master.config import (
    DEFAULT_MCP_URL,
    DEFAULT_MODEL,
    DEFAULT_WORKDIR,
    ConfigError,
    Settings,
    load_dotenv,
    parse_repo,
)
from pull_master.github import GitHubError
from pull_master.runner import FAILED_STATUSES, run
from pull_master.workspace import WorkspaceError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pull-master",
        description="Read a repository's GitHub issues over the GitHub MCP server, fix them locally with Claude, "
        "run the tests, and merge the fixes that pass.",
    )
    p.add_argument("repo", help="GitHub link: https://github.com/owner/repo, owner/repo, or an issue link")

    pick = p.add_argument_group("which issues")
    pick.add_argument("--issue", type=int, action="append", default=[], metavar="N", help="fix this issue (repeatable)")
    pick.add_argument("--label", action="append", default=[], metavar="NAME", help="only issues with this label (repeatable)")
    pick.add_argument("--limit", type=int, default=5, help="max issues per run (default 5)")
    pick.add_argument("--allow-any-author", action="store_true",
                      help="also fix issues opened by people who aren't owners, members or collaborators")
    pick.add_argument("--list", dest="list_only", action="store_true", help="only show which issues would be worked on")

    flow = p.add_argument_group("what to do with fixes")
    flow.add_argument("--base", help="branch to merge into (default: the repo's default branch)")
    flow.add_argument("--no-merge", dest="merge", action="store_false", help="open pull requests but don't merge them")
    flow.add_argument("--merge-method", choices=["squash", "merge", "rebase"], default="squash")
    flow.add_argument("--keep-branch", dest="delete_branch", action="store_false", help="keep the fix branch after merging")
    flow.add_argument("--dry-run", action="store_true", help="fix and commit locally only; no push, PR, merge or comments")
    flow.add_argument("--comment-on-failure", action="store_true",
                      help="comment on issues the agent couldn't fix or flagged for a person")

    env = p.add_argument_group("tests and sandbox")
    env.add_argument("--test-cmd", help="test command (default: detected from the project)")
    env.add_argument("--setup-cmd", help="setup command run once before testing (default: detected)")
    env.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help=f"where clones live (default {DEFAULT_WORKDIR})")
    env.add_argument("--command-timeout", type=int, default=600, help="seconds before a command is killed (default 600)")
    env.add_argument("--approve-commands", action="store_true", help="ask before every shell command the agent runs")

    agent = p.add_argument_group("agent")
    agent.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default {DEFAULT_MODEL})")
    agent.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="high",
                       help="how hard the model thinks (default high)")
    agent.add_argument("--no-fallback", dest="fallbacks", action="store_false",
                       help="don't retry on a fallback model when a safety classifier declines a turn")
    agent.add_argument("--max-attempts", type=int, default=3, help="test-and-retry rounds per issue (default 3)")
    agent.add_argument("--max-turns", type=int, default=80, help="model turns per issue (default 80)")

    mcp = p.add_argument_group("GitHub MCP server")
    mcp.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help=f"remote server URL (default {DEFAULT_MCP_URL})")
    mcp.add_argument("--mcp-command",
                     help="run a local server over stdio instead, e.g. "
                     "'docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server stdio'")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    try:
        settings = Settings(
            repo=parse_repo(args.repo),
            issues=args.issue,
            labels=args.label,
            limit=args.limit,
            base=args.base,
            test_cmd=args.test_cmd,
            setup_cmd=args.setup_cmd,
            model=args.model,
            effort=args.effort,
            fallbacks=args.fallbacks,
            max_attempts=args.max_attempts,
            max_turns=args.max_turns,
            merge=args.merge,
            merge_method=args.merge_method,
            delete_branch=args.delete_branch,
            dry_run=args.dry_run,
            list_only=args.list_only,
            allow_any_author=args.allow_any_author,
            comment_on_failure=args.comment_on_failure,
            approve_commands=args.approve_commands,
            command_timeout=args.command_timeout,
            workdir=args.workdir.expanduser(),
            mcp_url=args.mcp_url,
            mcp_command=args.mcp_command,
        )
        reports = asyncio.run(run(settings))
    except (ConfigError, GitHubError, WorkspaceError) as err:
        console.error(str(err))
        return 2
    except anthropic.AuthenticationError:
        console.error("The Anthropic API rejected the credentials. Check ANTHROPIC_API_KEY.")
        return 2
    except KeyboardInterrupt:
        console.error("Interrupted.")
        return 130
    return 1 if any(r.status in FAILED_STATUSES for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
