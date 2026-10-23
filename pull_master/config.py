"""Run settings, repository links, and credential lookup."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MCP_URL = "https://api.githubcopilot.com/mcp/"
DEFAULT_WORKDIR = Path.home() / ".pull_master" / "workspaces"

# Issue authors whose issues are processed unless --allow-any-author is passed.
TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})


class ConfigError(Exception):
    """Bad input or missing credentials. Shown to the user without a traceback."""


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str
    issue: int | None = None  # set when the link pointed at a single issue

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def https_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"


_NAME = r"[A-Za-z0-9_.-]+"
_HTTPS = re.compile(rf"^https?://(?:www\.)?github\.com/(?P<owner>{_NAME})/(?P<repo>{_NAME})(?P<rest>/.*)?$")
_SSH = re.compile(rf"^(?:ssh://)?git@github\.com[:/](?P<owner>{_NAME})/(?P<repo>{_NAME})/?$")
_SHORT = re.compile(rf"^(?P<owner>{_NAME})/(?P<repo>{_NAME})$")
_ISSUE_PATH = re.compile(r"^/issues/(?P<num>\d+)")


def parse_repo(link: str) -> RepoRef:
    """Accepts https/ssh GitHub links, `owner/repo`, and issue links like .../issues/12."""
    text = link.strip()
    match = _HTTPS.match(text) or _SSH.match(text) or _SHORT.match(text)
    if not match:
        raise ConfigError(f"Not a GitHub repository link: {link!r}")
    name = match["repo"]
    if name.endswith(".git"):
        name = name[: -len(".git")]
    issue = None
    rest = match.groupdict().get("rest") or ""
    issue_match = _ISSUE_PATH.match(rest.split("?")[0].split("#")[0])
    if issue_match:
        issue = int(issue_match["num"])
    return RepoRef(owner=match["owner"], name=name, issue=issue)


def resolve_github_token() -> str:
    """GITHUB_TOKEN / GITHUB_PERSONAL_ACCESS_TOKEN, else the token from `gh auth login`."""
    for var in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    if shutil.which("gh"):
        proc = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    raise ConfigError(
        "No GitHub token found. Run `gh auth login`, or export GITHUB_TOKEN "
        "(a token with repo scope)."
    )


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines from ./.env into the environment, without overriding what's set."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def check_anthropic_credentials() -> None:
    """Fail early with a clear message instead of a 401 halfway through a run."""
    if any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")):
        return
    if (Path.home() / ".config" / "anthropic").exists():
        return  # `ant auth login` profile; the SDK picks it up
    raise ConfigError(
        "No Anthropic credentials found. Export ANTHROPIC_API_KEY (or run `ant auth login`)."
    )


@dataclass
class Settings:
    repo: RepoRef
    issues: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    limit: int = 5
    base: str | None = None
    test_cmd: str | None = None
    setup_cmd: str | None = None
    model: str = DEFAULT_MODEL
    effort: str = "high"
    fallbacks: bool = True
    max_attempts: int = 3
    max_turns: int = 80
    merge: bool = True
    merge_method: str = "squash"
    delete_branch: bool = True
    dry_run: bool = False
    list_only: bool = False
    allow_any_author: bool = False
    comment_on_failure: bool = False
    approve_commands: bool = False
    command_timeout: int = 600
    workdir: Path = DEFAULT_WORKDIR
    mcp_url: str = DEFAULT_MCP_URL
    mcp_command: str | None = None
    # Tests point this at a local bare repo; normally it is the GitHub https URL.
    remote_url: str | None = None
    # Give Python projects their own virtualenv (tests turn this off to stay offline).
    python_venv: bool = True

    @property
    def clone_url(self) -> str:
        return self.remote_url or self.repo.https_url
