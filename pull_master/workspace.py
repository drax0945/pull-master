"""The local sandbox: a harness-owned clone of the repo, plus the environment the agent runs in.

"Sandbox" here means your own machine. What this module does to keep the agent contained:
- The agent works in a scratch clone under ~/.pull_master/workspaces, never in your own checkout.
- Its shell gets an environment with tokens and keys removed, git credential helpers
  switched off, and the GitHub CLI pointed at an empty config, so it can't push or call GitHub.
- Only the harness pushes. It pushes to the GitHub URL directly (never to a remote the agent
  could have edited), with hooks disabled and .git/config restored from a snapshot first,
  and the token is only in the environment of that one git process.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pull_master import console

OUTPUT_LIMIT = 30_000

_SECRET_NAME = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIAL|COOKIE)", re.I
)
_DROP_PREFIXES = ("ANTHROPIC_", "GITHUB_", "GH_", "AWS_", "OPENAI_", "SSH_AUTH_SOCK", "SSH_AGENT_PID")
_DROP_EXACT = {"GIT_ASKPASS", "SSH_ASKPASS", "PYTHONHOME", "PULL_MASTER_GH_TOKEN"}

# Files the agent or the test run tend to leave behind; kept out of commits.
_LOCAL_EXCLUDES = """\
# added by pull-master
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
*.egg-info/
.venv/
node_modules/
.DS_Store
"""

_CREDENTIAL_HELPER = (
    '!f() { test "$1" = get && echo username=x-access-token && echo "password=$PULL_MASTER_GH_TOKEN"; }; f'
)


class WorkspaceError(Exception):
    pass


@dataclass
class CommandResult:
    command: str
    exit_code: int | None
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def tail(self, chars: int = 6000) -> str:
        return self.output if len(self.output) <= chars else "...\n" + self.output[-chars:]

    def describe(self) -> str:
        if self.timed_out:
            return "timed out"
        return "passed" if self.ok else f"failed (exit {self.exit_code})"


@dataclass
class ProjectInfo:
    kind: str
    test_cmd: str | None
    setup_cmds: list[str] = field(default_factory=list)
    python: bool = False


def truncate_middle(text: str, limit: int = OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n\n... [{len(text) - limit} characters cut] ...\n\n{text[-tail:]}"


class Workspace:
    def __init__(self, root: Path, remote_url: str, token: str | None):
        self.root = root
        self.repo_dir = root / "repo"
        self.venv_dir = root / "sandbox-venv"
        self.gh_sandbox = root / "gh-sandbox"
        self.remote_url = remote_url
        self._token = token
        self._config_snapshot: bytes | None = None

    # -- git ---------------------------------------------------------------

    def git(self, *args: str, auth: bool = False, check: bool = True) -> subprocess.CompletedProcess:
        cmd = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false"]
        env = {k: v for k, v in os.environ.items() if k != "PULL_MASTER_GH_TOKEN"}
        env["GIT_TERMINAL_PROMPT"] = "0"
        if auth and self._token and self.remote_url.startswith("https://github.com/"):
            cmd += [
                "-c", "credential.helper=",
                "-c", "credential.https://github.com.helper=",
                "-c", f"credential.https://github.com.helper={_CREDENTIAL_HELPER}",
            ]
            env["PULL_MASTER_GH_TOKEN"] = self._token
        cmd += list(args)
        cwd = self.repo_dir if (self.repo_dir / ".git").exists() else self.root
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()[-800:]
            raise WorkspaceError(f"git {' '.join(args[:3])} failed: {detail}")
        return proc

    def _restore_git_config(self) -> None:
        """Undo any edits to .git/config (filters, fsmonitor, remotes) before the harness runs git."""
        if self._config_snapshot is not None:
            (self.repo_dir / ".git" / "config").write_bytes(self._config_snapshot)

    def remote_default_branch(self) -> str:
        out = self.git("ls-remote", "--symref", self.remote_url, "HEAD", auth=True).stdout
        match = re.search(r"ref: refs/heads/(\S+)\s+HEAD", out)
        if not match:
            raise WorkspaceError(f"Could not find the default branch of {self.remote_url}")
        return match.group(1)

    def prepare(self) -> None:
        """Clone on first use; otherwise make sure the folder really is our clone."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.gh_sandbox.mkdir(exist_ok=True)
        if not (self.repo_dir / ".git").exists():
            if self.repo_dir.exists() and any(self.repo_dir.iterdir()):
                raise WorkspaceError(f"{self.repo_dir} exists and is not a git clone; remove it or pick another --workdir")
            console.info(f"Cloning {self.remote_url} into {self.repo_dir}")
            self.git("clone", "--quiet", self.remote_url, str(self.repo_dir), auth=True)
        else:
            remote = self.git("config", "--get", "remote.origin.url", check=False).stdout.strip()
            if remote and remote.removesuffix(".git") != self.remote_url.removesuffix(".git"):
                raise WorkspaceError(
                    f"{self.repo_dir} is a clone of {remote}, not {self.remote_url}. Use another --workdir."
                )
        exclude = self.repo_dir / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        current = exclude.read_text() if exclude.exists() else ""
        if "added by pull-master" not in current:
            exclude.write_text(current.rstrip("\n") + "\n" + _LOCAL_EXCLUDES)
        # Snapshot .git/config right after the first clone and keep it outside the repo, so a
        # later run still restores the clean version even if an earlier run was interrupted.
        snapshot = self.root / "git-config.snapshot"
        if not snapshot.exists():
            snapshot.write_bytes((self.repo_dir / ".git" / "config").read_bytes())
        self._config_snapshot = snapshot.read_bytes()

    def reset_to(self, base: str) -> None:
        """Fetch the latest base branch and make the working tree match it exactly."""
        self._restore_git_config()
        self.git("fetch", "--quiet", self.remote_url, f"+refs/heads/{base}:refs/remotes/origin/{base}", auth=True)
        self.git("checkout", "--quiet", "-f", "-B", base, f"origin/{base}")
        self.git("clean", "-fdq")

    def start_branch(self, branch: str, base: str) -> None:
        self.reset_to(base)
        self.git("checkout", "--quiet", "-B", branch, f"origin/{base}")

    def has_changes(self, base: str) -> bool:
        self._restore_git_config()
        if self.git("status", "--porcelain").stdout.strip():
            return True
        ahead = self.git("rev-list", "--count", f"origin/{base}..HEAD").stdout.strip()
        return ahead not in ("", "0")

    def commit_all(self, message: str) -> str:
        self._restore_git_config()
        self.git("add", "-A")
        staged = self.git("diff", "--cached", "--quiet", check=False).returncode != 0
        if staged:
            identity: list[str] = []
            if not self.git("config", "user.email", check=False).stdout.strip():
                identity = ["-c", "user.name=pull-master", "-c", "user.email=pull-master@users.noreply.github.com"]
            self.git(*identity, "commit", "--quiet", "--no-verify", "-m", message)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def diff(self, base: str, stat: bool = False) -> str:
        args = ["diff", "--stat"] if stat else ["diff"]
        return self.git(*args, f"origin/{base}...HEAD").stdout

    def push(self, branch: str) -> None:
        self._restore_git_config()
        self.git("push", "--quiet", "--force", self.remote_url, f"HEAD:refs/heads/{branch}", auth=True)

    def delete_remote_branch(self, branch: str) -> None:
        self.git("push", "--quiet", self.remote_url, "--delete", branch, auth=True, check=False)

    # -- environment the agent runs in ------------------------------------

    def sandbox_env(self) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if not _SECRET_NAME.search(k) and not k.startswith(_DROP_PREFIXES) and k not in _DROP_EXACT
        }
        if (self.venv_dir / "bin").exists():
            env["VIRTUAL_ENV"] = str(self.venv_dir)
            env["PATH"] = f"{self.venv_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        env.update(
            {
                "GH_CONFIG_DIR": str(self.gh_sandbox),  # gh sees no login
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_COUNT": "2",  # no git credential helpers
                "GIT_CONFIG_KEY_0": "credential.helper",
                "GIT_CONFIG_VALUE_0": "",
                "GIT_CONFIG_KEY_1": "credential.https://github.com.helper",
                "GIT_CONFIG_VALUE_1": "",
                "CI": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PAGER": "cat",
                "GIT_PAGER": "cat",
                "TERM": "dumb",
            }
        )
        return env

    def ensure_python_env(self) -> None:
        if (self.venv_dir / "bin" / "python").exists():
            return
        console.info(f"Creating an isolated Python environment at {self.venv_dir}")
        uv = shutil.which("uv")
        if uv:
            cmd = [uv, "venv", "--quiet", "--seed", str(self.venv_dir)]
        else:
            cmd = [shutil.which("python3") or "python3", "-m", "venv", str(self.venv_dir)]
        proc = subprocess.run(cmd, cwd=self.repo_dir, capture_output=True, text=True)
        if proc.returncode != 0:
            console.warn(f"Could not create the Python environment: {proc.stderr.strip()[-300:]}")

    async def run(self, command: str, timeout: int) -> CommandResult:
        """Run a shell command in the repo with the sandbox environment (setup and test runs)."""
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash", "-c", command,
            cwd=self.repo_dir,
            env=self.sandbox_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        chunks: list[bytes] = []

        async def pump() -> None:
            assert proc.stdout is not None
            while chunk := await proc.stdout.read(65536):
                chunks.append(chunk)

        timed_out = False
        try:
            await asyncio.wait_for(asyncio.gather(pump(), proc.wait()), timeout)
        except (TimeoutError, asyncio.TimeoutError):
            timed_out = True
            kill_process_group(proc.pid)
            await proc.wait()
        output = truncate_middle(b"".join(chunks).decode("utf-8", "replace"))
        return CommandResult(command, proc.returncode, output, timed_out)


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# -- project detection -------------------------------------------------------


def _python_project(repo: Path) -> bool:
    markers = ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini")
    if any((repo / m).exists() for m in markers) or list(repo.glob("requirements*.txt")):
        return True
    return any(repo.glob("*.py")) or any(repo.glob("tests/**/*.py")) or any(repo.glob("test/**/*.py"))


def _python_setup(repo: Path) -> list[str]:
    cmds = []
    if (repo / "pyproject.toml").exists() or (repo / "setup.py").exists():
        # Missing extras only produce a warning, so asking for all common ones is safe.
        cmds.append('python -m pip install -q -e ".[test,tests,testing,dev]" || python -m pip install -q -e . || true')
    for req in sorted(repo.glob("*requirements*.txt")):
        cmds.append(f"python -m pip install -q -r {req.name} || true")
    cmds.append("python -m pip install -q pytest")
    return cmds


def _node_setup(repo: Path) -> list[str]:
    if (repo / "pnpm-lock.yaml").exists() and shutil.which("pnpm"):
        return ["pnpm install --frozen-lockfile"]
    if (repo / "yarn.lock").exists() and shutil.which("yarn"):
        return ["yarn install --frozen-lockfile"]
    if (repo / "package-lock.json").exists():
        return ["npm ci"]
    return ["npm install"]


def detect_project(repo: Path) -> ProjectInfo:
    package_json = repo / "package.json"
    if package_json.exists():
        try:
            scripts = json.loads(package_json.read_text()).get("scripts", {})
        except (json.JSONDecodeError, OSError):
            scripts = {}
        test = scripts.get("test", "")
        if test and "no test specified" not in test:
            return ProjectInfo("node", "npm test", _node_setup(repo))
    if _python_project(repo):
        return ProjectInfo("python", "python -m pytest -q", _python_setup(repo), python=True)
    if (repo / "go.mod").exists():
        return ProjectInfo("go", "go test ./...")
    if (repo / "Cargo.toml").exists():
        return ProjectInfo("rust", "cargo test")
    if (repo / "pom.xml").exists():
        return ProjectInfo("java", "mvn -q -B test")
    if (repo / "gradlew").exists():
        return ProjectInfo("java", "./gradlew test")
    makefile = repo / "Makefile"
    if makefile.exists() and re.search(r"^test:", makefile.read_text(errors="replace"), re.M):
        return ProjectInfo("make", "make test")
    if package_json.exists():
        return ProjectInfo("node", None, _node_setup(repo))
    return ProjectInfo("unknown", None)
