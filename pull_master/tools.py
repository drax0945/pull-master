"""Local tools the agent uses inside the workspace: a bash session and a file editor.

Both implement Anthropic-defined tools (bash_20250124 and text_editor_20250728), which the
model already knows how to use. The editor can only touch files inside the repository.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

from pull_master import console
from pull_master.workspace import OUTPUT_LIMIT, kill_process_group, truncate_middle

BASH_TOOL = {"type": "bash_20250124", "name": "bash"}
EDITOR_TOOL = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}


class ToolError(Exception):
    """Reported back to the model as an error tool result."""


# A speed bump, not a sandbox: the real protections are the scrubbed environment and the
# fact that only the harness can push. These catch the obvious mistakes and injected asks.
_COMMAND_START = r"(?:^|[;&|(\n]|\$\()\s*"  # start of a command, not just anywhere in the text
_BLOCKED_COMMANDS = [
    (re.compile(r"\bgit\s+(push|remote\s+(add|set-url))\b"), "the harness pushes; don't push or edit remotes"),
    (re.compile(_COMMAND_START + r"gh(\s|$)"), "the GitHub CLI is not available to the agent"),
    (re.compile(_COMMAND_START + r"sudo\s"), "no sudo"),
    (re.compile(r"\b(curl|wget)\b[^|;&]*\|\s*(ba|z)?sh\b"), "piping downloads into a shell is not allowed"),
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+(/|~|\$HOME)(\s|/?$)"), "refusing to delete outside the repo"),
]


def check_command_policy(command: str) -> str | None:
    for pattern, reason in _BLOCKED_COMMANDS:
        if pattern.search(command):
            return reason
    return None


class ShellSession:
    """A persistent bash process, so `cd` and exported variables carry over between calls."""

    def __init__(self, cwd: Path, env: dict[str, str], scratch: Path, timeout: int, approve: bool = False):
        self.cwd = cwd
        self.env = env
        self.scratch = scratch
        self.timeout = timeout
        self.approve = approve
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self.scratch.mkdir(parents=True, exist_ok=True)
        self._proc = await asyncio.create_subprocess_exec(
            "/bin/bash", "--noprofile", "--norc",
            cwd=self.cwd,
            env=self.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

    async def close(self) -> None:
        if self._proc and self._proc.returncode is None:
            kill_process_group(self._proc.pid)
            await self._proc.wait()
        self._proc = None

    async def restart(self) -> None:
        await self.close()
        await self.start()

    async def run(self, command: str) -> str:
        reason = check_command_policy(command)
        if reason:
            raise ToolError(f"Blocked by harness policy: {reason}.")
        if self.approve and not await _ask_approval(command):
            raise ToolError("The operator declined to run this command.")
        if self._proc is None or self._proc.returncode is not None:
            await self.start()
        assert self._proc and self._proc.stdin and self._proc.stdout

        # Sourcing a file (instead of writing the command to stdin) means a syntax error
        # fails cleanly instead of leaving bash waiting for more input.
        script = self.scratch / "command.sh"
        script.write_text(command + "\n")
        marker = f"__PULL_MASTER_DONE_{uuid.uuid4().hex}__"
        self._proc.stdin.write(f'{{ source "{script}"; }} < /dev/null 2>&1\necho "{marker}$?"\n'.encode())
        await self._proc.stdin.drain()

        buffer = bytearray()
        pattern = re.compile(re.escape(marker).encode() + rb"(\d+)")

        async def read_until_marker() -> int:
            while True:
                chunk = await self._proc.stdout.read(65536)  # type: ignore[union-attr]
                if not chunk:
                    raise EOFError
                buffer.extend(chunk)
                match = pattern.search(buffer)
                if match:
                    exit_code = int(match.group(1))
                    del buffer[match.start():]
                    return exit_code

        try:
            exit_code = await asyncio.wait_for(read_until_marker(), self.timeout)
        except (TimeoutError, asyncio.TimeoutError):
            await self.restart()
            output = truncate_middle(buffer.decode("utf-8", "replace"))
            raise ToolError(
                f"{output}\n\nCommand timed out after {self.timeout}s and was killed. "
                "The shell was restarted (working directory and variables were reset)."
            )
        except EOFError:
            await self.restart()
            output = truncate_middle(buffer.decode("utf-8", "replace"))
            raise ToolError(f"{output}\n\nThe shell exited. It was restarted in the repository root.")

        output = truncate_middle(buffer.decode("utf-8", "replace").rstrip("\n"), OUTPUT_LIMIT)
        if exit_code != 0:
            return f"{output}\n[exit code {exit_code}]" if output else f"[exit code {exit_code}]"
        return output or "(no output)"


async def _ask_approval(command: str) -> bool:
    console.tool_call("approve?", command)
    answer = await asyncio.to_thread(input, "    Run this command? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build"}
VIEW_LIMIT = 60_000


class FileEditor:
    """str_replace_based_edit_tool, confined to the repository root."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, raw: str) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError("`path` is required.")
        path = Path(raw)
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if not path.is_relative_to(self.root):
            raise ToolError(f"{raw} is outside the repository ({self.root}). Only files in the repository can be accessed.")
        if ".git" in path.relative_to(self.root).parts:
            raise ToolError("The .git directory is off limits.")
        return path

    def handle(self, args: dict) -> str:
        command = args.get("command")
        path = self.resolve(args.get("path", ""))
        if command == "view":
            return self._view(path, args.get("view_range"))
        if command == "create":
            text = args.get("file_text")
            if not isinstance(text, str):
                raise ToolError("`file_text` is required for create.")
            existed = path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            return f"{'Overwrote' if existed else 'Created'} {self._rel(path)}"
        if command == "str_replace":
            return self._str_replace(path, args.get("old_str"), args.get("new_str", ""))
        if command == "insert":
            return self._insert(path, args.get("insert_line"), args.get("insert_text"))
        raise ToolError(f"Unknown command {command!r}. Use view, create, str_replace or insert.")

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.root)) or "."

    def _read(self, path: Path) -> str:
        if not path.exists():
            raise ToolError(f"{self._rel(path)} does not exist.")
        if path.is_dir():
            raise ToolError(f"{self._rel(path)} is a directory.")
        try:
            return path.read_text()
        except UnicodeDecodeError:
            raise ToolError(f"{self._rel(path)} is not a UTF-8 text file.")

    def _view(self, path: Path, view_range: object) -> str:
        if path.is_dir():
            lines = []
            for child in sorted(path.rglob("*")):
                rel = child.relative_to(path)
                if len(rel.parts) > 2 or any(p in _SKIP_DIRS or p.startswith(".") for p in rel.parts):
                    continue
                lines.append(f"{rel}{'/' if child.is_dir() else ''}")
                if len(lines) >= 500:
                    lines.append("... (listing cut at 500 entries)")
                    break
            return f"Files in {self._rel(path)} (2 levels, hidden files skipped):\n" + "\n".join(lines)
        content = self._read(path).split("\n")
        start, end = 1, len(content)
        if view_range is not None:
            if not (isinstance(view_range, list) and len(view_range) == 2 and all(isinstance(n, int) for n in view_range)):
                raise ToolError("`view_range` must be [start_line, end_line]; use -1 as end_line for end of file.")
            start, end = view_range
            end = len(content) if end == -1 else end
            if start < 1 or start > len(content) or end < start:
                raise ToolError(f"Invalid view_range {view_range}; the file has {len(content)} lines.")
        numbered = "\n".join(f"{i:6}\t{line}" for i, line in enumerate(content[start - 1 : end], start))
        if len(numbered) > VIEW_LIMIT:
            numbered = numbered[:VIEW_LIMIT] + "\n... (output cut; use view_range to see the rest)"
        return numbered

    def _str_replace(self, path: Path, old: object, new: object) -> str:
        if not isinstance(old, str) or old == "":
            raise ToolError("`old_str` is required and must not be empty.")
        if not isinstance(new, str):
            raise ToolError("`new_str` must be a string.")
        text = self._read(path)
        count = text.count(old)
        if count == 0:
            raise ToolError(f"No match for old_str in {self._rel(path)}. It must match exactly, including whitespace.")
        if count > 1:
            raise ToolError(f"old_str matches {count} places in {self._rel(path)}. Include more context so it is unique.")
        updated = text.replace(old, new, 1)
        path.write_text(updated)
        line = text[: text.index(old)].count("\n") + 1
        lines = updated.split("\n")
        lo, hi = max(1, line - 3), min(len(lines), line + new.count("\n") + 3)
        snippet = "\n".join(f"{i:6}\t{lines[i - 1]}" for i in range(lo, hi + 1))
        return f"Edited {self._rel(path)}. Lines {lo}-{hi} now read:\n{snippet}"

    def _insert(self, path: Path, line: object, text: object) -> str:
        if not isinstance(line, int) or not isinstance(text, str):
            raise ToolError("insert needs an integer `insert_line` and a string `insert_text`.")
        lines = self._read(path).split("\n")
        if line < 0 or line > len(lines):
            raise ToolError(f"insert_line must be between 0 and {len(lines)}.")
        lines[line:line] = text.split("\n")
        path.write_text("\n".join(lines))
        return f"Inserted {text.count(chr(10)) + 1} line(s) after line {line} of {self._rel(path)}."
