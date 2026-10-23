"""Terminal output. Plain prints with optional color; no dependencies."""

from __future__ import annotations

import os
import sys

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_DEBUG = bool(os.environ.get("PULL_MASTER_DEBUG"))
_at_line_start = True  # tracks streamed model output so tool lines start on a fresh line


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _line(text: str, prefix: str = "    ", code: str | None = None, stream=sys.stdout) -> None:
    global _at_line_start
    if not _at_line_start:
        print(file=stream)
    body = "\n".join(prefix + line if i == 0 else " " * len(prefix) + line for i, line in enumerate(text.split("\n")))
    print(_c(code, body) if code else body, file=stream, flush=True)
    _at_line_start = True


def step(text: str) -> None:
    _line(text, "==> ", "1;36")


def info(text: str) -> None:
    _line(text)


def ok(text: str) -> None:
    _line(text, "    ✓ ", "32")


def warn(text: str) -> None:
    _line(text, "    ! ", "33")


def error(text: str) -> None:
    _line(text, "    ✗ ", "31", sys.stderr)


def debug(text: str) -> None:
    if _DEBUG:
        _line(text, "    [debug] ", "2", sys.stderr)


def _stream(text: str, code: str | None = None) -> None:
    """Write streamed text, indenting each new line to match the rest of the output."""
    global _at_line_start
    pieces = []
    for part in text.splitlines(keepends=True):
        if _at_line_start and part.strip():
            pieces.append("    ")
        pieces.append(part)
        _at_line_start = part.endswith("\n")
    if pieces:
        out = "".join(pieces)
        sys.stdout.write(_c(code, out) if code else out)
        sys.stdout.flush()


def agent_text(text: str) -> None:
    """Streamed model text, printed as it arrives."""
    _stream(text)


def agent_note(text: str) -> None:
    """Streamed progress notes (thinking display "updates"), dimmed."""
    _stream(text, "2")


def tool_call(name: str, summary: str) -> None:
    line = summary.replace("\n", " ⏎ ")
    if len(line) > 160:
        line = line[:157] + "..."
    _line(f"{_c('35', f'[{name}]')} {line}")
