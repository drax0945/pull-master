"""The agent loop: Claude works on one issue with local tools until it submits a result.

The loop is append-only (it never edits earlier messages), so the prompt cache keeps
hitting and replayed thinking blocks stay valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anthropic
from jsonschema import Draft202012Validator

from pull_master import console, prompts
from pull_master.tools import BASH_TOOL, EDITOR_TOOL, FileEditor, ShellSession, ToolError
from pull_master.workspace import CommandResult, Workspace

RUN_TESTS_TOOL = {
    "name": "run_tests",
    "description": (
        "Run the project's test suite with the same command and environment the harness uses before "
        "merging. Call it after changing code and before submit_result. Returns pass/fail and the output "
        "(long output is trimmed in the middle)."
    ),
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    "eager_input_streaming": True,
}

SUBMIT_TOOL = {
    "name": "submit_result",
    "description": (
        "Finish your work on the issue. Call it when you are done. With outcome 'fixed' the harness runs "
        "the test suite and, if it passes, commits your changes, opens a pull request and merges it; if the "
        "tests fail you get the output back and should keep working, then call this again. Use "
        "'needs_human' when the issue should not be changed automatically."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": ["fixed", "needs_human"]},
            "title": {
                "type": "string",
                "description": "Pull request title: imperative, under 72 characters, no issue number. "
                "Example: 'Handle empty config files in load_config'.",
            },
            "summary": {
                "type": "string",
                "description": "Markdown. For 'fixed': what was wrong, what you changed, and how you verified it. "
                "For 'needs_human': what a person needs to decide or do, and why.",
            },
            "test_command": {
                "type": "string",
                "description": "Only when you were told no test command was detected: the shell command that "
                "runs this project's tests. Otherwise leave it out.",
            },
        },
        "required": ["outcome", "title", "summary"],
        "additionalProperties": False,
    },
    "eager_input_streaming": True,
}

# Models that accept the request options used below.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_PROGRESS_UPDATE_MODELS = {"claude-opus-5-5", "claude-fable-5-1", "claude-fable-5", "claude-sonnet-5-5"}


class Model(Protocol):
    async def respond(self, system: str, tools: list[dict], messages: list[dict]) -> Any: ...


class ClaudeModel:
    """Streams one Messages API turn and prints text and progress notes as they arrive."""

    def __init__(
        self,
        model: str,
        effort: str,
        fallbacks: bool = True,
        max_tokens: int = 64_000,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.max_tokens = max_tokens
        self.client = client or anthropic.AsyncAnthropic(max_retries=4)

    def _params(self, system: str, tools: list[dict], messages: list[dict]) -> dict:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # The system prompt is identical for every issue, so it caches across the whole run;
            # top-level cache_control caches the growing conversation as well.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "cache_control": {"type": "ephemeral"},
            "tools": tools,
            "messages": messages,
            "output_config": {"effort": self.effort},
            "thinking": {"type": "adaptive"},
        }
        betas = []
        if self.model in _PROGRESS_UPDATE_MODELS:
            # Short progress notes between tool calls, without exposing the reasoning itself.
            params["thinking"] = {"type": "adaptive", "display": "updates"}
            betas.append("thinking-display-updates-2026-08-18")
        if self.fallbacks and self.model in _FALLBACK_MODELS:
            # If a safety classifier declines a turn, the API retries it on a fallback model.
            params["fallbacks"] = "default"
            betas.append("server-side-fallback-2026-07-01")
        if betas:
            params["betas"] = betas
        return params

    async def respond(self, system: str, tools: list[dict], messages: list[dict]) -> Any:
        params = self._params(system, tools, messages)
        for attempt in range(3):
            try:
                async with self.client.beta.messages.stream(**params) as stream:
                    async for event in stream:
                        if event.type == "thinking" and event.thinking:
                            console.agent_note(event.thinking)
                        elif event.type == "text":
                            console.agent_text(event.text)
                    return await stream.get_final_message()
            except ValueError as err:
                # With eager input streaming the SDK raises ValueError on tool-input JSON it
                # cannot parse at all. There is no tool_use id to answer, so re-issue the turn.
                if attempt == 2:
                    raise
                console.warn(f"Model sent unparseable tool input ({err}); retrying the turn")
        raise AssertionError("unreachable")


@dataclass
class FixResult:
    outcome: str  # fixed | needs_human | tests_failed | gave_up | refused
    title: str = ""
    summary: str = ""
    test_command: str | None = None
    test_result: CommandResult | None = None
    attempts: int = 0
    turns: int = 0
    usage: dict = field(default_factory=dict)


class IssueFixer:
    def __init__(
        self,
        *,
        model: Model,
        workspace: Workspace,
        shell: ShellSession,
        editor: FileEditor,
        github: Any | None,
        base: str,
        test_cmd: str | None,
        max_attempts: int,
        max_turns: int,
        command_timeout: int,
        transcript: Path,
    ):
        self.model = model
        self.workspace = workspace
        self.shell = shell
        self.editor = editor
        self.github = github
        self.base = base
        self.test_cmd = test_cmd
        self.max_attempts = max_attempts
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self.transcript = transcript
        self.messages: list[dict] = []
        self.attempts = 0
        self.usage = {"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 0}
        github_tools = github.agent_tool_specs() if github is not None else []
        self.tools = [BASH_TOOL, EDITOR_TOOL, RUN_TESTS_TOOL, SUBMIT_TOOL, *github_tools]
        self._validators = {
            t["name"]: Draft202012Validator(t["input_schema"]) for t in self.tools if "input_schema" in t
        }

    # -- transcript & bookkeeping -----------------------------------------

    def _append(self, message: dict) -> None:
        self.messages.append(message)
        content = message["content"]
        if not isinstance(content, str):
            content = [b.model_dump(mode="json", exclude_none=True) if hasattr(b, "model_dump") else b for b in content]
        with self.transcript.open("a") as f:
            f.write(json.dumps({"role": message["role"], "content": content}) + "\n")

    def _count_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        for key in self.usage:
            self.usage[key] += getattr(usage, key, 0) or 0

    def _result(self, outcome: str, turns: int, **kwargs: Any) -> FixResult:
        return FixResult(outcome=outcome, attempts=self.attempts, turns=turns, usage=dict(self.usage), **kwargs)

    # -- main loop ----------------------------------------------------------

    async def run(self, kickoff: str) -> FixResult:
        self._append({"role": "user", "content": kickoff})
        nudges = 0
        for turn in range(1, self.max_turns + 1):
            response = await self.model.respond(prompts.SYSTEM_PROMPT, self.tools, self.messages)
            self._count_usage(response)
            self._append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                category = getattr(details, "category", None) or "unspecified"
                return self._result("refused", turn, summary=f"The model declined this issue (category: {category}).")

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "max_tokens":
                if tool_uses:
                    self._append({"role": "user", "content": [
                        _tool_result(t.id, "Your response hit the output limit and this tool call was cut off, "
                                     "so it was not run. Retry in smaller steps.", True)
                        for t in tool_uses
                    ]})
                else:
                    self._append({"role": "user", "content": "Your reply was cut off at the output limit. Continue in smaller steps."})
                continue

            if not tool_uses:
                if response.stop_reason == "pause_turn":
                    continue
                nudges += 1
                if nudges > 2:
                    text = "".join(b.text for b in response.content if b.type == "text")
                    return self._result("gave_up", turn, summary=text.strip() or "Stopped without submitting a result.")
                self._append({"role": "user", "content": "You ended your turn without calling submit_result. "
                              "Keep working, or call submit_result to finish."})
                continue

            # Run submit_result after the other calls in the same turn, so it sees their effects.
            ordered = sorted(tool_uses, key=lambda t: t.name == "submit_result")
            results: dict[str, dict] = {}
            finished: FixResult | None = None
            for tool_use in ordered:
                if tool_use.name == "submit_result":
                    content, is_error, finished = await self._submit(tool_use.input, turn)
                else:
                    content, is_error = await self._execute(tool_use.name, tool_use.input)
                results[tool_use.id] = _tool_result(tool_use.id, content, is_error)
            self._append({"role": "user", "content": [results[t.id] for t in tool_uses]})
            if finished is not None:
                return finished

        return self._result("gave_up", self.max_turns, summary=f"Reached the limit of {self.max_turns} turns.")

    # -- tools --------------------------------------------------------------

    def _invalid(self, name: str, args: Any) -> str | None:
        if not isinstance(args, dict):
            return json.dumps({"INVALID_JSON": str(args)})
        validator = self._validators.get(name)
        if validator is None:
            return None
        errors = sorted(validator.iter_errors(args), key=lambda e: list(e.path))
        if errors:
            return json.dumps({"INVALID_JSON": json.dumps(args), "error": errors[0].message})
        return None

    async def _execute(self, name: str, args: Any) -> tuple[str, bool]:
        invalid = self._invalid(name, args)
        if invalid:
            return invalid, True
        try:
            if name == "bash":
                if args.get("restart"):
                    await self.shell.restart()
                    return "The shell was restarted.", False
                command = args.get("command")
                if not isinstance(command, str) or not command.strip():
                    return "`command` is required.", True
                console.tool_call("bash", command)
                return await self.shell.run(command), False
            if name == "str_replace_based_edit_tool":
                console.tool_call("edit", f"{args.get('command')} {args.get('path')}")
                return self.editor.handle(args), False
            if name == "run_tests":
                if not self.test_cmd:
                    return "No test command is configured. Find out how the project runs its tests, run them " \
                           "with bash, and pass the command as test_command in submit_result.", True
                result = await self._run_tests(self.test_cmd)
                return f"`{result.command}` {result.describe()}.\n```\n{result.tail(12000)}\n```", not result.ok
            if self.github is not None and name.startswith("github__"):
                console.tool_call(name, json.dumps(args))
                return await self.github.call_agent_tool(name, args)
            return f"Unknown tool: {name}", True
        except ToolError as err:
            return str(err), True
        except Exception as err:  # a tool bug should not end the whole run
            console.debug(f"tool {name} crashed: {err!r}")
            return f"The tool failed: {err}", True

    async def _run_tests(self, command: str) -> CommandResult:
        console.tool_call("tests", command)
        result = await self.workspace.run(command, self.command_timeout)
        (console.ok if result.ok else console.warn)(f"tests {result.describe()}")
        return result

    async def _submit(self, args: Any, turn: int) -> tuple[str, bool, FixResult | None]:
        invalid = self._invalid("submit_result", args)
        if invalid:
            return invalid, True, None
        outcome, title, summary = args["outcome"], args["title"].strip(), args["summary"].strip()
        console.tool_call("submit", f"{outcome}: {title}")
        if outcome == "needs_human":
            return "Recorded. A person will follow up.", False, self._result("needs_human", turn, title=title, summary=summary)

        self.attempts += 1
        attempts_left = self.max_attempts - self.attempts
        if not self.workspace.has_changes(self.base):
            if attempts_left <= 0:
                return "No changes found.", True, self._result("gave_up", turn, title=title, summary=summary)
            return ("Rejected: there are no changes in the working tree. Make the fix first, or submit "
                    f"needs_human. Attempts left: {attempts_left}."), True, None

        command = self.test_cmd or (args.get("test_command") or "").strip() or None
        if command is None:
            if attempts_left <= 0:
                return "No test command.", True, self._result("needs_human", turn, title=title, summary=summary
                                                               + "\n\n(No test command could be found, so this was not merged.)")
            return ("Rejected: no test command is known, so the change can't be verified. Pass the command "
                    f"that runs this project's tests as test_command. Attempts left: {attempts_left}."), True, None
        self.test_cmd = command

        result = await self._run_tests(command)
        if result.ok:
            return "Accepted: the test suite passed.", False, self._result(
                "fixed", turn, title=title, summary=summary, test_command=command, test_result=result
            )
        if attempts_left <= 0:
            return "Tests failed; out of attempts.", True, self._result(
                "tests_failed", turn, title=title, summary=summary, test_command=command, test_result=result
            )
        return prompts.test_feedback(result, attempts_left), True, None


def _tool_result(tool_use_id: str, content: str, is_error: bool) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content or "(empty)"}
    if is_error:
        block["is_error"] = True
    return block
