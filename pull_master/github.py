"""GitHub access through the GitHub MCP server.

The harness itself uses the write tools (open a pull request, merge it, comment).
The agent only gets a short list of read-only tools, so it can look things up on
GitHub but cannot change anything there.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from pull_master import console

# Read-only GitHub MCP tools the agent may call. Anything else stays with the harness.
AGENT_TOOL_ALLOWLIST = (
    "issue_read",
    "search_issues",
    "search_code",
    "pull_request_read",
    "list_commits",
    "get_commit",
)
AGENT_TOOL_PREFIX = "github__"
MAX_TOOL_TEXT = 40_000


class GitHubError(Exception):
    """A GitHub MCP tool returned an error."""


@dataclass
class PullRequest:
    number: int
    url: str


def _strip_vendor_keys(schema: Any) -> Any:
    """Drop `x-*` extension keys from a JSON schema; the Claude API doesn't need them."""
    if isinstance(schema, dict):
        return {k: _strip_vendor_keys(v) for k, v in schema.items() if not k.startswith("x-")}
    if isinstance(schema, list):
        return [_strip_vendor_keys(v) for v in schema]
    return schema


def _result_text(result: Any) -> str:
    parts = [getattr(block, "text", "") for block in (result.content or [])]
    return "\n".join(p for p in parts if p)


class GitHubMCP:
    def __init__(self, token: str, *, url: str, command: str | None = None):
        self._token = token
        self._url = url
        self._command = command
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self.tools: dict[str, Any] = {}

    async def __aenter__(self) -> "GitHubMCP":
        await self._connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._close()

    # -- connection -------------------------------------------------------

    def _server(self) -> Any:
        if self._command:
            argv = shlex.split(self._command)
            # The server is GitHub's own code, so it gets the normal environment (Docker needs HOME etc.).
            return StdioServerParameters(
                command=argv[0],
                args=argv[1:],
                env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": self._token},
            )
        http = create_mcp_http_client(headers={"Authorization": f"Bearer {self._token}"})
        return streamable_http_client(self._url, http_client=http)

    async def _connect(self) -> None:
        self._stack = AsyncExitStack()
        self._client = await self._stack.enter_async_context(Client(self._server()))
        tools: dict[str, Any] = {}
        cursor = None
        while True:
            page = await self._client.list_tools(cursor=cursor)
            tools.update({t.name: t for t in page.tools})
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                break
        self.tools = tools
        missing = {"list_issues", "issue_read", "create_pull_request", "merge_pull_request"} - tools.keys()
        if missing:
            raise GitHubError(
                f"The GitHub MCP server is missing tools this harness needs: {sorted(missing)}. "
                "Make sure the issues and pull_requests toolsets are enabled."
            )

    async def _close(self) -> None:
        if self._stack is not None:
            stack, self._stack, self._client = self._stack, None, None
            try:
                await stack.aclose()
            except Exception as err:  # closing a dead HTTP session can fail; nothing to do about it
                console.debug(f"MCP close: {err}")

    async def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Call a tool. Returns (text, is_error). Reconnects once if the connection dropped."""
        for attempt in (1, 2):
            try:
                assert self._client is not None
                result = await self._client.call_tool(name, args)
                return _result_text(result)[:MAX_TOOL_TEXT], bool(getattr(result, "is_error", False))
            except Exception as err:
                if attempt == 2:
                    raise GitHubError(f"{name} failed: {err}") from err
                console.debug(f"MCP call {name} failed ({err!r}); reconnecting")
                await self._close()
                await self._connect()
        raise AssertionError("unreachable")

    async def call_json(self, name: str, args: dict[str, Any]) -> Any:
        text, is_error = await self.call(name, args)
        if is_error:
            raise GitHubError(f"{name}: {text.strip()[:500]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    # -- what the harness uses --------------------------------------------

    async def whoami(self) -> str:
        data = await self.call_json("get_me", {})
        return str(data.get("login", ""))

    async def list_open_issues(self, owner: str, repo: str, labels: list[str], max_items: int) -> list[dict]:
        """Open issues (oldest first when the server supports ordering), up to max_items."""
        schema_props = self.tools["list_issues"].input_schema.get("properties", {})
        args: dict[str, Any] = {"owner": owner, "repo": repo, "state": "OPEN", "perPage": min(max_items, 100)}
        if labels:
            args["labels"] = labels
        if "CREATED_AT" in schema_props.get("orderBy", {}).get("enum", []):
            args["orderBy"], args["direction"] = "CREATED_AT", "ASC"
        if "fields" in schema_props:
            args["fields"] = ["number", "title", "labels", "user"]
        issues: list[dict] = []
        while len(issues) < max_items:
            data = await self.call_json("list_issues", args)
            issues.extend(data.get("issues", []))
            page = data.get("pageInfo") or {}
            if not page.get("hasNextPage") or not page.get("endCursor"):
                break
            args["after"] = page["endCursor"]
        return issues[:max_items]

    async def get_issue(self, owner: str, repo: str, number: int) -> dict:
        data = await self.call_json(
            "issue_read", {"method": "get", "owner": owner, "repo": repo, "issue_number": number}
        )
        issue = data.get("issue", data)
        return {
            "number": issue.get("number", number),
            "title": issue.get("title", ""),
            "body": issue.get("body") or "",
            "state": (issue.get("state") or "").lower(),
            "author": (issue.get("user") or {}).get("login", "unknown"),
            "author_association": (issue.get("author_association") or "NONE").upper(),
            "url": issue.get("html_url", f"https://github.com/{owner}/{repo}/issues/{number}"),
            "is_pull_request": bool(issue.get("pull_request")),
        }

    async def get_comments(self, owner: str, repo: str, number: int) -> list[dict]:
        data = await self.call_json(
            "issue_read",
            {"method": "get_comments", "owner": owner, "repo": repo, "issue_number": number, "perPage": 50},
        )
        comments = data.get("comments", data if isinstance(data, list) else [])
        return [
            {
                "author": (c.get("user") or {}).get("login", "unknown"),
                "author_association": (c.get("author_association") or "NONE").upper(),
                "body": c.get("body") or "",
            }
            for c in comments
        ]

    async def create_pull_request(
        self, owner: str, repo: str, *, title: str, body: str, head: str, base: str
    ) -> PullRequest:
        text, is_error = await self.call(
            "create_pull_request",
            {"owner": owner, "repo": repo, "title": title, "body": body, "head": head, "base": base},
        )
        if is_error:
            if "already exists" in text.lower():
                return await self._find_open_pr(owner, repo, head)
            raise GitHubError(f"create_pull_request: {text.strip()[:500]}")
        return _parse_pr(text)

    async def _find_open_pr(self, owner: str, repo: str, head: str) -> PullRequest:
        data = await self.call_json(
            "list_pull_requests", {"owner": owner, "repo": repo, "state": "open", "head": f"{owner}:{head}"}
        )
        prs = data if isinstance(data, list) else data.get("pull_requests", data.get("items", []))
        for pr in prs:
            if (pr.get("head") or {}).get("ref") in (head, None):
                return PullRequest(number=int(pr["number"]), url=pr.get("html_url", ""))
        raise GitHubError(f"A pull request for {head} exists but could not be found")

    async def merge_pull_request(
        self, owner: str, repo: str, number: int, *, method: str, title: str, message: str
    ) -> str:
        data = await self.call_json(
            "merge_pull_request",
            {
                "owner": owner,
                "repo": repo,
                "pullNumber": number,
                "merge_method": method,
                "commit_title": title,
                "commit_message": message,
            },
        )
        if isinstance(data, dict) and data.get("merged") is False:
            raise GitHubError(f"merge_pull_request: {data.get('message', 'not merged')}")
        return str(data.get("sha", "")) if isinstance(data, dict) else ""

    async def comment(self, owner: str, repo: str, number: int, body: str) -> None:
        await self.call_json(
            "add_issue_comment", {"owner": owner, "repo": repo, "issue_number": number, "body": body}
        )

    # -- what the agent gets ----------------------------------------------

    def agent_tool_specs(self) -> list[dict]:
        specs = []
        for name in AGENT_TOOL_ALLOWLIST:
            tool = self.tools.get(name)
            annotations = getattr(tool, "annotations", None)
            if tool is None or not getattr(annotations, "read_only_hint", False):
                continue
            schema = _strip_vendor_keys(tool.input_schema or {"type": "object", "properties": {}})
            specs.append(
                {
                    "name": AGENT_TOOL_PREFIX + name,
                    "description": f"(GitHub, read-only) {tool.description or name}",
                    "input_schema": schema,
                    "eager_input_streaming": True,
                }
            )
        return specs

    async def call_agent_tool(self, prefixed_name: str, args: dict[str, Any]) -> tuple[str, bool]:
        name = prefixed_name.removeprefix(AGENT_TOOL_PREFIX)
        if name not in AGENT_TOOL_ALLOWLIST:
            return f"Tool {prefixed_name} is not available.", True
        return await self.call(name, args)


def _parse_pr(text: str) -> PullRequest:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
    url = data.get("html_url") or data.get("url") or ""
    number = data.get("number")
    if number is None:
        match = re.search(r"/pull/(\d+)", url or text)
        if not match:
            raise GitHubError(f"Could not read the pull request number from: {text[:300]}")
        number = match.group(1)
    if not url or "api.github.com" in url:
        found = re.search(r"https://github\.com/[^\s\"]+/pull/\d+", text)
        url = found.group(0) if found else url
    return PullRequest(number=int(number), url=url)
