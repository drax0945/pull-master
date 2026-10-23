"""Runs ClaudeModel through the real Anthropic SDK against a local fake API server.

This checks the request the harness sends (betas, fallbacks, thinking, effort, caching, tool
flags) and that a streamed response, thinking signature included, goes back unchanged on the
next turn. It can't check that the real API accepts every option; only a live run does that.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import pytest

from pull_master.agent import RUN_TESTS_TOOL, SUBMIT_TOOL, ClaudeModel
from pull_master.tools import BASH_TOOL, EDITOR_TOOL


def sse(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def message_start():
    return {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 50, "output_tokens": 1}}}


TOOL_TURN = sse(
    message_start(),
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Checking calc.py."}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"cat "}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "calc.py\"}"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 30}},
    {"type": "message_stop"},
)

TEXT_TURN = sse(
    message_start(),
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Done."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 3}},
    {"type": "message_stop"},
)


@pytest.fixture
def fake_api():
    requests = []
    replies = [TOOL_TURN, TEXT_TURN]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
            payload = replies.pop(0)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", requests
    server.shutdown()


def test_request_shape_and_round_trip(fake_api):
    base_url, requests = fake_api
    client = anthropic.AsyncAnthropic(api_key="test-key", base_url=base_url, max_retries=0)
    model = ClaudeModel("claude-opus-5-5", "high", client=client)
    tools = [BASH_TOOL, EDITOR_TOOL, RUN_TESTS_TOOL, SUBMIT_TOOL]
    messages = [{"role": "user", "content": "Fix issue #1"}]

    async def two_turns():
        first = await model.respond("system prompt", tools, messages)
        assert first.stop_reason == "tool_use"
        tool_use = next(b for b in first.content if b.type == "tool_use")
        assert tool_use.input == {"command": "cat calc.py"}
        messages.append({"role": "assistant", "content": first.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_use.id, "content": "def add(a, b): ..."}]})
        return await model.respond("system prompt", tools, messages)

    second = asyncio.run(two_turns())
    assert second.stop_reason == "end_turn"

    first_req = requests[0]
    assert first_req["path"].startswith("/v1/messages")
    betas = first_req["headers"].get("anthropic-beta", "")
    assert "server-side-fallback-2026-07-01" in betas and "thinking-display-updates-2026-08-18" in betas
    body = first_req["body"]
    assert body["model"] == "claude-opus-5-5" and body["stream"] is True and body["max_tokens"] == 64000
    assert body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "adaptive", "display": "updates"}
    assert body["output_config"] == {"effort": "high"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    sent_tools = {t["name"]: t for t in body["tools"]}
    assert sent_tools["bash"] == {"type": "bash_20250124", "name": "bash"}
    assert sent_tools["run_tests"]["eager_input_streaming"] is True
    assert sent_tools["submit_result"]["eager_input_streaming"] is True

    # The assistant turn goes back exactly as received, thinking signature included.
    replayed = requests[1]["body"]["messages"][1]["content"]
    assert replayed[0] == {"type": "thinking", "thinking": "Checking calc.py.", "signature": "sig-abc"}
    assert replayed[1] == {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "cat calc.py"}}
    assert requests[1]["body"]["messages"][2]["content"][0]["tool_use_id"] == "toolu_1"


def test_older_models_skip_unsupported_options():
    model = ClaudeModel("claude-haiku-5-5", "low", client=anthropic.AsyncAnthropic(api_key="x"))
    params = model._params("s", [], [{"role": "user", "content": "hi"}])
    assert "fallbacks" not in params and "betas" not in params
    assert params["thinking"] == {"type": "adaptive"}
