"""Tests for the agent loop.

The loop is exercised against a fake client so the shape of what goes back to
the API can be asserted without spending tokens.  The Go implementation this
replaces had no tests, and two of the cases below cover defects it carried: a
hardcoded model that ignored the configured one, and a tool that crashed the
process on malformed input instead of reporting the failure to Claude.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from co_agent.agent import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, Agent
from co_agent.tools import READ_FILE, ToolDefinition, schema_for
from co_agent.tools.read_file import ReadFileInput


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def thinking(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=value)


def tool_use(block_id: str, name: str, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload)


class FakeMessages:
    def __init__(self, responses: list[list[SimpleNamespace]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("the loop asked for more turns than were scripted")
        return SimpleNamespace(content=self.responses.pop(0))


class FakeClient:
    def __init__(self, responses: list[list[SimpleNamespace]]) -> None:
        self.messages = FakeMessages(responses)


def scripted(*lines: str):
    """A get_user_message that plays back lines, then reports EOF."""
    pending = list(lines)

    def read() -> tuple[str, bool]:
        if not pending:
            return "", False
        return pending.pop(0), True

    return read


def echo_tool(name: str = "echo", result: str = "ok") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="echoes",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        run=lambda payload: result,
    )


def test_tool_call_round_trip():
    client = FakeClient([[tool_use("tu1", "echo", {"a": 1})], [text("done")]])
    Agent(client, scripted("go"), [echo_tool()]).run()

    first, second = client.messages.calls
    assert first["messages"] == [{"role": "user", "content": "go"}]

    # The second request carries the assistant turn verbatim, then the results.
    assert len(second["messages"]) == 3
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}],
    }


def test_parallel_tool_calls_return_in_one_user_message():
    """Splitting results teaches Claude to stop calling tools in parallel."""
    client = FakeClient(
        [
            [tool_use("a", "echo", {}), tool_use("b", "other", {})],
            [text("done")],
        ]
    )
    Agent(client, scripted("go"), [echo_tool(), echo_tool("other", "second")]).run()

    results = client.messages.calls[1]["messages"][2]
    assert results["role"] == "user"
    assert [block["tool_use_id"] for block in results["content"]] == ["a", "b"]
    assert [block["content"] for block in results["content"]] == ["ok", "second"]


def test_unknown_tool_is_reported_as_an_error_not_raised():
    client = FakeClient([[tool_use("tu1", "nope", {})], [text("done")]])
    Agent(client, scripted("go"), [echo_tool()]).run()

    result = client.messages.calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert "not found" in result["content"]


def test_a_failing_tool_reports_back_instead_of_crashing():
    def explode(payload: dict) -> str:
        raise FileNotFoundError("no such file: nope.txt")

    tool = ToolDefinition(
        name="boom",
        description="fails",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        run=explode,
    )
    client = FakeClient([[tool_use("tu1", "boom", {})], [text("recovered")]])
    Agent(client, scripted("go"), [tool]).run()

    result = client.messages.calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert "FileNotFoundError" in result["content"]
    assert "nope.txt" in result["content"]


def test_thinking_blocks_are_echoed_back_unchanged():
    blocks = [thinking("weighing it up"), text("here goes"), tool_use("tu1", "echo", {})]
    client = FakeClient([blocks, [text("done")]])
    Agent(client, scripted("go"), [echo_tool()]).run()

    assistant = client.messages.calls[1]["messages"][1]
    assert assistant["content"] is blocks


def test_a_turn_without_tool_calls_hands_back_to_the_user():
    client = FakeClient([[text("first")], [text("second")]])
    Agent(client, scripted("one", "two"), [echo_tool()]).run()
    assert len(client.messages.calls) == 2
    assert client.messages.calls[1]["messages"][-1] == {"role": "user", "content": "two"}


def test_the_configured_model_is_the_one_requested():
    """The Go version hardcoded a model and ignored the one it was given."""
    client = FakeClient([[text("hi")]])
    Agent(client, scripted("go"), [], model="claude-sonnet-5", max_tokens=2048).run()
    assert client.messages.calls[0]["model"] == "claude-sonnet-5"
    assert client.messages.calls[0]["max_tokens"] == 2048


def test_defaults_are_opus_5_and_adaptive_thinking():
    client = FakeClient([[text("hi")]])
    Agent(client, scripted("go"), []).run()
    call = client.messages.calls[0]
    assert call["model"] == DEFAULT_MODEL == "claude-opus-5"
    assert call["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call["thinking"]["type"] == "adaptive"


def test_tools_are_sent_with_a_required_list():
    """A parameter listed without being required lets Claude omit it."""
    client = FakeClient([[text("hi")]])
    Agent(client, scripted("go"), [READ_FILE]).run()

    (tool,) = client.messages.calls[0]["tools"]
    assert tool["name"] == "read_file"
    assert tool["input_schema"]["required"] == ["path"]
    assert tool["input_schema"]["additionalProperties"] is False


def test_read_file_returns_contents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    assert READ_FILE.run({"path": "note.txt"}) == "hello\n"


def test_read_file_rejects_a_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    with pytest.raises(IsADirectoryError):
        READ_FILE.run({"path": "sub"})


def test_read_file_rejects_malformed_input():
    """The Go version panicked here, taking the process with it."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        READ_FILE.run({"wrong_key": "x"})


def test_schema_for_drops_the_pydantic_title():
    schema = schema_for(ReadFileInput)
    assert "title" not in schema
    assert schema["type"] == "object"
