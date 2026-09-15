"""A hand-written agentic loop over the Anthropic Messages API.

The loop is deliberately explicit: send the conversation, print what came back,
run any tools Claude asked for, send the results, repeat until Claude stops
calling tools and it is the user's turn again.

The SDK also ships a tool runner (``client.beta.messages.tool_runner``) that
owns this loop for you.  It is the better default for new code; this module
exists because owning the loop is the point of the exercise, and because a
manual loop avoids a beta dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import anthropic
from anthropic.types import Message, MessageParam

from .tools import ToolDefinition

#: Claude Opus 5.  Thinking is on by default on this model.
DEFAULT_MODEL = "claude-opus-5"

#: Non-streaming responses stay comfortably inside the SDK's HTTP timeout at
#: this size.  Raise it only alongside a switch to ``client.messages.stream``.
DEFAULT_MAX_TOKENS = 16_000

_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_GREY = "\033[90m"
_RED = "\033[91m"
_RESET = "\033[0m"


class Agent:
    """A chat loop with tools."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        get_user_message: Callable[[], tuple[str, bool]],
        tools: Sequence[ToolDefinition],
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        show_thinking: bool = True,
    ) -> None:
        self.client = client
        self.get_user_message = get_user_message
        self.tools = list(tools)
        self.model = model
        self.max_tokens = max_tokens
        self.show_thinking = show_thinking
        self._by_name = {tool.name: tool for tool in self.tools}

    def run(self) -> None:
        """Drive the conversation until the user stops it."""
        conversation: list[MessageParam] = []
        print("Chat with Claude (use 'ctrl-c' to quit)")

        read_user_input = True
        while True:
            if read_user_input:
                print(f"{_BLUE}You{_RESET}: ", end="", flush=True)
                user_input, ok = self.get_user_message()
                if not ok:
                    return
                conversation.append({"role": "user", "content": user_input})

            try:
                message = self._run_inference(conversation)
            except anthropic.CredentialsError as exc:
                self._fail(
                    f"no usable credentials: {exc}\n"
                    "Set ANTHROPIC_API_KEY (a .env file works) or run 'ant auth login'."
                )
                return
            except anthropic.AuthenticationError as exc:
                self._fail(f"the credentials were rejected: {exc.message}")
                return
            except anthropic.NotFoundError as exc:
                self._fail(f"model {self.model!r} is not available: {exc.message}")
                return
            except anthropic.RateLimitError as exc:
                # Retryable: keep the conversation and let the user try again.
                self._warn(f"rate limited: {exc.message}")
                read_user_input = True
                continue
            except anthropic.APIStatusError as exc:
                self._fail(f"API error {exc.status_code}: {exc.message}")
                return
            except anthropic.APIConnectionError as exc:
                self._warn(f"could not reach the API: {exc}")
                read_user_input = True
                continue

            # Append the assistant turn verbatim.  Thinking blocks must go back
            # unchanged, so this appends response content rather than just text.
            conversation.append({"role": "assistant", "content": message.content})

            tool_results: list[dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    print(f"{_YELLOW}Claude{_RESET}: {block.text}")
                elif block.type == "thinking":
                    if self.show_thinking and block.thinking.strip():
                        print(f"{_GREY}thinking{_RESET}: {block.thinking.strip()}")
                elif block.type == "tool_use":
                    tool_results.append(self._execute_tool(block))

            if not tool_results:
                read_user_input = True
                continue

            # Every tool_result for one assistant turn goes back in a single
            # user message; splitting them teaches Claude to stop calling tools
            # in parallel.
            read_user_input = False
            conversation.append({"role": "user", "content": tool_results})

    def _execute_tool(self, block: Any) -> dict[str, Any]:
        """Run one tool call and build its ``tool_result`` block."""
        tool = self._by_name.get(block.name)
        if tool is None:
            return _tool_result(block.id, f"tool {block.name!r} not found", is_error=True)

        print(f"{_GREEN}tool{_RESET}: {block.name}({block.input})")
        try:
            return _tool_result(block.id, tool.run(block.input))
        except Exception as exc:  # noqa: BLE001 - a failing tool is Claude's to handle
            # Hand the failure back rather than raising: Claude can correct a bad
            # path or pick another tool, which is the whole point of is_error.
            return _tool_result(block.id, f"{type(exc).__name__}: {exc}", is_error=True)

    def _run_inference(self, conversation: Sequence[MessageParam]) -> Message:
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=list(conversation),
            tools=[tool.to_param() for tool in self.tools],
            # Adaptive thinking is the current mode; "summarized" is needed to
            # see anything, since these models omit reasoning text by default.
            thinking={"type": "adaptive", "display": "summarized"},
        )

    @staticmethod
    def _warn(msg: str) -> None:
        print(f"{_RED}warning{_RESET}: {msg}")

    @staticmethod
    def _fail(msg: str) -> None:
        print(f"{_RED}error{_RESET}: {msg}")


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        result["is_error"] = True
    return result
