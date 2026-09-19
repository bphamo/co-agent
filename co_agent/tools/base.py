"""Tool definitions for the agent loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One tool the agent can call.

    ``run`` receives the already-parsed input object from the API and returns
    the text to hand back.  It raises on failure; the agent turns the exception
    into a ``tool_result`` with ``is_error`` set, which is how a tool reports a
    problem to Claude without ending the turn.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[dict[str, Any]], str]

    def to_param(self) -> dict[str, Any]:
        """Render for the ``tools`` argument of ``messages.create``."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Build a tool input schema from a pydantic model.

    ``required`` is part of what comes back, which matters: a schema that lists a
    parameter without marking it required lets Claude omit it, and the tool then
    fails at call time instead of being called correctly.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.setdefault("type", "object")
    # Tools take a fixed set of parameters; anything else is a model mistake
    # worth surfacing rather than ignoring.
    schema["additionalProperties"] = False
    return schema
