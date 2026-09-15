"""The read_file tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .base import ToolDefinition, schema_for


class ReadFileInput(BaseModel):
    """Parameters for the read_file tool."""

    path: str = Field(
        description="The relative path of a file in the working directory.",
    )


def read_file(payload: dict[str, Any]) -> str:
    """Return the contents of a file.

    No sandboxing: the tool reads whatever path it is given, relative to the
    process's working directory.  That is the behaviour of the Go implementation
    this replaces, and it is a deliberate property of a local single-user
    developer tool rather than an oversight -- but it means the agent can read
    anything the user can, so do not point this at an untrusted conversation.
    """
    args = ReadFileInput.model_validate(payload)
    target = Path(args.path)
    if target.is_dir():
        raise IsADirectoryError(f"{args.path} is a directory, not a file")
    return target.read_text(encoding="utf-8")


READ_FILE = ToolDefinition(
    name="read_file",
    description=(
        "Read the contents of a given relative file path. Use this when you "
        "want to see what's inside a file. Do not use this with directory names."
    ),
    input_schema=schema_for(ReadFileInput),
    run=read_file,
)
