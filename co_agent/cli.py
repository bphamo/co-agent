"""Entry point: a terminal chat loop with the read_file tool."""

from __future__ import annotations

import sys

import anthropic
from dotenv import load_dotenv

from .agent import Agent
from .tools import READ_FILE


def read_line() -> tuple[str, bool]:
    """Read one line from stdin.  Returns ``(line, ok)``; ok is False at EOF."""
    line = sys.stdin.readline()
    if line == "":
        return "", False
    return line.rstrip("\n"), True


def main() -> int:
    # A .env file is convenient but not required: the SDK also resolves
    # ANTHROPIC_AUTH_TOKEN and an `ant auth login` profile, so a missing .env is
    # not an error here.  Credentials are checked when the first request is made.
    load_dotenv()

    client = anthropic.Anthropic()
    agent = Agent(client, read_line, [READ_FILE])
    try:
        agent.run()
    except KeyboardInterrupt:
        print()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
