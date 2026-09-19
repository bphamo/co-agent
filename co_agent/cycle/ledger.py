"""An append-only file ledger.

A stand-in for the Postgres schema in `db/`, with the same field names, for
running before a database exists. Append-only is structural here: the file is
only ever opened for append, and there is no update or delete method to call.

This is not the real ledger. It has no foreign keys, no CHECK constraints and no
privilege separation -- the three things that make `db/` trustworthy. Move to
Postgres before anything depends on the history being unedited.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator


def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _encode(v) for k, v in asdict(value).items()}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


class Ledger:
    """One JSONL file per record kind, under ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, kind: str) -> Path:
        return self.root / f"{kind}.jsonl"

    def append(self, kind: str, record: Any) -> None:
        with self.path(kind).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_encode(record), sort_keys=True) + "\n")

    def read(self, kind: str) -> Iterator[dict]:
        path = self.path(kind)
        if not path.exists():
            return iter(())
        with path.open(encoding="utf-8") as handle:
            return iter([json.loads(line) for line in handle if line.strip()])

    def count(self, kind: str) -> int:
        return sum(1 for _ in self.read(kind))
