"""Price-path falsifier predicates.

A predicate is evaluated against a batch of synthetic paths at once: ``levels``
is a ``(paths, horizon + 1)`` array of price levels normalised so that
``levels[:, 0] == 1``.  Only the predicate sees it; no path leaves this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .params import SimInputError


@runtime_checkable
class Falsifier(Protocol):
    """A price-path predicate."""

    def trips(self, levels: np.ndarray) -> np.ndarray:
        """Return a boolean array, one entry per path."""

    @property
    def kind(self) -> str: ...

    def spec(self) -> dict[str, object]:
        """Provenance for ``sim_params.falsifier``."""

    def validate(self) -> None:
        """Raise :class:`SimInputError` if the predicate is malformed."""


def _check_fraction(name: str, value: float) -> None:
    if not 0 < value < 1:
        raise SimInputError(f"{name} must be in (0,1), got {value!r}")


def _check_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise SimInputError(f"{name} must be >= 0, got {value!r}")


@dataclass(frozen=True, slots=True)
class TouchBelow:
    """Trips if the level falls ``drop`` below the start at any point.

    ``drop`` is a positive fraction: 0.12 means "down 12%".
    """

    drop: float

    @property
    def kind(self) -> str:
        return "touch_below"

    def spec(self) -> dict[str, object]:
        return {"kind": self.kind, "drop": self.drop}

    def validate(self) -> None:
        _check_fraction("drop", self.drop)

    def trips(self, levels: np.ndarray) -> np.ndarray:
        return (levels <= 1 - self.drop).any(axis=1)


@dataclass(frozen=True, slots=True)
class TouchAbove:
    """Trips if the level rises ``rise`` above the start at any point."""

    rise: float

    @property
    def kind(self) -> str:
        return "touch_above"

    def spec(self) -> dict[str, object]:
        return {"kind": self.kind, "rise": self.rise}

    def validate(self) -> None:
        _check_non_negative("rise", self.rise)

    def trips(self, levels: np.ndarray) -> np.ndarray:
        return (levels >= 1 + self.rise).any(axis=1)


@dataclass(frozen=True, slots=True)
class TerminalBelow:
    """Trips on the horizon date only.

    "Closes below" rather than "trades below" -- a materially different and
    usually much harder falsifier, which is exactly the kind of difference
    ``null_probability`` exists to expose.
    """

    drop: float

    @property
    def kind(self) -> str:
        return "terminal_below"

    def spec(self) -> dict[str, object]:
        return {"kind": self.kind, "drop": self.drop}

    def validate(self) -> None:
        _check_fraction("drop", self.drop)

    def trips(self, levels: np.ndarray) -> np.ndarray:
        return levels[:, -1] <= 1 - self.drop


@dataclass(frozen=True, slots=True)
class TerminalAbove:
    """Trips on the horizon date only."""

    rise: float

    @property
    def kind(self) -> str:
        return "terminal_above"

    def spec(self) -> dict[str, object]:
        return {"kind": self.kind, "rise": self.rise}

    def validate(self) -> None:
        _check_non_negative("rise", self.rise)

    def trips(self, levels: np.ndarray) -> np.ndarray:
        return levels[:, -1] >= 1 + self.rise


@dataclass(frozen=True, slots=True)
class DrawdownExceeds:
    """Trips if peak-to-trough drawdown within the horizon exceeds ``limit``."""

    limit: float

    @property
    def kind(self) -> str:
        return "drawdown_exceeds"

    def spec(self) -> dict[str, object]:
        return {"kind": self.kind, "limit": self.limit}

    def validate(self) -> None:
        _check_fraction("limit", self.limit)

    def trips(self, levels: np.ndarray) -> np.ndarray:
        return max_drawdown(levels) >= self.limit


def max_drawdown(levels: np.ndarray) -> np.ndarray:
    """Largest peak-to-trough decline per path, as a fraction."""
    peak = np.maximum.accumulate(levels, axis=1)
    return ((peak - levels) / peak).max(axis=1)
