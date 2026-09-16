"""Resolving falsifiers, blind.

The blinding is enforced by the signature, which is the only enforcement that
cannot be forgotten: :func:`resolve` receives a falsifier and a price path and
has no parameter through which a confidence, a decision, or an author could
reach it. The ledger schema refuses a resolution not marked blinded; this is the
other half of that guarantee.

A thesis is `false` when its falsifier trips, `true` when it survives to
`resolve_by`, and `unresolvable` when the window closed without enough price
data to say either -- a first-class outcome (FR2), never a silent drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..sim.falsifier import Falsifier


@dataclass(frozen=True, slots=True)
class Resolution:
    result: str  # "true" | "false" | "unresolvable"
    resolved_at: date | None
    evidence: str
    #: Always true. The resolver is structurally incapable of seeing anything else.
    blinded: bool = True

    @property
    def tripped(self) -> bool:
        return self.result == "false"


def resolve(
    falsifier: Falsifier,
    entry_close: float,
    observed: list[tuple[date, float]],
    horizon_days: int,
) -> Resolution:
    """Decide a falsifier against the closes observed since entry.

    ``observed`` is (date, close) from the day after entry onward, in order.
    """
    if entry_close <= 0:
        raise ValueError("entry close must be positive")

    for i, (when, close) in enumerate(observed, start=1):
        levels = np.array([[1.0, *[c / entry_close for _, c in observed[:i]]]])
        if bool(falsifier.trips(levels)[0]):
            return Resolution(
                result="false",
                resolved_at=when,
                evidence=(
                    f"falsifier {falsifier.kind} tripped on day {i} of "
                    f"{horizon_days}: close {close:.4f} against entry "
                    f"{entry_close:.4f} ({close / entry_close - 1:+.2%})"
                ),
            )

    if len(observed) >= horizon_days:
        last_date, last_close = observed[horizon_days - 1]
        return Resolution(
            result="true",
            resolved_at=last_date,
            evidence=(
                f"falsifier {falsifier.kind} survived {horizon_days} sessions; "
                f"worst close {min(c for _, c in observed[:horizon_days]) / entry_close - 1:+.2%}, "
                f"final {last_close / entry_close - 1:+.2%}"
            ),
        )

    return Resolution(
        result="unresolvable",
        resolved_at=None,
        evidence=(
            f"only {len(observed)} of {horizon_days} sessions observed; "
            f"the window has not closed"
        ),
    )
