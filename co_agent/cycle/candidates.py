"""Where theses come from.

Two sources behind one interface. The rules-based screen is not a placeholder
for the research agent -- it is the **benchmark arm**. FR7 asks for calibration
against an index, the passed-candidate basket and cash; a deterministic screen
adds the comparison that actually matters, which is whether model-driven
research beats a rule anyone could write in an afternoon. Keeping both behind
one interface means they see identical snapshots and identical gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..sim.falsifier import Falsifier, TouchBelow
from .snapshot import Snapshot

#: Drop thresholds whose realised trip rate lands inside FR9's 0.30-0.70 band,
#: measured on 48 TSX names over non-overlapping windows:
#:
#:   horizon   2%     3%     4%     6%     8%    12%
#:      10d  .445   .331   .243   .137   .078   .031
#:      20d  .572   .465   .369   .239   .154   .072
#:      30d  .626   .528   .439   .306   .209   .106
#:      60d  .706   .620   .541   .415   .321   .191
#:
#: A falsifier written outside this map is not wrong, but the gate will reject
#: most of them as too hard, and the cycle pays model calls for candidates no
#: human ever sees.
GATE_THRESHOLD_BY_HORIZON: dict[int, float] = {10: 0.02, 20: 0.03, 30: 0.04, 60: 0.06}


def suggested_threshold(horizon_days: int) -> float:
    """The drop threshold whose base rate sits nearest mid-band at this horizon."""
    nearest = min(GATE_THRESHOLD_BY_HORIZON, key=lambda h: abs(h - horizon_days))
    return GATE_THRESHOLD_BY_HORIZON[nearest]


@dataclass(frozen=True, slots=True)
class Candidate:
    """A thesis before the gate has seen it."""

    symbol: str
    claim: str
    mechanism: str
    falsifier_text: str
    falsifier: Falsifier
    horizon_days: int
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0,1]")
        for name in ("claim", "mechanism", "falsifier_text"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} is required (FR3: prose is an attachment)")


class CandidateSource(Protocol):
    """Anything that proposes theses against a frozen snapshot."""

    @property
    def name(self) -> str: ...

    def propose(self, snapshot: Snapshot, n: int) -> list[Candidate]: ...


@dataclass(frozen=True, slots=True)
class MomentumScreen:
    """Rank by trailing return; claim the leaders keep leading.

    Deliberately simple and fully deterministic, so that any edge the research
    agent shows has to be an edge over *this*, not over nothing. The claim is
    falsifiable in the FR3 sense and the falsifier is sized to the gate.
    """

    lookback: int = 60
    horizon_days: int = 20
    confidence: float = 0.55

    @property
    def name(self) -> str:
        return f"momentum_screen/{self.lookback}d"

    def propose(self, snapshot: Snapshot, n: int) -> list[Candidate]:
        ranked = sorted(
            snapshot.universe,
            key=lambda s: float(np.exp(snapshot.histories[s][-self.lookback :].sum())),
            reverse=True,
        )
        drop = suggested_threshold(self.horizon_days)
        out: list[Candidate] = []
        for symbol in ranked[:n]:
            trailing = float(np.exp(snapshot.histories[symbol][-self.lookback :].sum()) - 1)
            out.append(
                Candidate(
                    symbol=symbol,
                    claim=(
                        f"{symbol} does not give back its recent advance: it stays "
                        f"within {drop:.0%} of the {snapshot.taken_at} close "
                        f"through the next {self.horizon_days} trading days."
                    ),
                    mechanism=(
                        f"Trailing {self.lookback}-day return of {trailing:+.1%} ranked it "
                        f"top {n} of {len(snapshot.universe)}. The screen asserts "
                        f"short-horizon persistence and nothing else -- no company-specific "
                        f"reasoning is claimed, which is the point of a benchmark arm."
                    ),
                    falsifier_text=(
                        f"{symbol} closes {drop:.0%} or more below its "
                        f"{snapshot.taken_at} close on any of the next "
                        f"{self.horizon_days} trading days."
                    ),
                    falsifier=TouchBelow(drop),
                    horizon_days=self.horizon_days,
                    confidence=self.confidence,
                    source=self.name,
                )
            )
        return out
