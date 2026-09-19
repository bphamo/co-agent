"""Candidate universes for validation studies.

These are **placeholder lists for validating the estimator**, not a trading
universe. TRD open question 2 (hand-picked list vs rules-based screen) is still
open and does not need to be settled to measure whether ``null_probability`` is
calibrated -- any set of liquid names with long history will do for that.

**Survivorship.** Every list here was assembled today, so it contains only names
that still trade. Everything acquired, delisted or wound up is missing, and that
is where the large drawdowns are. A walk-forward over these names will therefore
understate realised trip frequencies for downside falsifiers -- in the same
direction as the estimator's measured bias, which means the two are confounded
and a result cannot cleanly separate them. Names that suffered severe drawdowns
*without* delisting are included deliberately to claw back some tail coverage,
but that is mitigation, not a fix.
"""

from __future__ import annotations

#: Liquid TSX names with long history, spread across the sectors that dominate
#: the index. Yahoo ticker format: class shares use a dash (``RCI-B.TO``).
TSX_PLACEHOLDER: tuple[str, ...] = (
    # Banks and insurers -- roughly a third of the TSX by weight.
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    "MFC.TO", "SLF.TO", "GWO.TO", "IFC.TO", "POW.TO",
    # Energy -- the 2015 and 2020 drawdowns are the tail this study needs.
    "ENB.TO", "TRP.TO", "SU.TO", "CNQ.TO", "IMO.TO", "CVE.TO", "PPL.TO",
    # Materials and mining.
    "ABX.TO", "AEM.TO", "K.TO", "FNV.TO", "WPM.TO", "TECK-B.TO", "FM.TO",
    # Industrials and transport.
    "CNR.TO", "CP.TO", "WSP.TO", "TFII.TO", "STN.TO",
    # Telecom and utilities. AQN and BB are here on purpose: both fell hard
    # without delisting, so they carry tail behaviour a survivor list usually
    # loses.
    "BCE.TO", "T.TO", "RCI-B.TO", "FTS.TO", "EMA.TO", "AQN.TO", "H.TO",
    # Consumer.
    "L.TO", "ATD.TO", "MRU.TO", "QSR.TO", "DOL.TO", "CTC-A.TO",
    # Technology.
    "CSU.TO", "OTEX.TO", "GIB-A.TO", "SHOP.TO", "BB.TO",
)

UNIVERSES: dict[str, tuple[str, ...]] = {"tsx": TSX_PLACEHOLDER}
