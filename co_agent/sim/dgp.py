"""Return-generating processes with a computable truth.

Real market data has no known answer: you observe one realisation and cannot ask
it what the probability *was*.  A synthetic process can be asked, by simulating
forward from a known state, which makes it the only way to measure whether the
bootstrap estimator is biased rather than merely self-consistent.

That is the division of labour in validation:

* these processes measure the **estimator** -- is ``null_probability`` the number
  it claims to be, on data whose truth we can compute;
* historical walk-forward (:mod:`co_agent.sim.validate`) measures the
  **estimator on the data it will actually see** -- fat tails, gaps, regime
  shifts and all, where truth is unknown and only frequencies are observable.

Neither substitutes for the other.  A process that flatters the estimator here
and fails on history means the process is too kind; passing here and failing
there localises the problem to properties the process does not reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

#: A process's carried state: whatever the next step depends on.  Opaque to
#: callers; only the process interprets it.
State = object


class DGP(Protocol):
    """A return-generating process that can be run forward from a known state."""

    @property
    def name(self) -> str: ...

    def params(self) -> dict[str, object]:
        """Provenance, for the study report."""

    def simulate(
        self, rng: np.random.Generator, n: int, state: State | None = None
    ) -> tuple[np.ndarray, State]:
        """Draw ``n`` log returns, returning them and the terminal state."""

    def forward_paths(
        self,
        rng: np.random.Generator,
        state: State,
        horizon: int,
        paths: int,
    ) -> np.ndarray:
        """Draw ``paths`` independent futures of length ``horizon`` from ``state``.

        This is what makes truth computable: the falsifier's true trip
        probability, conditional on the state the history ended in, is the
        fraction of these futures that trip it.
        """


@dataclass(frozen=True, slots=True)
class IIDNormal:
    """The easy case, and the baseline the estimator must not fail.

    No volatility clustering and no fat tails, so a block bootstrap has nothing
    to preserve and should be close to unbiased here.  A bias on this process is
    a bug in the estimator, not a limitation of the method.
    """

    vol: float = 0.02

    @property
    def name(self) -> str:
        return "iid_normal"

    def params(self) -> dict[str, object]:
        return {"vol": self.vol}

    def simulate(self, rng, n, state=None):
        return rng.normal(0.0, self.vol, n), None

    def forward_paths(self, rng, state, horizon, paths):
        return rng.normal(0.0, self.vol, (paths, horizon))


@dataclass(frozen=True, slots=True)
class StudentT:
    """Fat tails without clustering.

    Isolates the tail problem: GBM's failure mode per FR9 is understating tails,
    and this process says whether the bootstrap inherits it.  Scaled so the
    unconditional standard deviation is ``vol`` regardless of ``df``.
    """

    vol: float = 0.02
    df: float = 4.0

    def __post_init__(self) -> None:
        if self.df <= 2:
            raise ValueError("df must be > 2 for the variance to exist")

    @property
    def name(self) -> str:
        return "student_t"

    def params(self) -> dict[str, object]:
        return {"vol": self.vol, "df": self.df}

    def _scale(self) -> float:
        return self.vol / np.sqrt(self.df / (self.df - 2))

    def simulate(self, rng, n, state=None):
        return rng.standard_t(self.df, n) * self._scale(), None

    def forward_paths(self, rng, state, horizon, paths):
        return rng.standard_t(self.df, (paths, horizon)) * self._scale()


@dataclass(frozen=True, slots=True)
class Garch11:
    """Volatility clustering: the property the block bootstrap exists to keep.

    ``sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]``

    State is the next step's conditional variance, so ``forward_paths`` starts
    every future from the volatility the history ended in.  That is exactly the
    conditioning the simulator's ``cond_vol`` option tries to reproduce, which
    makes this the process that says whether it works.
    """

    omega: float = 8.0e-6
    alpha: float = 0.08
    beta: float = 0.90

    def __post_init__(self) -> None:
        if self.alpha + self.beta >= 1:
            raise ValueError("alpha + beta must be < 1 for a stationary variance")

    @property
    def name(self) -> str:
        return "garch11"

    def params(self) -> dict[str, object]:
        return {"omega": self.omega, "alpha": self.alpha, "beta": self.beta}

    def unconditional_vol(self) -> float:
        return float(np.sqrt(self.omega / (1 - self.alpha - self.beta)))

    def simulate(self, rng, n, state=None):
        sigma2 = float(state) if state is not None else self.omega / (
            1 - self.alpha - self.beta
        )
        out = np.empty(n)
        z = rng.normal(0.0, 1.0, n)
        for t in range(n):
            out[t] = np.sqrt(sigma2) * z[t]
            sigma2 = self.omega + self.alpha * out[t] ** 2 + self.beta * sigma2
        return out, sigma2

    def forward_paths(self, rng, state, horizon, paths):
        sigma2 = np.full(paths, float(state))
        out = np.empty((paths, horizon))
        z = rng.normal(0.0, 1.0, (paths, horizon))
        for t in range(horizon):
            out[:, t] = np.sqrt(sigma2) * z[:, t]
            sigma2 = self.omega + self.alpha * out[:, t] ** 2 + self.beta * sigma2
        return out


@dataclass(frozen=True, slots=True)
class RegimeSwitch:
    """Two volatility regimes with Markov transitions.

    The adversarial case for an unconditional estimator.  A history that spent
    most of its life calm and ended in a storm has an unconditional volatility
    that describes neither, and the null probability computed from it is wrong in
    a direction that depends on which regime the history happens to end in.
    """

    vols: tuple[float, float] = (0.008, 0.035)
    #: ``stay[i]`` is the probability of remaining in regime ``i``.
    stay: tuple[float, float] = (0.98, 0.94)

    @property
    def name(self) -> str:
        return "regime_switch"

    def params(self) -> dict[str, object]:
        return {"vols": list(self.vols), "stay": list(self.stay)}

    def _step_regime(self, rng: np.random.Generator, regime: np.ndarray) -> np.ndarray:
        stay = np.asarray(self.stay)[regime]
        switch = rng.random(regime.shape) > stay
        return np.where(switch, 1 - regime, regime)

    def simulate(self, rng, n, state=None):
        regime = np.array(0 if state is None else int(state))
        out = np.empty(n)
        vols = np.asarray(self.vols)
        for t in range(n):
            out[t] = rng.normal(0.0, vols[regime])
            regime = self._step_regime(rng, regime)
        return out, int(regime)

    def forward_paths(self, rng, state, horizon, paths):
        regime = np.full(paths, int(state))
        vols = np.asarray(self.vols)
        out = np.empty((paths, horizon))
        for t in range(horizon):
            out[:, t] = rng.normal(0.0, 1.0, paths) * vols[regime]
            regime = self._step_regime(rng, regime)
        return out


#: The default panel for a bias study: easy baseline, fat tails, clustering, and
#: the case built to break an unconditional estimator.
DEFAULT_PANEL: tuple[DGP, ...] = (
    IIDNormal(),
    StudentT(),
    Garch11(),
    RegimeSwitch(),
)
