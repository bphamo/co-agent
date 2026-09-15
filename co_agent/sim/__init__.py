"""FR9 simulation service.

Computes ``null_probability`` and ``p95_drawdown`` for a candidate thesis, and
applies the two gates FR9 puts in code: falsifiers that trip by chance too often
or too rarely are returned for restatement, and proposed weights are cut until
the 95th-percentile drawdown fits the per-position limit.

Nothing here calls a model.
"""

from .bootstrap import bootstrap_draws, build_pool, draws_to_levels, ewma_sigma
from .falsifier import (
    DrawdownExceeds,
    Falsifier,
    TerminalAbove,
    TerminalBelow,
    TouchAbove,
    TouchBelow,
    max_drawdown,
)
from .gate import Band, Sizing, Verdict, cap_weight, wilson
from .params import (
    VERSION,
    Drift,
    History,
    InsufficientHistoryError,
    Method,
    Params,
    Prior,
    ProxyInfo,
    SimInputError,
    new_proxy_history,
    sha256_returns,
)
from .simulate import Config, Request, Result, run

__all__ = [
    "VERSION",
    "Band",
    "Config",
    "DrawdownExceeds",
    "Drift",
    "Falsifier",
    "History",
    "InsufficientHistoryError",
    "Method",
    "Params",
    "Prior",
    "ProxyInfo",
    "Request",
    "Result",
    "SimInputError",
    "Sizing",
    "TerminalAbove",
    "TerminalBelow",
    "TouchAbove",
    "TouchBelow",
    "Verdict",
    "bootstrap_draws",
    "build_pool",
    "cap_weight",
    "draws_to_levels",
    "ewma_sigma",
    "max_drawdown",
    "new_proxy_history",
    "run",
    "sha256_returns",
    "wilson",
]
