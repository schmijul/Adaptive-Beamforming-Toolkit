"""Simulation configuration and execution helpers."""

from __future__ import annotations

import warnings

from . import config as _config
from . import runner as _runner

__all__ = [
    "ScenarioConfig",
    "load_scenario_config",
    "run_monte_carlo",
    "run_single_simulation",
]

_FROM_CONFIG = {"ScenarioConfig", "load_scenario_config"}


def __getattr__(name: str):
    if name in __all__:
        warnings.warn(
            "Importing from `simulations` is deprecated and will be removed in the next release. Use `abf.simulations` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        source = _config if name in _FROM_CONFIG else _runner
        value = getattr(source, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
