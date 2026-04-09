"""Visualization helpers."""

from __future__ import annotations

import warnings

from . import plots as _plots

__all__ = [
    "build_elevation_cut",
    "build_heatmap",
    "build_pattern_3d",
    "build_weights_plot",
]


def __getattr__(name: str):
    if name in __all__:
        warnings.warn(
            "Importing from `visualize` is deprecated and will be removed in the next release. Use `abf.visualize` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = getattr(_plots, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
