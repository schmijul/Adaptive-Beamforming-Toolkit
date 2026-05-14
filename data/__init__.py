"""IQ import and simulation helpers."""

from __future__ import annotations

import warnings

from . import iq as _iq

__all__ = [
    "apply_element_calibration",
    "beamform_iq",
    "compare_sim_vs_measurement",
    "estimate_element_calibration",
    "estimate_element_response",
    "load_iq_samples",
    "simulate_array_iq",
    "simulate_array_iq_components",
    "simulate_mimo_iq",
    "simulate_polarimetric_array_iq",
]


def __getattr__(name: str):
    if name in __all__:
        warnings.warn(
            "Importing from `data` is deprecated and will be removed in the next release. Use `abf.data` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = getattr(_iq, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
