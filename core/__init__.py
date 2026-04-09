"""Core beamforming models."""

from __future__ import annotations

import warnings

from . import advanced_models as _advanced_models
from . import beamforming as _beamforming

__all__ = [
    "ArchitectureWeights",
    "amplitude_taper",
    "array_factor_linear",
    "array_factor_linear_field_mode",
    "array_factor_linear_from_weights",
    "array_factor_linear_near_field",
    "array_factor_linear_with_impairments",
    "array_factor_planar_from_weights",
    "array_factor_planar",
    "build_mutual_coupling_matrix",
    "element_pattern_gain",
    "element_positions_linear",
    "element_positions_planar",
    "null_steering_weights_linear",
    "steering_weights_linear",
    "steering_weights_near_field_linear",
    "steering_weights_planar",
    "synthesize_beamforming_architecture",
    "wideband_array_factor_linear",
]

_FROM_BEAMFORMING = {
    "amplitude_taper",
    "array_factor_linear",
    "array_factor_linear_from_weights",
    "array_factor_planar",
    "element_positions_linear",
    "element_positions_planar",
    "null_steering_weights_linear",
    "steering_weights_linear",
    "steering_weights_planar",
}


def __getattr__(name: str):
    if name in __all__:
        warnings.warn(
            "Importing from `core` is deprecated and will be removed in the next release. Use `abf.core` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        source = _beamforming if name in _FROM_BEAMFORMING else _advanced_models
        value = getattr(source, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
