"""Adaptive beamforming algorithms."""

from __future__ import annotations

import warnings

from . import adaptive as _adaptive

__all__ = [
    "beamform_frequency_snapshots",
    "doa_music_linear",
    "estimate_covariance_matrix",
    "estimate_wideband_covariance_matrices",
    "linear_steering_vector",
    "lms_weights",
    "music_spectrum",
    "mvdr_weights",
    "mimo_virtual_steering_vector_linear",
    "nlms_weights",
    "planar_steering_vector",
    "polarimetric_steering_vector",
    "rls_weights",
    "wideband_linear_steering_vectors",
    "wideband_mvdr_weights",
]


def __getattr__(name: str):
    if name in __all__:
        warnings.warn(
            "Importing from `algorithms` is deprecated and will be removed in the next release. Use `abf.algorithms` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = getattr(_adaptive, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
