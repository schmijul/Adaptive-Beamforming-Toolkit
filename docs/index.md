# Documentation

The Adaptive Beamforming Toolkit combines a C++ array-factor core with Python helpers for adaptive beamforming, signal simulation, visualization, and a small Dash UI. This documentation is written as a technical companion to the repository: it explains the mathematical model used by the code, the practical meaning of the main parameters, and the limitations of the current implementation.

## Reading Guide

- [Installation](installation.md): environment setup, editable install, and verification steps.
- [Getting Started](getting-started.md): shortest path from clone to dashboard, CLI, and Python usage.
- [Theory](theory.md): beamforming concepts, array tradeoffs, and modeling assumptions.
- [Signal Model](signal-model.md): snapshot model, steering vectors, covariance estimation, and notation.
- [Algorithms](algorithms.md): implemented beamforming and DoA methods, with formulas and interpretation.
- [Examples](examples.md): runnable CLI and Python examples built from the current codebase.
- [API Reference](api-reference.md): module-by-module description of public entry points.
- [Notes](NOTES.md): current scope, simplifications, and known boundaries of the toolkit.

## What This Repository Covers

The current repository includes:

- Uniform linear-array and planar-array factor evaluation.
- Steering, tapering, and deterministic null steering.
- Near-field focusing and far-field plane-wave modes.
- Wideband beam-squint analysis with fixed phase-shifter weights.
- Simplified element-pattern and mutual-coupling impairments.
- MVDR/Capon beamforming and MUSIC direction-of-arrival estimation.
- IQ snapshot simulation, loading, beamforming, and comparison metrics.
- A Dash dashboard plus a CLI for reproducible simulation runs.

## Documentation Conventions

- Angles are expressed in degrees in the public API.
- Element spacing is normalized by wavelength and written as `spacing_lambda = d / lambda`.
- Snapshot matrices are arranged as `(num_elements, num_snapshots)`.
- Pattern magnitudes are normalized so that the peak response is approximately `1` or `0 dB`.
- The theory sections describe the narrowband complex-baseband model used by the implemented algorithms unless stated otherwise.

## Recommended Order

If you are new to the project, read the documentation in this order:

1. [Installation](installation.md)
2. [Getting Started](getting-started.md)
3. [Theory](theory.md)
4. [Signal Model](signal-model.md)
5. [Algorithms](algorithms.md)

If you already know beamforming and only need repo-specific context, start with [Examples](examples.md) and [API Reference](api-reference.md).
