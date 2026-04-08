# Documentation

Adaptive Beamforming Toolkit is a small Python package for simulating array patterns, testing beamforming ideas, and running a simple Dash UI on top of a C++ compute core.

## Pages

- [Installation](installation.md): environment setup and package install
- [Getting Started](getting-started.md): quickest path to running the app and tests
- [Theory](theory.md): core beamforming concepts used in this repo
- [Signal Model](signal-model.md): how signals, steering vectors, and snapshots are represented
- [Algorithms](algorithms.md): summary of conventional and adaptive methods
- [Examples](examples.md): runnable entrypoints and test-backed usage patterns
- [API Reference](api-reference.md): public functions and modules

## Scope

This toolkit currently includes:

- Linear and planar array factor models
- Steering and null-steering weights
- Near-field, wideband, and impairment-aware extensions
- MVDR beamforming and MUSIC DoA estimation
- IQ simulation, loading, and comparison helpers
- A Dash dashboard for interactive exploration
