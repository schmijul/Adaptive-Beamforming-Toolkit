# Adaptive Beamforming Toolkit

Adaptive Beamforming Toolkit is a research-oriented Python package for array-factor simulation, adaptive beamforming experiments, IQ snapshot generation, and interactive visualization. The repository combines a C++ compute core with a stable `abf.*` Python API, a small CLI, and a Dash dashboard.

## What You Get

- fast ULA and planar array-factor evaluation through `core._beamforming_cpp`
- deterministic steering, tapering, and linear null steering
- near-field focusing and far-field pattern evaluation
- wideband beam-squint analysis for phase-steered arrays
- simplified element-pattern and mutual-coupling impairment modeling
- MVDR/Capon, LMS, NLMS, and RLS beamforming plus MUSIC direction finding
- frequency-domain wideband MVDR helpers and MIMO/polarimetric snapshot synthesis
- IQ loading, synthesis, beamforming, and simulation-vs-measurement metrics
- config-driven simulation runs and optional Plotly HTML outputs

## Project Status

The codebase is functional and test-backed, but it is best treated as a compact simulation and exploration toolkit, not as a hardened production beamforming stack.

## Current Limitations

- adaptive processing is still built around narrowband or per-frequency-bin models; there is no true time-delay or STAP-style wideband beamforming path
- MIMO and polarimetric support currently lives in the Python API helpers rather than the CLI or dashboard
- the interactive dashboard remains a compact exploration UI rather than a full multi-geometry control surface
- the toolkit is still research-oriented software, not a hard real-time embedded beamforming stack

## Documentation Map

The full documentation lives in [`docs/`](docs/):

- [`docs/index.md`](docs/index.md): entry point and reading guide
- [`docs/installation.md`](docs/installation.md): setup and verification
- [`docs/getting-started.md`](docs/getting-started.md): first run through UI, CLI, and API
- [`docs/theory.md`](docs/theory.md): beamforming fundamentals and tradeoffs
- [`docs/signal-model.md`](docs/signal-model.md): notation, steering vectors, and covariance model
- [`docs/algorithms.md`](docs/algorithms.md): implemented methods with formulas and interpretation
- [`docs/examples.md`](docs/examples.md): executable examples based on current code
- [`docs/api-reference.md`](docs/api-reference.md): module-level API overview

## Repository Structure

```text
core/         C++ extension and idealized array-factor interfaces
algorithms/   adaptive beamforming and DoA estimation helpers
abf/          stable public Python namespace for external users
data/         IQ loading, simulation, and comparison utilities
simulations/  YAML-driven simulation runner and output generation
visualize/    Plotly figure builders
ui/           Dash dashboard
tests/        analytical and regression-style verification
docs/         technical documentation
examples/     runnable user-facing workflows
config/       example scenario files
imgs/         figures used by the docs
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Requirements:

- Python 3.10+
- a C++17-capable compiler
- `pip` and virtual-environment support

## Quick Start

Launch the dashboard:

```bash
abf dashboard
```

Run one config-driven simulation:

```bash
abf simulate --config config/default.yaml
```

Run the tests:

```bash
pytest -q
```

## CLI

The installed CLI entry point is `abf`.

```bash
abf dashboard
abf simulate --config config/default.yaml
abf montecarlo --config config/default.yaml --runs 50 --jobs 4
abf gallery --config config/default.yaml
```

Current runner scope:

- array geometry: `ula`, `planar`
- algorithms: `conventional`, `mvdr`, `lms`, `nlms`, `rls`

The broader Python API also includes wideband MVDR helpers plus MIMO and polarimetric snapshot utilities beyond the current YAML runner surface.

## Python Example

```python
import numpy as np

from abf.core import array_factor_planar
from abf.algorithms import linear_steering_vector

theta = np.linspace(0.0, 180.0, 401)[:, None] * np.ones((1, 361))
phi = np.ones((401, 1)) * np.linspace(-180.0, 180.0, 361)[None, :]

planar = array_factor_planar(
    num_x=4,
    num_y=4,
    spacing_x_lambda=0.5,
    spacing_y_lambda=0.5,
    theta_grid_deg=theta,
    phi_grid_deg=phi,
    theta_steer_deg=25.0,
    phi_steer_deg=30.0,
)

steer = linear_steering_vector(
    num_elements=8,
    spacing_lambda=0.5,
    theta_deg=20.0,
    phi_deg=0.0,
)
```

External code should prefer the `abf.*` namespace:

- `abf.core`
- `abf.algorithms`
- `abf.data`
- `abf.simulations`
- `abf.visualize`

The older package roots remain importable for compatibility, but they are no longer the recommended public interface.

Runnable examples are included in `examples/`:

- `python examples/linear_array_pattern.py`
- `python examples/adaptive_mvdr_music.py`
- `python examples/reproducible_cli_scenario.py`

## Dashboard Preview

![Dashboard overview](imgs/fullview.png)

![2D elevation cut](imgs/beam-cut.png)

![Theta/Phi heatmap](imgs/beam-heatmap.png)

![3D radiation pattern](imgs/beam-3d.png)

![Amplitude and phase weights](imgs/beam-weights.png)

## Verification

The test suite checks both numerical behavior and higher-level workflows, including:

- steering accuracy and beam pointing
- half-power beamwidth and sidelobe behavior
- grating lobes for large spacing
- taper correctness
- planar-array agreement with an independent NumPy reference
- deterministic null formation
- near-field, wideband, and impairment-aware models
- MVDR, MUSIC, and IQ-data utilities

Run all tests with:

```bash
pytest -q
```

In this branch, test coverage also includes:

- the public `abf` namespace
- CLI JSON output and invalid-config handling
- runnable example scripts

## References

- C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed., Wiley. https://bcs.wiley.com/he-bcs/Books?action=contents&bcsId=9777&itemId=1118642066
- R. J. Mailloux, *Phased Array Antenna Handbook*, 3rd ed., Artech House. https://us.artechhouse.com/Phased-Array-Antenna-Handbook-Third-Edition-P1938.aspx
- J. Capon, "High-resolution frequency-wavenumber spectrum analysis," *Proceedings of the IEEE*, 1969. https://ieeexplore.ieee.org/document/1449208
- R. O. Schmidt, "Multiple emitter location and signal parameter estimation," *IEEE Transactions on Antennas and Propagation*, 1986. https://ieeexplore.ieee.org/document/1143830/
