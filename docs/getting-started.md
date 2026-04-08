# Getting Started

This page is the shortest path from a fresh clone to useful output.

## 1. Launch the Dashboard

Run either of the following:

```bash
python app.py
```

```bash
abf dashboard
```

The dashboard exposes live controls for:

- number of elements
- spacing in wavelengths
- steering angles `theta` and `phi`
- amplitude taper

The main plots are:

- an elevation cut at `phi = 0 deg`
- a theta/phi heatmap
- a 3D radiation-pattern view
- amplitude and phase-weight plots

## 2. Call the Python API

The supported public import root is `abf.*`.

```python
import numpy as np

from abf.core import array_factor_linear

theta = np.linspace(0.0, 180.0, 181)[:, None] * np.ones((1, 241))
phi = np.ones((181, 1)) * np.linspace(-180.0, 180.0, 241)[None, :]

result = array_factor_linear(
    num_elements=8,
    spacing_lambda=0.5,
    theta_grid_deg=theta,
    phi_grid_deg=phi,
    theta_steer_deg=20.0,
    phi_steer_deg=0.0,
    taper_name="hamming",
)
```

Important fields in the returned dictionary:

- `result["magnitude"]`: normalized linear magnitude
- `result["magnitude_db"]`: normalized magnitude in dB
- `result["weights"]`: complex steering/taper weights
- `result["positions_lambda"]`: element positions in wavelength units

Other public modules follow the same namespace pattern:

- `abf.algorithms`
- `abf.data`
- `abf.simulations`
- `abf.visualize`

## 3. Run a Reproducible Simulation

The repository also supports config-driven runs through the CLI:

```bash
abf simulate --config config/default.yaml
```

This produces a JSON artifact in the configured output directory. For repeated trials:

```bash
abf montecarlo --config config/default.yaml --runs 50
```

The current runner supports:

- `ula` geometry
- `conventional` and `mvdr` weight selection
- one desired source plus zero or more interferers
- optional HTML plot export

## 4. Check the Reference Tests

```bash
pytest -q
```

The tests are not just smoke tests. They check specific beamforming behavior, including:

- steering accuracy
- half-power beamwidth consistency
- expected sidelobe levels
- grating-lobe behavior for large spacing
- null-steering constraints
- MVDR/MUSIC operation on simulated snapshots
- wideband squint and impairment-aware responses

## 5. Run the Example Scripts

```bash
python examples/linear_array_pattern.py
python examples/adaptive_mvdr_music.py
python examples/reproducible_cli_scenario.py
```

These examples are intended as the shortest external-user workflows that stay aligned with the supported public namespace and CLI behavior.

## 6. Know the Current Scope

This toolkit is strongest as a simulation and algorithm-exploration environment. The current repository is not a full phased-array hardware stack, a calibrated measurement framework, or a large-scale production beamforming library.

For the mathematical background behind the code, continue with [Theory](theory.md) and [Signal Model](signal-model.md).
