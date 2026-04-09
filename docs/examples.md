# Examples

This repository includes a dedicated `examples/` directory with runnable scripts that exercise the supported public package surface.

## Example Scripts

### `examples/linear_array_pattern.py`

```bash
python examples/linear_array_pattern.py
```

This shows:

- importing `abf.core`
- building a conventional linear-array pattern
- extracting the peak steering angle from the `phi = 0 deg` cut

### `examples/adaptive_mvdr_music.py`

```bash
python examples/adaptive_mvdr_music.py
```

This shows:

- snapshot generation through `abf.data`
- MUSIC DoA estimation through `abf.algorithms`
- MVDR weight synthesis from the estimated arrival direction

### `examples/reproducible_cli_scenario.py`

```bash
python examples/reproducible_cli_scenario.py
```

This shows:

- cloning `config/default.yaml` into a temporary scenario
- running `python -m abf simulate`
- loading the resulting `simulate.json` artifact

## Dashboard

Run the interactive UI:

```bash
python -m abf dashboard
```

or

```bash
abf dashboard
```

Expected outcome:

- a local Dash server starts
- the browser shows the elevation cut, heatmap, 3D pattern, and weight plots
- changing `spacing_lambda` above `0.5` makes grating-lobe behavior easier to observe

## Config-Driven Simulation

Run one deterministic scenario:

```bash
abf simulate --config config/default.yaml
```

This writes `simulate.json` to the configured output directory. The payload contains:

- the scenario configuration
- the generated pattern over the scan grid
- elevation-cut data
- complex-weight summaries
- computed SINR

Run repeated trials:

```bash
abf montecarlo --config config/default.yaml --runs 30
```

This writes `montecarlo.json` with summary statistics such as mean, standard deviation, minimum, and maximum SINR over the selected seeds.

## Equivalent Inline Example: Conventional Pattern

```python
import numpy as np

from abf.core import array_factor_linear

theta = np.linspace(0.0, 180.0, 241)[:, None] * np.ones((1, 241))
phi = np.ones((241, 1)) * np.linspace(-180.0, 180.0, 241)[None, :]

result = array_factor_linear(
    num_elements=12,
    spacing_lambda=0.5,
    theta_grid_deg=theta,
    phi_grid_deg=phi,
    theta_steer_deg=30.0,
    phi_steer_deg=0.0,
    taper_name="uniform",
)

cut_db = result["magnitude_db"][:, int(np.argmin(np.abs(phi[0])))]
```

Use this when you want a deterministic baseline before switching to adaptive processing.

## Equivalent Inline Example: MVDR on Simulated Snapshots

```python
import numpy as np

from abf.algorithms import (
    estimate_covariance_matrix,
    linear_steering_vector,
    mvdr_weights,
)
from abf.data import simulate_array_iq

snapshots = simulate_array_iq(
    num_elements=12,
    num_snapshots=2048,
    spacing_lambda=0.5,
    source_thetas_deg=np.array([25.0]),
    source_phis_deg=np.array([0.0]),
    source_snr_db=np.array([20.0]),
    random_seed=7,
)

covariance = estimate_covariance_matrix(snapshots, diagonal_loading=1e-3)
steering = linear_steering_vector(12, 0.5, 25.0, 0.0)
weights = mvdr_weights(covariance, steering, diagonal_loading=1e-3)
```

This example follows the same path used in the adaptive regression tests.

## Equivalent Inline Example: MUSIC Scan

```python
import numpy as np

from abf.algorithms import doa_music_linear
from abf.data import simulate_array_iq

snapshots = simulate_array_iq(
    num_elements=12,
    num_snapshots=2048,
    spacing_lambda=0.5,
    source_thetas_deg=np.array([25.0]),
    source_phis_deg=np.array([0.0]),
    source_snr_db=np.array([20.0]),
    random_seed=7,
)

music = doa_music_linear(
    snapshots=snapshots,
    spacing_lambda=0.5,
    theta_scan_deg=np.linspace(0.0, 90.0, 361),
    num_sources=1,
)

estimated_theta = float(music["estimated_thetas_deg"][0])
```

## Test Files as Executable Documentation

The following tests are especially useful as worked examples:

- `tests/test_ground_truth.py`: deterministic array-factor behavior, tapering, nulls, and planar checks
- `tests/test_next_steps.py`: near-field, wideband, impairments, MVDR, MUSIC, and IQ workflows
- `tests/test_simulation_runner.py`: config loading and artifact generation

If you need realistic usage patterns that are guaranteed to stay aligned with the implementation, start with the example scripts first and then use the tests as lower-level reference material.
