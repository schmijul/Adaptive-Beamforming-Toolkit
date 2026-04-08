# Adaptive Beamforming Toolkit

Interactive beamforming simulator with a C++ compute core and a Python UI layer.

## Implemented core features

- C++ linear array factor core importable from Python
- C++ planar array factor core importable from Python
- Steering in `theta` and `phi`
- `d/lambda` controls to compare `lambda/2` against larger spacing
- Amplitude tapers: uniform, Hamming, Taylor
- Separate amplitude and phase weights in the core model
- Linear null-steering weights with active interference suppression
- 2D cut, theta/phi heatmap, and interactive 3D pattern
- Dash UI with live controls for the MVP dashboard

## Repo structure

```text
core/        array math and weighting
algorithms/  placeholder for MVDR, MUSIC, nulling
visualize/   plot builders
ui/          Dash dashboard
data/        simulated and measured datasets
notebooks/   demos
docs/        theory notes
```

## Build and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python app.py
```

Then open `http://127.0.0.1:8050`.

## Python API

The native core is exposed as `core._beamforming_cpp` and re-exported via `core.beamforming`.

Available entry points include:

- `array_factor_linear(...)`
- `array_factor_planar(...)`
- `steering_weights_linear(...)`
- `steering_weights_planar(...)`
- `null_steering_weights_linear(...)`
- `array_factor_linear_from_weights(...)`

Example:

```python
import numpy as np
from core.beamforming import array_factor_planar, null_steering_weights_linear

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

weights = null_steering_weights_linear(
    num_elements=8,
    spacing_lambda=0.5,
    theta_steer_deg=0.0,
    phi_steer_deg=0.0,
    null_thetas_deg=np.array([20.0]),
    null_phis_deg=np.array([0.0]),
)
```

## Verification

The core has analytical and reference-driven tests for:

- linear steering accuracy
- half-power beamwidth
- sidelobe levels
- grating lobes
- taper correctness
- planar array factor against an independent NumPy reference
- null-steering constraint satisfaction and notch formation

Run them with:

```bash
source .venv/bin/activate
pytest -q
```

Current status: `13 passed`

## Next steps

- Near-field vs far-field toggle
- Analog/digital/hybrid beamforming models
- Wideband beam squint
- Element patterns and mutual coupling
- MVDR and MUSIC
- IQ import and sim vs measurement overlays
