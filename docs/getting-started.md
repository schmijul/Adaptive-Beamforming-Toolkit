# Getting Started

## 1. Launch the UI

```bash
python app.py
```

The dashboard lets you change:

- number of elements
- element spacing in wavelengths
- steering angles `theta` and `phi`
- taper type

## 2. Use the Python API

```python
import numpy as np
from core.beamforming import array_factor_linear

theta = np.linspace(0.0, 180.0, 181)[:, None] * np.ones((1, 241))
phi = np.ones((181, 1)) * np.linspace(-180.0, 180.0, 241)[None, :]

result = array_factor_linear(
    num_elements=8,
    spacing_lambda=0.5,
    theta_grid_deg=theta,
    phi_grid_deg=phi,
    theta_steer_deg=20.0,
    phi_steer_deg=0.0,
)
```

Expected result:

- `result["magnitude"]` contains a normalized pattern
- `result["magnitude_db"]` contains the same pattern in dB
- the main lobe points near the steering angle

## 3. Run the tests

```bash
pytest -q
```

The tests validate steering, sidelobes, grating lobes, planar arrays, adaptive methods, and IQ helpers.
