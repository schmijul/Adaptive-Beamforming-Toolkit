from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abf.core import array_factor_linear


def main() -> None:
    theta = np.linspace(0.0, 180.0, 181)
    phi = np.linspace(-180.0, 180.0, 241)
    theta_grid = theta[:, None] * np.ones((1, phi.size))
    phi_grid = np.ones((theta.size, 1)) * phi[None, :]

    result = array_factor_linear(
        num_elements=8,
        spacing_lambda=0.5,
        theta_grid_deg=theta_grid,
        phi_grid_deg=phi_grid,
        theta_steer_deg=20.0,
        phi_steer_deg=0.0,
    )

    phi_zero_idx = int(np.argmin(np.abs(phi)))
    peak_theta = float(theta[int(np.argmax(result["magnitude"][:, phi_zero_idx]))])
    print(f"Peak steering angle: {peak_theta:.1f} deg")


if __name__ == "__main__":
    main()
