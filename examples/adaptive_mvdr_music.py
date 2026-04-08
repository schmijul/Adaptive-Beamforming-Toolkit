from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abf.algorithms import doa_music_linear, linear_steering_vector, mvdr_weights
from abf.data import simulate_array_iq


def main() -> None:
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
    estimate = float(music["estimated_thetas_deg"][0])

    steer = linear_steering_vector(12, 0.5, estimate, 0.0)
    weights = mvdr_weights(music["covariance_matrix"], steer)
    print(f"MUSIC estimate: {estimate:.2f} deg")
    print(f"MVDR distortionless gain: {np.vdot(weights, steer):.3f}")


if __name__ == "__main__":
    main()
