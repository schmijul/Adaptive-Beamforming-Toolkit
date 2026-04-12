from __future__ import annotations

import numpy as np

import abf
from abf.algorithms import doa_music_linear, linear_steering_vector, mvdr_weights
from abf.core import array_factor_linear
from abf.data import simulate_array_iq
from abf.simulations import load_scenario_config


def test_abf_namespace_exposes_supported_modules() -> None:
    assert abf.core is not None
    assert abf.algorithms is not None
    assert abf.data is not None
    assert abf.ml is not None
    assert abf.simulations is not None
    assert abf.visualize is not None


def test_abf_core_linear_array_factor_matches_expected_shape() -> None:
    theta = np.linspace(0.0, 180.0, 31)
    phi = np.linspace(-180.0, 180.0, 41)
    theta_grid = theta[:, None] * np.ones((1, phi.size))
    phi_grid = np.ones((theta.size, 1)) * phi[None, :]

    result = array_factor_linear(
        num_elements=8,
        spacing_lambda=0.5,
        theta_grid_deg=theta_grid,
        phi_grid_deg=phi_grid,
        theta_steer_deg=15.0,
        phi_steer_deg=0.0,
    )

    assert result["magnitude"].shape == theta_grid.shape
    assert result["magnitude_db"].shape == theta_grid.shape
    assert result["positions_lambda"].shape == (8,)


def test_abf_algorithms_and_data_modules_interoperate() -> None:
    snapshots = simulate_array_iq(
        num_elements=8,
        num_snapshots=1024,
        spacing_lambda=0.5,
        source_thetas_deg=np.array([20.0]),
        source_phis_deg=np.array([0.0]),
        source_snr_db=np.array([15.0]),
        random_seed=9,
    )
    music = doa_music_linear(
        snapshots=snapshots,
        spacing_lambda=0.5,
        theta_scan_deg=np.linspace(0.0, 90.0, 181),
        num_sources=1,
    )

    steer = linear_steering_vector(8, 0.5, float(music["estimated_thetas_deg"][0]), 0.0)
    weights = mvdr_weights(music["covariance_matrix"], steer)
    assert weights.shape == (8,)


def test_abf_simulations_loads_existing_yaml_scenarios() -> None:
    config = load_scenario_config("config/default.yaml")
    assert config.name == "baseline_mvdr"
