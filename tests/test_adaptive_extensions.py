from __future__ import annotations

import numpy as np

from abf.algorithms import (
    estimate_wideband_covariance_matrices,
    lms_weights,
    mimo_virtual_steering_vector_linear,
    nlms_weights,
    planar_steering_vector,
    polarimetric_steering_vector,
    rls_weights,
    wideband_linear_steering_vectors,
    wideband_mvdr_weights,
)
from abf.data import simulate_array_iq, simulate_mimo_iq, simulate_polarimetric_array_iq


def test_lms_nlms_and_rls_recover_complex_weights() -> None:
    rng = np.random.default_rng(12)
    snapshots = (
        rng.normal(size=(3, 2000)) + 1j * rng.normal(size=(3, 2000))
    ) / np.sqrt(2.0)
    true_weights = np.array([0.8 + 0.1j, -0.4 + 0.3j, 0.25 - 0.2j], dtype=np.complex128)
    desired = np.conj(true_weights) @ snapshots

    lms = lms_weights(snapshots, desired, step_size=0.02)
    nlms = nlms_weights(snapshots, desired, step_size=0.6)
    rls = rls_weights(snapshots, desired, forgetting_factor=0.995, initialization_delta=0.5)

    assert np.linalg.norm(lms["weights"] - true_weights) < 0.15
    assert np.linalg.norm(nlms["weights"] - true_weights) < 0.08
    assert np.linalg.norm(rls["weights"] - true_weights) < 0.02


def test_planar_geometry_simulation_matches_expected_channel_count() -> None:
    snapshots = simulate_array_iq(
        num_elements=6,
        num_snapshots=128,
        spacing_lambda=None,
        source_thetas_deg=np.array([20.0]),
        source_phis_deg=np.array([10.0]),
        source_snr_db=np.array([15.0]),
        random_seed=5,
        geometry="planar",
        num_x=3,
        num_y=2,
        spacing_x_lambda=0.5,
        spacing_y_lambda=0.5,
    )
    steer = planar_steering_vector(3, 2, 0.5, 0.5, 20.0, 10.0)
    assert snapshots.shape == (6, 128)
    assert steer.shape == (6,)


def test_wideband_mvdr_operates_per_frequency_bin() -> None:
    rng = np.random.default_rng(7)
    freqs = np.array([8.0e9, 10.0e9, 12.0e9], dtype=float)
    steering = wideband_linear_steering_vectors(
        num_elements=6,
        spacing_lambda=0.5,
        theta_deg=25.0,
        phi_deg=0.0,
        center_frequency_hz=10.0e9,
        frequency_hz=freqs,
    )
    source = (
        rng.normal(size=(freqs.size, 512)) + 1j * rng.normal(size=(freqs.size, 512))
    ) / np.sqrt(2.0)
    noise = 0.05 * (
        rng.normal(size=(freqs.size, 6, 512)) + 1j * rng.normal(size=(freqs.size, 6, 512))
    ) / np.sqrt(2.0)
    frequency_snapshots = steering[:, :, None] * source[:, None, :] + noise

    covariance = estimate_wideband_covariance_matrices(frequency_snapshots, diagonal_loading=1e-4)
    weights = wideband_mvdr_weights(covariance, steering, diagonal_loading=1e-4)
    distortionless = np.sum(np.conj(weights) * steering, axis=1)

    assert weights.shape == steering.shape
    assert np.allclose(distortionless, np.ones(freqs.size), atol=2e-2)


def test_mimo_and_polarimetric_helpers_produce_stackable_channels() -> None:
    mimo = simulate_mimo_iq(
        num_tx=2,
        num_rx=3,
        num_snapshots=64,
        tx_spacing_lambda=0.5,
        rx_spacing_lambda=0.5,
        source_thetas_deg=np.array([15.0]),
        source_phis_deg=np.array([0.0]),
        source_snr_db=np.array([10.0]),
        random_seed=4,
    )
    mimo_steer = mimo_virtual_steering_vector_linear(2, 3, 0.5, 0.5, 15.0, 0.0)

    spatial = np.exp(1j * np.linspace(0.0, 0.4, 4))
    polarization = np.array([1.0 + 0.0j, 0.5j], dtype=np.complex128)
    polarimetric = polarimetric_steering_vector(spatial, polarization)
    polarimetric_snapshots = simulate_polarimetric_array_iq(
        num_elements=4,
        num_snapshots=64,
        spacing_lambda=0.5,
        source_thetas_deg=np.array([10.0]),
        source_phis_deg=np.array([0.0]),
        source_snr_db=np.array([12.0]),
        source_polarizations=np.array([[1.0 + 0.0j, 0.5j]], dtype=np.complex128),
        random_seed=9,
    )

    assert mimo.shape == (6, 64)
    assert mimo_steer.shape == (6,)
    assert polarimetric.shape == (8,)
    assert polarimetric_snapshots.shape == (8, 64)
