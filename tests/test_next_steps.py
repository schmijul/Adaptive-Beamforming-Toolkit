from __future__ import annotations

import numpy as np

from algorithms.adaptive import doa_music_linear, linear_steering_vector, mvdr_weights
from core.advanced_models import (
    array_factor_linear_field_mode,
    array_factor_linear_with_impairments,
    build_mutual_coupling_matrix,
    synthesize_beamforming_architecture,
    wideband_array_factor_linear,
)
from core.beamforming import steering_weights_linear
from data.iq import compare_sim_vs_measurement, load_iq_samples, simulate_array_iq


THETA = np.linspace(0.0, 180.0, 241)
PHI = np.linspace(-180.0, 180.0, 241)
THETA_GRID = THETA[:, None] * np.ones((1, PHI.size))
PHI_GRID = np.ones((THETA.size, 1)) * PHI[None, :]
PHI_ZERO_INDEX = int(np.argmin(np.abs(PHI)))


def test_near_and_far_field_modes_produce_valid_patterns() -> None:
    far = array_factor_linear_field_mode(
        num_elements=12,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=40.0,
        phi_steer_deg=0.0,
        field_mode="far",
    )
    near = array_factor_linear_field_mode(
        num_elements=12,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=40.0,
        phi_steer_deg=0.0,
        field_mode="near",
        focus_range_lambda=8.0,
    )
    assert far["magnitude"].shape == THETA_GRID.shape
    assert near["magnitude"].shape == THETA_GRID.shape
    assert not np.allclose(far["magnitude"], near["magnitude"])


def test_digital_analog_hybrid_architectures_are_constructed() -> None:
    ideal = np.exp(1j * np.linspace(-1.0, 1.0, 8))
    digital = synthesize_beamforming_architecture(ideal, architecture="digital")
    analog = synthesize_beamforming_architecture(ideal, architecture="analog", phase_bits=5)
    hybrid = synthesize_beamforming_architecture(ideal, architecture="hybrid", num_rf_chains=2, phase_bits=5)

    assert np.allclose(digital.weights, ideal)
    assert np.allclose(np.abs(analog.weights), np.abs(analog.weights[0]))
    assert hybrid.rf_weights.shape == (8, 2)
    assert hybrid.digital_weights.shape == (2, 1)


def test_wideband_model_shows_beam_squint() -> None:
    f0 = 10e9
    freqs = np.array([8e9, 10e9, 12e9], dtype=float)
    result = wideband_array_factor_linear(
        num_elements=16,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=50.0,
        phi_steer_deg=0.0,
        center_frequency_hz=f0,
        frequency_hz=freqs,
    )
    cut_lo = result["magnitude"][0, :, PHI_ZERO_INDEX]
    cut_hi = result["magnitude"][2, :, PHI_ZERO_INDEX]
    peak_lo = float(THETA[int(np.argmax(cut_lo))])
    peak_hi = float(THETA[int(np.argmax(cut_hi))])
    assert abs(peak_lo - peak_hi) >= 4.0


def test_element_pattern_and_coupling_are_applied() -> None:
    coupling = build_mutual_coupling_matrix(num_elements=10, nearest_neighbor_db=-12.0, phase_deg=-20.0)
    impaired = array_factor_linear_with_impairments(
        num_elements=10,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=0.0,
        phi_steer_deg=0.0,
        element_pattern_name="cosine",
        element_pattern_exponent=2.0,
        coupling_matrix=coupling,
    )
    broadside = float(impaired["magnitude"][int(np.argmin(np.abs(THETA - 0.0))), PHI_ZERO_INDEX])
    near_horizon = float(impaired["magnitude"][int(np.argmin(np.abs(THETA - 90.0))), PHI_ZERO_INDEX])
    assert broadside > near_horizon


def test_mvdr_and_music_operate_on_simulated_snapshots() -> None:
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
    assert abs(estimate - 25.0) <= 2.0

    steer = linear_steering_vector(12, 0.5, 25.0, 0.0)
    covariance = music["covariance_matrix"]
    weights = mvdr_weights(covariance, steer)
    assert weights.shape == (12,)
    gain = np.vdot(weights, steer)
    assert np.isclose(gain, 1.0 + 0.0j, atol=1e-2)


def test_iq_import_and_overlay_metrics(tmp_path) -> None:
    n = 256
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    measured = np.exp(1j * 2.0 * np.pi * 5.0 * t)
    csv_data = np.column_stack([measured.real, measured.imag])
    iq_path = tmp_path / "measured_iq.csv"
    np.savetxt(iq_path, csv_data, delimiter=",")

    loaded = load_iq_samples(iq_path)
    assert loaded.shape == (n,)
    assert np.allclose(loaded, measured)

    simulated = measured + 0.02 * (np.random.default_rng(123).normal(size=n) + 1j * np.random.default_rng(123).normal(size=n))
    metrics = compare_sim_vs_measurement(simulated, loaded)
    assert metrics["correlation"] > 0.95
    assert metrics["nmse_db"] < -20.0


def test_measurement_overlay_works_with_steering_weights() -> None:
    weights, _, _ = steering_weights_linear(
        num_elements=8,
        spacing_lambda=0.5,
        theta_steer_deg=0.0,
        phi_steer_deg=0.0,
    )
    snapshots = simulate_array_iq(
        num_elements=8,
        num_snapshots=1024,
        spacing_lambda=0.5,
        source_thetas_deg=np.array([0.0]),
        source_phis_deg=np.array([0.0]),
        source_snr_db=np.array([15.0]),
        random_seed=3,
    )
    beamformed = np.conj(np.asarray(weights)) @ snapshots
    assert beamformed.shape == (1024,)
