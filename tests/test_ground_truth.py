from __future__ import annotations

import math

import numpy as np

from core.beamforming import (
    amplitude_taper,
    array_factor_linear,
    array_factor_linear_from_weights,
    array_factor_planar,
    null_steering_weights_linear,
)


THETA_AXIS = np.linspace(0.0, 180.0, 4001)
PHI_AXIS = np.linspace(-180.0, 180.0, 721)
PHI_ZERO_INDEX = int(np.argmin(np.abs(PHI_AXIS)))
THETA_GRID = THETA_AXIS[:, None] * np.ones((1, PHI_AXIS.size))
PHI_GRID = np.ones((THETA_AXIS.size, 1)) * PHI_AXIS[None, :]


def _run_array_factor(
    num_elements: int,
    spacing_lambda: float,
    theta_steer_deg: float,
    phi_steer_deg: float,
    taper_name: str = "uniform",
):
    return array_factor_linear(
        num_elements=num_elements,
        spacing_lambda=spacing_lambda,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=theta_steer_deg,
        phi_steer_deg=phi_steer_deg,
        taper_name=taper_name,
    )


def _reference_amplitude_taper(num_elements: int, taper_name: str) -> np.ndarray:
    taper_name = taper_name.lower()
    if taper_name == "uniform":
        return np.ones(num_elements, dtype=float)
    if taper_name == "hamming":
        window = np.array(
            [0.54 - 0.46 * np.cos((2.0 * np.pi * n) / (num_elements - 1)) for n in range(num_elements)],
            dtype=float,
        )
        return window / window.max()
    raise ValueError(f"Unsupported taper in reference model: {taper_name}")


def _reference_cut(
    num_elements: int,
    spacing_lambda: float,
    theta_axis_deg: np.ndarray,
    theta_steer_deg: float,
    phi_steer_deg: float,
    taper_name: str = "uniform",
) -> np.ndarray:
    positions = (np.arange(num_elements, dtype=float) - (num_elements - 1) / 2.0) * spacing_lambda
    amplitudes = _reference_amplitude_taper(num_elements, taper_name)
    u0 = np.sin(np.deg2rad(theta_steer_deg)) * np.cos(np.deg2rad(phi_steer_deg))
    u = np.sin(np.deg2rad(theta_axis_deg))

    response = np.zeros(theta_axis_deg.shape, dtype=np.complex128)
    for index, position in enumerate(positions):
        response += amplitudes[index] * np.exp(1j * 2.0 * np.pi * position * (u - u0))

    magnitude = np.abs(response)
    return magnitude / magnitude.max()


def _reference_planar_pattern(
    num_x: int,
    num_y: int,
    spacing_x_lambda: float,
    spacing_y_lambda: float,
    theta_grid_deg: np.ndarray,
    phi_grid_deg: np.ndarray,
    theta_steer_deg: float,
    phi_steer_deg: float,
    taper_x: str = "uniform",
    taper_y: str = "uniform",
) -> np.ndarray:
    x_positions = (np.arange(num_x, dtype=float) - (num_x - 1) / 2.0) * spacing_x_lambda
    y_positions = (np.arange(num_y, dtype=float) - (num_y - 1) / 2.0) * spacing_y_lambda
    taper_x_values = _reference_amplitude_taper(num_x, taper_x)
    taper_y_values = _reference_amplitude_taper(num_y, taper_y)
    ux0 = np.sin(np.deg2rad(theta_steer_deg)) * np.cos(np.deg2rad(phi_steer_deg))
    uy0 = np.sin(np.deg2rad(theta_steer_deg)) * np.sin(np.deg2rad(phi_steer_deg))
    ux = np.sin(np.deg2rad(theta_grid_deg)) * np.cos(np.deg2rad(phi_grid_deg))
    uy = np.sin(np.deg2rad(theta_grid_deg)) * np.sin(np.deg2rad(phi_grid_deg))

    response = np.zeros(theta_grid_deg.shape, dtype=np.complex128)
    for iy, y_pos in enumerate(y_positions):
        for ix, x_pos in enumerate(x_positions):
            amplitude = taper_x_values[ix] * taper_y_values[iy]
            phase = 2.0 * np.pi * (x_pos * (ux - ux0) + y_pos * (uy - uy0))
            response += amplitude * np.exp(1j * phase)

    magnitude = np.abs(response)
    return magnitude / magnitude.max()


def _find_local_maxima(values: np.ndarray) -> np.ndarray:
    return np.where((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1


def _half_power_beamwidth(theta_axis_deg: np.ndarray, magnitude: np.ndarray) -> float:
    peak_index = int(np.argmax(magnitude))
    half_power = 1.0 / math.sqrt(2.0)

    left_indices = np.where(magnitude[: peak_index + 1] < half_power)[0]
    right_indices = np.where(magnitude[peak_index:] < half_power)[0]
    if left_indices.size == 0 or right_indices.size == 0:
        raise AssertionError("half-power crossing not found")

    left_high = int(left_indices[-1] + 1)
    left_low = int(left_indices[-1])
    right_low = int(peak_index + right_indices[0] - 1)
    right_high = int(peak_index + right_indices[0])

    left_theta = np.interp(
        half_power,
        [magnitude[left_low], magnitude[left_high]],
        [theta_axis_deg[left_low], theta_axis_deg[left_high]],
    )
    right_theta = np.interp(
        half_power,
        [magnitude[right_high], magnitude[right_low]],
        [theta_axis_deg[right_high], theta_axis_deg[right_low]],
    )
    return float(right_theta - left_theta)


def _first_sidelobe_level_db(magnitude: np.ndarray) -> float:
    maxima = _find_local_maxima(magnitude)
    peak_index = int(np.argmax(magnitude))
    sidelobe_peaks = [magnitude[index] for index in maxima if abs(index - peak_index) > 50]
    if not sidelobe_peaks:
        raise AssertionError("sidelobe peak not found")
    return float(20.0 * np.log10(max(sidelobe_peaks)))


def test_single_element_is_isotropic_ground_truth() -> None:
    result = _run_array_factor(num_elements=1, spacing_lambda=0.5, theta_steer_deg=0.0, phi_steer_deg=0.0)
    assert np.allclose(result["magnitude"], 1.0, atol=1e-12)
    assert np.allclose(result["magnitude_db"], 0.0, atol=1e-9)


def test_cpp_cut_matches_independent_numpy_reference_off_broadside() -> None:
    result = _run_array_factor(num_elements=12, spacing_lambda=0.5, theta_steer_deg=35.0, phi_steer_deg=0.0)
    actual_cut = result["magnitude"][:, PHI_ZERO_INDEX]
    expected_cut = _reference_cut(
        num_elements=12,
        spacing_lambda=0.5,
        theta_axis_deg=THETA_AXIS,
        theta_steer_deg=35.0,
        phi_steer_deg=0.0,
        taper_name="uniform",
    )
    assert np.allclose(actual_cut, expected_cut, atol=2e-6, rtol=1e-6)


def test_main_lobe_hits_requested_steering_angle_within_quarter_degree() -> None:
    theta_target_deg = 37.0
    result = _run_array_factor(num_elements=16, spacing_lambda=0.5, theta_steer_deg=theta_target_deg, phi_steer_deg=0.0)
    cut = result["magnitude"][:, PHI_ZERO_INDEX]
    peak_theta_deg = float(THETA_AXIS[int(np.argmax(cut))])
    assert abs(peak_theta_deg - theta_target_deg) <= 0.25


def test_half_power_beamwidth_matches_independent_reference() -> None:
    result = _run_array_factor(num_elements=16, spacing_lambda=0.5, theta_steer_deg=20.0, phi_steer_deg=0.0)
    actual_cut = result["magnitude"][:, PHI_ZERO_INDEX]
    expected_cut = _reference_cut(
        num_elements=16,
        spacing_lambda=0.5,
        theta_axis_deg=THETA_AXIS,
        theta_steer_deg=20.0,
        phi_steer_deg=0.0,
        taper_name="uniform",
    )

    actual_hpbw = _half_power_beamwidth(THETA_AXIS, actual_cut)
    expected_hpbw = _half_power_beamwidth(THETA_AXIS, expected_cut)
    assert abs(actual_hpbw - expected_hpbw) <= 0.1


def test_uniform_linear_array_first_sidelobe_level_is_about_minus_13_db() -> None:
    result = _run_array_factor(num_elements=16, spacing_lambda=0.5, theta_steer_deg=0.0, phi_steer_deg=0.0)
    cut = result["magnitude"][:, PHI_ZERO_INDEX]
    sidelobe_level_db = _first_sidelobe_level_db(cut)
    assert -13.8 <= sidelobe_level_db <= -12.0


def test_two_element_broadside_has_deep_endfire_null() -> None:
    result = _run_array_factor(num_elements=2, spacing_lambda=0.5, theta_steer_deg=0.0, phi_steer_deg=0.0)
    cut = result["magnitude"][:, PHI_ZERO_INDEX]
    endfire_index = int(np.argmin(np.abs(THETA_AXIS - 90.0)))
    assert cut[endfire_index] <= 1e-9


def test_spacing_above_lambda_over_2_creates_unit_strength_grating_lobe() -> None:
    result = _run_array_factor(num_elements=8, spacing_lambda=1.0, theta_steer_deg=0.0, phi_steer_deg=0.0)
    cut = result["magnitude"][:, PHI_ZERO_INDEX]
    broadside_index = int(np.argmin(np.abs(THETA_AXIS - 0.0)))
    grating_index = int(np.argmin(np.abs(THETA_AXIS - 90.0)))

    assert np.isclose(cut[broadside_index], 1.0, atol=1e-9)
    assert np.isclose(cut[grating_index], 1.0, atol=1e-9)


def test_hamming_taper_matches_closed_form_window() -> None:
    num_elements = 8
    expected = _reference_amplitude_taper(num_elements, "hamming")
    actual = amplitude_taper(num_elements, "hamming")
    assert np.allclose(actual, expected, atol=1e-12)


def test_hamming_reduces_peak_sidelobe_vs_uniform() -> None:
    uniform = _run_array_factor(num_elements=16, spacing_lambda=0.5, theta_steer_deg=0.0, phi_steer_deg=0.0, taper_name="uniform")
    hamming = _run_array_factor(num_elements=16, spacing_lambda=0.5, theta_steer_deg=0.0, phi_steer_deg=0.0, taper_name="hamming")

    uniform_sll_db = _first_sidelobe_level_db(uniform["magnitude"][:, PHI_ZERO_INDEX])
    hamming_sll_db = _first_sidelobe_level_db(hamming["magnitude"][:, PHI_ZERO_INDEX])

    assert hamming_sll_db < uniform_sll_db - 10.0


def test_planar_cpp_matches_independent_numpy_reference() -> None:
    result = array_factor_planar(
        num_x=4,
        num_y=5,
        spacing_x_lambda=0.5,
        spacing_y_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=28.0,
        phi_steer_deg=35.0,
        taper_x="uniform",
        taper_y="uniform",
    )
    expected = _reference_planar_pattern(
        num_x=4,
        num_y=5,
        spacing_x_lambda=0.5,
        spacing_y_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=28.0,
        phi_steer_deg=35.0,
    )
    assert np.allclose(result["magnitude"], expected, atol=2e-6, rtol=1e-6)


def test_planar_main_lobe_hits_requested_theta_and_phi() -> None:
    theta_target_deg = 32.0
    phi_target_deg = 40.0
    result = array_factor_planar(
        num_x=6,
        num_y=6,
        spacing_x_lambda=0.5,
        spacing_y_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=theta_target_deg,
        phi_steer_deg=phi_target_deg,
    )
    theta_index = int(np.argmin(np.abs(THETA_AXIS - theta_target_deg)))
    phi_index = int(np.argmin(np.abs(PHI_AXIS - phi_target_deg)))
    target_gain = float(result["magnitude"][theta_index, phi_index])
    assert np.isclose(target_gain, 1.0, atol=1e-6)


def test_planar_spacing_above_lambda_over_2_creates_grating_lobe() -> None:
    result = array_factor_planar(
        num_x=4,
        num_y=4,
        spacing_x_lambda=1.0,
        spacing_y_lambda=1.0,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=0.0,
        phi_steer_deg=0.0,
    )
    broadside_index = (int(np.argmin(np.abs(THETA_AXIS - 0.0))), PHI_ZERO_INDEX)
    grating_index = (int(np.argmin(np.abs(THETA_AXIS - 90.0))), PHI_ZERO_INDEX)
    assert np.isclose(result["magnitude"][broadside_index], 1.0, atol=1e-9)
    assert np.isclose(result["magnitude"][grating_index], 1.0, atol=1e-9)


def test_null_steering_creates_deep_null_while_preserving_desired_beam() -> None:
    null_theta_deg = 20.0
    positions = (np.arange(8, dtype=float) - 3.5) * 0.5
    adaptive_weights = null_steering_weights_linear(
        num_elements=8,
        spacing_lambda=0.5,
        theta_steer_deg=0.0,
        phi_steer_deg=0.0,
        null_thetas_deg=np.array([null_theta_deg]),
        null_phis_deg=np.array([0.0]),
        taper_name="uniform",
    )
    adaptive_weights = np.asarray(adaptive_weights)
    adaptive = array_factor_linear_from_weights(
        weights=adaptive_weights,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
    )
    baseline = array_factor_linear(
        num_elements=8,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=0.0,
        phi_steer_deg=0.0,
        taper_name="uniform",
    )

    cut_adaptive = adaptive["magnitude"][:, PHI_ZERO_INDEX]
    cut_baseline = baseline["magnitude"][:, PHI_ZERO_INDEX]
    null_window = np.where(np.abs(THETA_AXIS - null_theta_deg) <= 0.2)[0]
    adaptive_null_db = 20.0 * np.log10(max(float(np.min(cut_adaptive[null_window])), 1e-12))
    baseline_null_db = 20.0 * np.log10(max(float(np.min(cut_baseline[null_window])), 1e-12))

    desired_response = np.sum(adaptive_weights * np.exp(1j * 2.0 * np.pi * positions * 0.0))
    null_response = np.sum(adaptive_weights * np.exp(1j * 2.0 * np.pi * positions * np.sin(np.deg2rad(null_theta_deg))))

    assert np.isclose(desired_response, 1.0 + 0.0j, atol=1e-9)
    assert abs(null_response) <= 1e-9
    assert adaptive_null_db <= -50.0
    assert adaptive_null_db < baseline_null_db - 30.0
