from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.beamforming import amplitude_taper, array_factor_linear


def _direction_cosines(theta_deg: np.ndarray, phi_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    ux = np.sin(theta_rad) * np.cos(phi_rad)
    uz = np.cos(theta_rad)
    return ux, uz


def _centered_positions(num_elements: int, spacing_lambda: float) -> np.ndarray:
    return (np.arange(num_elements, dtype=float) - (num_elements - 1) / 2.0) * spacing_lambda


def steering_weights_near_field_linear(
    num_elements: int,
    spacing_lambda: float,
    theta_focus_deg: float,
    phi_focus_deg: float,
    focus_range_lambda: float,
    taper_name: str = "uniform",
) -> np.ndarray:
    if focus_range_lambda <= 0.0:
        raise ValueError("focus_range_lambda must be > 0")

    positions = _centered_positions(num_elements, spacing_lambda)
    ux_focus, uz_focus = _direction_cosines(np.array(theta_focus_deg), np.array(phi_focus_deg))
    x_focus = float(focus_range_lambda * ux_focus)
    z_focus = float(focus_range_lambda * uz_focus)
    distances = np.sqrt((x_focus - positions) ** 2 + z_focus**2)
    distances -= distances.min()
    taper = np.asarray(amplitude_taper(num_elements, taper_name), dtype=float)
    return taper * np.exp(-1j * 2.0 * np.pi * distances)


def array_factor_linear_near_field(
    num_elements: int,
    spacing_lambda: float,
    theta_grid_deg: np.ndarray,
    phi_grid_deg: np.ndarray,
    theta_focus_deg: float,
    phi_focus_deg: float,
    focus_range_lambda: float,
    eval_range_lambda: float | None = None,
    taper_name: str = "uniform",
) -> dict[str, np.ndarray]:
    if theta_grid_deg.shape != phi_grid_deg.shape:
        raise ValueError("theta_grid_deg and phi_grid_deg must have the same shape")
    if focus_range_lambda <= 0.0:
        raise ValueError("focus_range_lambda must be > 0")

    positions = _centered_positions(num_elements, spacing_lambda)
    weights = steering_weights_near_field_linear(
        num_elements=num_elements,
        spacing_lambda=spacing_lambda,
        theta_focus_deg=theta_focus_deg,
        phi_focus_deg=phi_focus_deg,
        focus_range_lambda=focus_range_lambda,
        taper_name=taper_name,
    )

    eval_range = focus_range_lambda if eval_range_lambda is None else eval_range_lambda
    if eval_range <= 0.0:
        raise ValueError("eval_range_lambda must be > 0")

    ux, uz = _direction_cosines(theta_grid_deg, phi_grid_deg)
    x_eval = eval_range * ux
    z_eval = eval_range * uz

    response = np.zeros(theta_grid_deg.shape, dtype=np.complex128)
    for idx, x_element in enumerate(positions):
        distances = np.sqrt((x_eval - x_element) ** 2 + z_eval**2)
        response += weights[idx] * np.exp(1j * 2.0 * np.pi * distances)

    magnitude = np.abs(response)
    magnitude /= magnitude.max() + 1e-12
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    return {
        "response": response,
        "magnitude": magnitude,
        "magnitude_db": magnitude_db,
        "positions_lambda": positions,
        "weights": weights,
    }


def array_factor_linear_field_mode(
    num_elements: int,
    spacing_lambda: float,
    theta_grid_deg: np.ndarray,
    phi_grid_deg: np.ndarray,
    theta_steer_deg: float,
    phi_steer_deg: float,
    field_mode: str = "far",
    focus_range_lambda: float = 50.0,
    taper_name: str = "uniform",
) -> dict[str, np.ndarray]:
    if field_mode == "far":
        return array_factor_linear(
            num_elements=num_elements,
            spacing_lambda=spacing_lambda,
            theta_grid_deg=theta_grid_deg,
            phi_grid_deg=phi_grid_deg,
            theta_steer_deg=theta_steer_deg,
            phi_steer_deg=phi_steer_deg,
            taper_name=taper_name,
        )
    if field_mode == "near":
        return array_factor_linear_near_field(
            num_elements=num_elements,
            spacing_lambda=spacing_lambda,
            theta_grid_deg=theta_grid_deg,
            phi_grid_deg=phi_grid_deg,
            theta_focus_deg=theta_steer_deg,
            phi_focus_deg=phi_steer_deg,
            focus_range_lambda=focus_range_lambda,
            taper_name=taper_name,
        )
    raise ValueError(f"Unsupported field_mode: {field_mode}")


def _quantize_angles(angles_rad: np.ndarray, phase_bits: int | None) -> np.ndarray:
    if phase_bits is None:
        return angles_rad
    levels = 2**phase_bits
    wrapped = (angles_rad + np.pi) % (2.0 * np.pi) - np.pi
    step = 2.0 * np.pi / levels
    return np.round(wrapped / step) * step


def _quantize_amplitude(amplitude: np.ndarray, amplitude_bits: int | None) -> np.ndarray:
    if amplitude_bits is None:
        return amplitude
    levels = max(2**amplitude_bits - 1, 1)
    clipped = np.clip(amplitude, 0.0, 1.0)
    return np.round(clipped * levels) / levels


@dataclass(frozen=True)
class ArchitectureWeights:
    architecture: str
    weights: np.ndarray
    rf_weights: np.ndarray
    digital_weights: np.ndarray


def synthesize_beamforming_architecture(
    ideal_weights: np.ndarray,
    architecture: str = "digital",
    num_rf_chains: int = 1,
    phase_bits: int | None = None,
    amplitude_bits: int | None = None,
) -> ArchitectureWeights:
    if num_rf_chains < 1:
        raise ValueError("num_rf_chains must be >= 1")

    ideal = np.asarray(ideal_weights, dtype=np.complex128).reshape(-1)
    if ideal.size == 0:
        raise ValueError("ideal_weights must be non-empty")

    if architecture == "digital":
        amplitudes = _quantize_amplitude(np.abs(ideal), amplitude_bits)
        phases = _quantize_angles(np.angle(ideal), phase_bits)
        digital = amplitudes * np.exp(1j * phases)
        rf = np.eye(ideal.size, dtype=np.complex128)
        return ArchitectureWeights("digital", digital, rf, digital[:, None])

    rf_vector = np.exp(1j * _quantize_angles(np.angle(ideal), phase_bits)) / np.sqrt(ideal.size)
    bb_scalar = np.vdot(rf_vector, ideal)

    if architecture == "analog":
        analog = np.exp(1j * _quantize_angles(np.angle(ideal), phase_bits)) / np.sqrt(ideal.size)
        return ArchitectureWeights("analog", analog, analog[:, None], np.array([[1.0 + 0.0j]]))

    if architecture == "hybrid":
        chain_count = min(num_rf_chains, ideal.size)
        rf = np.zeros((ideal.size, chain_count), dtype=np.complex128)
        rf[:, 0] = rf_vector
        for chain in range(1, chain_count):
            phase_offset = 2.0 * np.pi * chain / chain_count
            rf[:, chain] = np.exp(1j * (np.angle(rf_vector) + phase_offset)) / np.sqrt(ideal.size)
        digital = np.zeros((chain_count, 1), dtype=np.complex128)
        digital[0, 0] = bb_scalar
        hybrid = (rf @ digital).reshape(-1)
        return ArchitectureWeights("hybrid", hybrid, rf, digital)

    raise ValueError(f"Unsupported architecture: {architecture}")


def wideband_array_factor_linear(
    num_elements: int,
    spacing_lambda: float,
    theta_grid_deg: np.ndarray,
    phi_grid_deg: np.ndarray,
    theta_steer_deg: float,
    phi_steer_deg: float,
    center_frequency_hz: float,
    frequency_hz: np.ndarray,
    taper_name: str = "uniform",
) -> dict[str, np.ndarray]:
    if theta_grid_deg.shape != phi_grid_deg.shape:
        raise ValueError("theta_grid_deg and phi_grid_deg must have the same shape")
    if center_frequency_hz <= 0.0:
        raise ValueError("center_frequency_hz must be > 0")

    freqs = np.asarray(frequency_hz, dtype=float).reshape(-1)
    if np.any(freqs <= 0.0):
        raise ValueError("All frequencies must be > 0")

    positions = _centered_positions(num_elements, spacing_lambda)
    taper = np.asarray(amplitude_taper(num_elements, taper_name), dtype=float)
    theta0 = np.deg2rad(theta_steer_deg)
    phi0 = np.deg2rad(phi_steer_deg)
    u0 = np.sin(theta0) * np.cos(phi0)
    fixed_weights = taper * np.exp(-1j * 2.0 * np.pi * positions * u0)

    theta_rad = np.deg2rad(theta_grid_deg)
    phi_rad = np.deg2rad(phi_grid_deg)
    u = np.sin(theta_rad) * np.cos(phi_rad)

    response = np.zeros((freqs.size, *theta_grid_deg.shape), dtype=np.complex128)
    magnitude = np.zeros_like(response.real)
    for idx, freq in enumerate(freqs):
        electrical_spacing = spacing_lambda * (freq / center_frequency_hz)
        for n, x_pos in enumerate(positions):
            response[idx] += fixed_weights[n] * np.exp(1j * 2.0 * np.pi * electrical_spacing * x_pos * u / spacing_lambda)
        mag = np.abs(response[idx])
        magnitude[idx] = mag / (mag.max() + 1e-12)

    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    return {
        "frequency_hz": freqs,
        "response": response,
        "magnitude": magnitude,
        "magnitude_db": magnitude_db,
        "weights_center_frequency": fixed_weights,
    }


def build_mutual_coupling_matrix(
    num_elements: int,
    nearest_neighbor_db: float = -20.0,
    phase_deg: float = -35.0,
) -> np.ndarray:
    if num_elements < 1:
        raise ValueError("num_elements must be >= 1")
    coupling_mag = 10.0 ** (nearest_neighbor_db / 20.0)
    phase = np.deg2rad(phase_deg)
    alpha = coupling_mag * np.exp(1j * phase)
    distances = np.abs(np.arange(num_elements)[:, None] - np.arange(num_elements)[None, :])
    return alpha**distances


def element_pattern_gain(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    pattern: str = "isotropic",
    exponent: float = 1.0,
) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    if pattern == "isotropic":
        return np.ones_like(theta_rad, dtype=float)
    if pattern == "cosine":
        return np.clip(np.cos(theta_rad), 0.0, None) ** exponent
    if pattern == "cardioid":
        return 0.5 * (1.0 + np.clip(np.cos(theta_rad), -1.0, 1.0))
    raise ValueError(f"Unsupported pattern: {pattern}")


def array_factor_linear_with_impairments(
    num_elements: int,
    spacing_lambda: float,
    theta_grid_deg: np.ndarray,
    phi_grid_deg: np.ndarray,
    theta_steer_deg: float,
    phi_steer_deg: float,
    taper_name: str = "uniform",
    element_pattern_name: str = "isotropic",
    element_pattern_exponent: float = 1.0,
    coupling_matrix: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    base = array_factor_linear(
        num_elements=num_elements,
        spacing_lambda=spacing_lambda,
        theta_grid_deg=theta_grid_deg,
        phi_grid_deg=phi_grid_deg,
        theta_steer_deg=theta_steer_deg,
        phi_steer_deg=phi_steer_deg,
        taper_name=taper_name,
    )
    weights = np.asarray(base["weights"], dtype=np.complex128)
    if coupling_matrix is None:
        coupling_matrix = np.eye(num_elements, dtype=np.complex128)
    coupling = np.asarray(coupling_matrix, dtype=np.complex128)
    if coupling.shape != (num_elements, num_elements):
        raise ValueError("coupling_matrix shape must be (num_elements, num_elements)")
    effective_weights = coupling @ weights

    positions = _centered_positions(num_elements, spacing_lambda)
    ux, _ = _direction_cosines(theta_grid_deg, phi_grid_deg)
    response = np.zeros(theta_grid_deg.shape, dtype=np.complex128)
    for idx, x_pos in enumerate(positions):
        response += effective_weights[idx] * np.exp(1j * 2.0 * np.pi * x_pos * ux)

    pattern_gain = element_pattern_gain(
        theta_deg=theta_grid_deg,
        phi_deg=phi_grid_deg,
        pattern=element_pattern_name,
        exponent=element_pattern_exponent,
    )
    response *= pattern_gain
    magnitude = np.abs(response)
    magnitude /= magnitude.max() + 1e-12
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    return {
        "response": response,
        "magnitude": magnitude,
        "magnitude_db": magnitude_db,
        "pattern_gain": pattern_gain,
        "weights": effective_weights,
        "positions_lambda": positions,
    }
