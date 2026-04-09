from __future__ import annotations

import numpy as np


def estimate_covariance_matrix(snapshots: np.ndarray, diagonal_loading: float = 0.0) -> np.ndarray:
    x = np.asarray(snapshots, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError("snapshots must be a 2D array of shape (num_elements, num_snapshots)")
    if x.shape[1] < 2:
        raise ValueError("num_snapshots must be >= 2")
    r = (x @ x.conj().T) / x.shape[1]
    if diagonal_loading > 0.0:
        r = r + diagonal_loading * np.eye(r.shape[0], dtype=np.complex128)
    return r


def estimate_wideband_covariance_matrices(
    frequency_snapshots: np.ndarray,
    diagonal_loading: float = 0.0,
) -> np.ndarray:
    x = np.asarray(frequency_snapshots, dtype=np.complex128)
    if x.ndim != 3:
        raise ValueError("frequency_snapshots must have shape (num_frequency_bins, num_elements, num_snapshots)")
    return np.stack(
        [estimate_covariance_matrix(bin_snapshots, diagonal_loading=diagonal_loading) for bin_snapshots in x],
        axis=0,
    )


def _centered_positions(num_elements: int, spacing_lambda: float) -> np.ndarray:
    if num_elements < 1:
        raise ValueError("num_elements must be >= 1")
    if spacing_lambda <= 0.0:
        raise ValueError("spacing_lambda must be > 0")
    return (np.arange(num_elements, dtype=float) - (num_elements - 1) / 2.0) * spacing_lambda


def _planar_positions(
    num_x: int,
    num_y: int,
    spacing_x_lambda: float,
    spacing_y_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    if num_x < 1 or num_y < 1:
        raise ValueError("num_x and num_y must be >= 1")
    if spacing_x_lambda <= 0.0 or spacing_y_lambda <= 0.0:
        raise ValueError("spacing_x_lambda and spacing_y_lambda must be > 0")

    x_axis = _centered_positions(num_x, spacing_x_lambda)
    y_axis = _centered_positions(num_y, spacing_y_lambda)
    return np.tile(x_axis, num_y), np.repeat(y_axis, num_x)


def linear_steering_vector(
    num_elements: int,
    spacing_lambda: float,
    theta_deg: float,
    phi_deg: float = 0.0,
) -> np.ndarray:
    positions = _centered_positions(num_elements, spacing_lambda)
    u = np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg))
    return np.exp(1j * 2.0 * np.pi * positions * u)


def planar_steering_vector(
    num_x: int,
    num_y: int,
    spacing_x_lambda: float,
    spacing_y_lambda: float,
    theta_deg: float,
    phi_deg: float = 0.0,
) -> np.ndarray:
    x_flat, y_flat = _planar_positions(num_x, num_y, spacing_x_lambda, spacing_y_lambda)
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    ux = np.sin(theta_rad) * np.cos(phi_rad)
    uy = np.sin(theta_rad) * np.sin(phi_rad)
    return np.exp(1j * 2.0 * np.pi * (x_flat * ux + y_flat * uy))


def mimo_virtual_steering_vector_linear(
    num_tx: int,
    num_rx: int,
    tx_spacing_lambda: float,
    rx_spacing_lambda: float,
    theta_deg: float,
    phi_deg: float = 0.0,
) -> np.ndarray:
    tx = linear_steering_vector(num_tx, tx_spacing_lambda, theta_deg, phi_deg)
    rx = linear_steering_vector(num_rx, rx_spacing_lambda, theta_deg, phi_deg)
    return np.kron(tx, rx)


def polarimetric_steering_vector(
    spatial_steering_vector: np.ndarray,
    polarization_vector: np.ndarray,
) -> np.ndarray:
    spatial = np.asarray(spatial_steering_vector, dtype=np.complex128).reshape(-1)
    polarization = np.asarray(polarization_vector, dtype=np.complex128).reshape(-1)
    if spatial.size == 0:
        raise ValueError("spatial_steering_vector must be non-empty")
    if polarization.size == 0:
        raise ValueError("polarization_vector must be non-empty")
    return np.kron(spatial, polarization)


def wideband_linear_steering_vectors(
    num_elements: int,
    spacing_lambda: float,
    theta_deg: float,
    phi_deg: float,
    center_frequency_hz: float,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    if center_frequency_hz <= 0.0:
        raise ValueError("center_frequency_hz must be > 0")

    freqs = np.asarray(frequency_hz, dtype=float).reshape(-1)
    if np.any(freqs <= 0.0):
        raise ValueError("All frequencies must be > 0")

    positions = _centered_positions(num_elements, spacing_lambda)
    u = np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg))
    electrical_scale = (freqs / center_frequency_hz)[:, None]
    return np.exp(1j * 2.0 * np.pi * electrical_scale * positions[None, :] * u)


def mvdr_weights(
    covariance_matrix: np.ndarray,
    steering_vector: np.ndarray,
    diagonal_loading: float = 1e-3,
) -> np.ndarray:
    r = np.asarray(covariance_matrix, dtype=np.complex128)
    a = np.asarray(steering_vector, dtype=np.complex128).reshape(-1, 1)
    if r.shape[0] != r.shape[1]:
        raise ValueError("covariance_matrix must be square")
    if a.shape[0] != r.shape[0]:
        raise ValueError("steering_vector length must match covariance dimension")

    loaded = r + diagonal_loading * np.eye(r.shape[0], dtype=np.complex128)
    r_inv_a = np.linalg.solve(loaded, a)
    denom = a.conj().T @ r_inv_a
    return (r_inv_a / denom).reshape(-1)


def wideband_mvdr_weights(
    covariance_matrices: np.ndarray,
    steering_vectors: np.ndarray,
    diagonal_loading: float = 1e-3,
) -> np.ndarray:
    r = np.asarray(covariance_matrices, dtype=np.complex128)
    a = np.asarray(steering_vectors, dtype=np.complex128)
    if r.ndim != 3:
        raise ValueError("covariance_matrices must have shape (num_frequency_bins, num_elements, num_elements)")
    if a.ndim != 2:
        raise ValueError("steering_vectors must have shape (num_frequency_bins, num_elements)")
    if r.shape[0] != a.shape[0]:
        raise ValueError("Number of covariance matrices must match number of steering vectors")
    if r.shape[1] != r.shape[2] or r.shape[1] != a.shape[1]:
        raise ValueError("Each steering vector must match the covariance matrix dimension")

    return np.stack(
        [mvdr_weights(covariance, steer, diagonal_loading=diagonal_loading) for covariance, steer in zip(r, a)],
        axis=0,
    )


def beamform_frequency_snapshots(
    frequency_snapshots: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    x = np.asarray(frequency_snapshots, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.complex128)
    if x.ndim != 3:
        raise ValueError("frequency_snapshots must have shape (num_frequency_bins, num_elements, num_snapshots)")
    if w.ndim != 2:
        raise ValueError("weights must have shape (num_frequency_bins, num_elements)")
    if x.shape[:2] != w.shape:
        raise ValueError("weights must match the first two dimensions of frequency_snapshots")
    return np.einsum("fm,fmk->fk", np.conj(w), x)


def _prepare_adaptive_inputs(
    snapshots: np.ndarray,
    desired_signal: np.ndarray,
    initial_weights: np.ndarray | None,
    min_snapshots: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(snapshots, dtype=np.complex128)
    d = np.asarray(desired_signal, dtype=np.complex128).reshape(-1)
    if x.ndim != 2:
        raise ValueError("snapshots must have shape (num_elements, num_snapshots)")
    if x.shape[1] < min_snapshots:
        raise ValueError(f"num_snapshots must be >= {min_snapshots}")
    if d.size != x.shape[1]:
        raise ValueError("desired_signal length must match num_snapshots")

    if initial_weights is None:
        w = np.zeros(x.shape[0], dtype=np.complex128)
    else:
        w = np.asarray(initial_weights, dtype=np.complex128).reshape(-1)
        if w.size != x.shape[0]:
            raise ValueError("initial_weights length must match num_elements")
    return x, d, w


def _finalize_adaptive_result(
    weights: np.ndarray,
    output: np.ndarray,
    error: np.ndarray,
    weight_history: np.ndarray | None,
) -> dict[str, np.ndarray]:
    result = {
        "weights": np.asarray(weights, dtype=np.complex128),
        "output": np.asarray(output, dtype=np.complex128),
        "error": np.asarray(error, dtype=np.complex128),
    }
    if weight_history is not None:
        result["weight_history"] = np.asarray(weight_history, dtype=np.complex128)
    return result


def lms_weights(
    snapshots: np.ndarray,
    desired_signal: np.ndarray,
    step_size: float = 0.05,
    leakage: float = 0.0,
    initial_weights: np.ndarray | None = None,
    return_history: bool = False,
) -> dict[str, np.ndarray]:
    if step_size <= 0.0:
        raise ValueError("step_size must be > 0")
    if leakage < 0.0:
        raise ValueError("leakage must be >= 0")

    x, d, w = _prepare_adaptive_inputs(snapshots, desired_signal, initial_weights)
    output = np.zeros(x.shape[1], dtype=np.complex128)
    error = np.zeros_like(output)
    history = np.zeros((x.shape[1], x.shape[0]), dtype=np.complex128) if return_history else None

    for idx in range(x.shape[1]):
        x_k = x[:, idx]
        output[idx] = np.vdot(w, x_k)
        error[idx] = d[idx] - output[idx]
        w = (1.0 - step_size * leakage) * w + step_size * x_k * np.conj(error[idx])
        if history is not None:
            history[idx] = w

    return _finalize_adaptive_result(w, output, error, history)


def nlms_weights(
    snapshots: np.ndarray,
    desired_signal: np.ndarray,
    step_size: float = 0.5,
    epsilon: float = 1e-6,
    leakage: float = 0.0,
    initial_weights: np.ndarray | None = None,
    return_history: bool = False,
) -> dict[str, np.ndarray]:
    if step_size <= 0.0:
        raise ValueError("step_size must be > 0")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    if leakage < 0.0:
        raise ValueError("leakage must be >= 0")

    x, d, w = _prepare_adaptive_inputs(snapshots, desired_signal, initial_weights)
    output = np.zeros(x.shape[1], dtype=np.complex128)
    error = np.zeros_like(output)
    history = np.zeros((x.shape[1], x.shape[0]), dtype=np.complex128) if return_history else None

    for idx in range(x.shape[1]):
        x_k = x[:, idx]
        output[idx] = np.vdot(w, x_k)
        error[idx] = d[idx] - output[idx]
        power = float(np.vdot(x_k, x_k).real)
        mu = step_size / (power + epsilon)
        w = (1.0 - mu * leakage) * w + mu * x_k * np.conj(error[idx])
        if history is not None:
            history[idx] = w

    return _finalize_adaptive_result(w, output, error, history)


def rls_weights(
    snapshots: np.ndarray,
    desired_signal: np.ndarray,
    forgetting_factor: float = 0.995,
    initialization_delta: float = 1.0,
    initial_weights: np.ndarray | None = None,
    return_history: bool = False,
) -> dict[str, np.ndarray]:
    if not (0.0 < forgetting_factor <= 1.0):
        raise ValueError("forgetting_factor must be in (0, 1]")
    if initialization_delta <= 0.0:
        raise ValueError("initialization_delta must be > 0")

    x, d, w = _prepare_adaptive_inputs(snapshots, desired_signal, initial_weights)
    output = np.zeros(x.shape[1], dtype=np.complex128)
    error = np.zeros_like(output)
    history = np.zeros((x.shape[1], x.shape[0]), dtype=np.complex128) if return_history else None
    p = np.eye(x.shape[0], dtype=np.complex128) / initialization_delta

    for idx in range(x.shape[1]):
        x_k = x[:, idx].reshape(-1, 1)
        output[idx] = np.vdot(w, x_k[:, 0])
        error[idx] = d[idx] - output[idx]

        gain_numer = p @ x_k
        gain_denom = forgetting_factor + (x_k.conj().T @ gain_numer)[0, 0]
        gain = gain_numer / gain_denom

        w = w + gain[:, 0] * np.conj(error[idx])
        p = (p - gain @ x_k.conj().T @ p) / forgetting_factor
        if history is not None:
            history[idx] = w

    return _finalize_adaptive_result(w, output, error, history)


def music_spectrum(
    covariance_matrix: np.ndarray,
    scan_manifold: np.ndarray,
    num_sources: int,
) -> np.ndarray:
    r = np.asarray(covariance_matrix, dtype=np.complex128)
    a_scan = np.asarray(scan_manifold, dtype=np.complex128)
    if r.shape[0] != r.shape[1]:
        raise ValueError("covariance_matrix must be square")
    if a_scan.ndim != 2 or a_scan.shape[0] != r.shape[0]:
        raise ValueError("scan_manifold must have shape (num_elements, num_scan_points)")
    if not (1 <= num_sources < r.shape[0]):
        raise ValueError("num_sources must be in [1, num_elements-1]")

    eigvals, eigvecs = np.linalg.eigh(r)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    noise_subspace = eigvecs[:, : r.shape[0] - num_sources]

    numerator = np.sum(np.abs(a_scan) ** 2, axis=0)
    projection = noise_subspace.conj().T @ a_scan
    denominator = np.sum(np.abs(projection) ** 2, axis=0)
    return np.real(numerator / np.maximum(denominator, 1e-15))


def doa_music_linear(
    snapshots: np.ndarray,
    spacing_lambda: float,
    theta_scan_deg: np.ndarray,
    num_sources: int = 1,
    phi_deg: float = 0.0,
    diagonal_loading: float = 1e-3,
) -> dict[str, np.ndarray]:
    x = np.asarray(snapshots, dtype=np.complex128)
    theta_scan = np.asarray(theta_scan_deg, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("snapshots must be 2D")

    num_elements = x.shape[0]
    scan_vectors = np.stack(
        [linear_steering_vector(num_elements, spacing_lambda, theta, phi_deg) for theta in theta_scan],
        axis=1,
    )
    covariance = estimate_covariance_matrix(x, diagonal_loading=diagonal_loading)
    spectrum = music_spectrum(covariance, scan_vectors, num_sources=num_sources)

    peak_indices = np.argsort(spectrum)[-num_sources:]
    peak_indices.sort()
    return {
        "theta_scan_deg": theta_scan,
        "spectrum": spectrum,
        "estimated_thetas_deg": theta_scan[peak_indices],
        "covariance_matrix": covariance,
    }
