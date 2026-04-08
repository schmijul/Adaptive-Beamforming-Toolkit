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


def linear_steering_vector(
    num_elements: int,
    spacing_lambda: float,
    theta_deg: float,
    phi_deg: float = 0.0,
) -> np.ndarray:
    positions = (np.arange(num_elements, dtype=float) - (num_elements - 1) / 2.0) * spacing_lambda
    u = np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg))
    return np.exp(1j * 2.0 * np.pi * positions * u)


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
