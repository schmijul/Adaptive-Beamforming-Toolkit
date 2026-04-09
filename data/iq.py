from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.adaptive import (
    linear_steering_vector,
    mimo_virtual_steering_vector_linear,
    planar_steering_vector,
    polarimetric_steering_vector,
)


def load_iq_samples(path: str | Path) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".npy":
        data = np.load(source)
        return np.asarray(data, dtype=np.complex128).reshape(-1)

    if suffix == ".npz":
        container = np.load(source)
        if "iq" in container:
            return np.asarray(container["iq"], dtype=np.complex128).reshape(-1)
        first_key = next(iter(container.keys()))
        return np.asarray(container[first_key], dtype=np.complex128).reshape(-1)

    if suffix in {".csv", ".txt"}:
        values = np.loadtxt(source, delimiter="," if suffix == ".csv" else None)
        values = np.atleast_2d(values)
        if values.shape[1] < 2:
            raise ValueError("Text IQ files must contain at least two columns: I,Q")
        return values[:, 0].astype(np.float64) + 1j * values[:, 1].astype(np.float64)

    raise ValueError(f"Unsupported IQ file extension: {suffix}")


def _validate_source_vectors(
    source_thetas_deg: np.ndarray,
    source_phis_deg: np.ndarray,
    source_snr_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thetas = np.asarray(source_thetas_deg, dtype=float).reshape(-1)
    phis = np.asarray(source_phis_deg, dtype=float).reshape(-1)
    snr_db = np.asarray(source_snr_db, dtype=float).reshape(-1)
    if not (thetas.size == phis.size == snr_db.size):
        raise ValueError("source_thetas_deg, source_phis_deg, and source_snr_db must have equal length")
    return thetas, phis, snr_db


def _simulate_from_steering_vectors(
    steering_vectors: np.ndarray,
    num_snapshots: int,
    source_snr_db: np.ndarray,
    random_seed: int,
) -> dict[str, np.ndarray]:
    manifold = np.asarray(steering_vectors, dtype=np.complex128)
    snr_db = np.asarray(source_snr_db, dtype=float).reshape(-1)
    if manifold.ndim != 2:
        raise ValueError("steering_vectors must have shape (num_sources, num_channels)")
    if manifold.shape[0] != snr_db.size:
        raise ValueError("steering_vectors must match source_snr_db length")
    if num_snapshots < 1:
        raise ValueError("num_snapshots must be >= 1")

    rng = np.random.default_rng(random_seed)
    source_signals = np.zeros((manifold.shape[0], num_snapshots), dtype=np.complex128)
    snapshots = np.zeros((manifold.shape[1], num_snapshots), dtype=np.complex128)

    for idx, snr in enumerate(snr_db):
        signal = (
            rng.normal(0.0, 1.0, num_snapshots) + 1j * rng.normal(0.0, 1.0, num_snapshots)
        ) / np.sqrt(2.0)
        scaled_signal = (10.0 ** (snr / 20.0)) * signal
        source_signals[idx] = scaled_signal
        snapshots += manifold[idx][:, None] @ scaled_signal[None, :]

    noise = (
        rng.normal(0.0, 1.0, (manifold.shape[1], num_snapshots))
        + 1j * rng.normal(0.0, 1.0, (manifold.shape[1], num_snapshots))
    ) / np.sqrt(2.0)
    return {
        "snapshots": snapshots + noise,
        "source_signals": source_signals,
        "steering_vectors": manifold,
        "noise": noise,
    }


def _array_steering_matrix(
    source_thetas_deg: np.ndarray,
    source_phis_deg: np.ndarray,
    geometry: str,
    num_elements: int | None,
    spacing_lambda: float | None,
    num_x: int | None,
    num_y: int | None,
    spacing_x_lambda: float | None,
    spacing_y_lambda: float | None,
) -> np.ndarray:
    geometry_name = str(geometry).lower()
    if geometry_name == "ula":
        if num_elements is None or spacing_lambda is None:
            raise ValueError("ULA simulation requires num_elements and spacing_lambda")
        return np.stack(
            [linear_steering_vector(num_elements, spacing_lambda, theta, phi) for theta, phi in zip(source_thetas_deg, source_phis_deg)],
            axis=0,
        )
    if geometry_name in {"planar", "upa"}:
        if num_x is None or num_y is None or spacing_x_lambda is None or spacing_y_lambda is None:
            raise ValueError("Planar simulation requires num_x, num_y, spacing_x_lambda, and spacing_y_lambda")
        return np.stack(
            [
                planar_steering_vector(num_x, num_y, spacing_x_lambda, spacing_y_lambda, theta, phi)
                for theta, phi in zip(source_thetas_deg, source_phis_deg)
            ],
            axis=0,
        )
    raise ValueError(f"Unsupported geometry: {geometry}")


def simulate_array_iq_components(
    num_elements: int | None,
    num_snapshots: int,
    spacing_lambda: float | None,
    source_thetas_deg: np.ndarray,
    source_phis_deg: np.ndarray,
    source_snr_db: np.ndarray,
    random_seed: int = 0,
    geometry: str = "ula",
    num_x: int | None = None,
    num_y: int | None = None,
    spacing_x_lambda: float | None = None,
    spacing_y_lambda: float | None = None,
) -> dict[str, np.ndarray]:
    thetas, phis, snr_db = _validate_source_vectors(source_thetas_deg, source_phis_deg, source_snr_db)
    manifold = _array_steering_matrix(
        source_thetas_deg=thetas,
        source_phis_deg=phis,
        geometry=geometry,
        num_elements=num_elements,
        spacing_lambda=spacing_lambda,
        num_x=num_x,
        num_y=num_y,
        spacing_x_lambda=spacing_x_lambda,
        spacing_y_lambda=spacing_y_lambda,
    )
    return _simulate_from_steering_vectors(manifold, num_snapshots=num_snapshots, source_snr_db=snr_db, random_seed=random_seed)


def simulate_array_iq(
    num_elements: int | None,
    num_snapshots: int,
    spacing_lambda: float | None,
    source_thetas_deg: np.ndarray,
    source_phis_deg: np.ndarray,
    source_snr_db: np.ndarray,
    random_seed: int = 0,
    geometry: str = "ula",
    num_x: int | None = None,
    num_y: int | None = None,
    spacing_x_lambda: float | None = None,
    spacing_y_lambda: float | None = None,
) -> np.ndarray:
    components = simulate_array_iq_components(
        num_elements=num_elements,
        num_snapshots=num_snapshots,
        spacing_lambda=spacing_lambda,
        source_thetas_deg=source_thetas_deg,
        source_phis_deg=source_phis_deg,
        source_snr_db=source_snr_db,
        random_seed=random_seed,
        geometry=geometry,
        num_x=num_x,
        num_y=num_y,
        spacing_x_lambda=spacing_x_lambda,
        spacing_y_lambda=spacing_y_lambda,
    )
    return components["snapshots"]


def simulate_mimo_iq(
    num_tx: int,
    num_rx: int,
    num_snapshots: int,
    tx_spacing_lambda: float,
    rx_spacing_lambda: float,
    source_thetas_deg: np.ndarray,
    source_phis_deg: np.ndarray,
    source_snr_db: np.ndarray,
    random_seed: int = 0,
) -> np.ndarray:
    thetas, phis, snr_db = _validate_source_vectors(source_thetas_deg, source_phis_deg, source_snr_db)
    manifold = np.stack(
        [
            mimo_virtual_steering_vector_linear(num_tx, num_rx, tx_spacing_lambda, rx_spacing_lambda, theta, phi)
            for theta, phi in zip(thetas, phis)
        ],
        axis=0,
    )
    return _simulate_from_steering_vectors(manifold, num_snapshots=num_snapshots, source_snr_db=snr_db, random_seed=random_seed)[
        "snapshots"
    ]


def simulate_polarimetric_array_iq(
    num_elements: int,
    num_snapshots: int,
    spacing_lambda: float,
    source_thetas_deg: np.ndarray,
    source_phis_deg: np.ndarray,
    source_snr_db: np.ndarray,
    source_polarizations: np.ndarray,
    random_seed: int = 0,
) -> np.ndarray:
    thetas, phis, snr_db = _validate_source_vectors(source_thetas_deg, source_phis_deg, source_snr_db)
    polarizations = np.asarray(source_polarizations, dtype=np.complex128)
    if polarizations.ndim != 2:
        raise ValueError("source_polarizations must have shape (num_sources, num_polarization_channels)")
    if polarizations.shape[0] != thetas.size:
        raise ValueError("source_polarizations must match the number of sources")

    manifold = np.stack(
        [
            polarimetric_steering_vector(
                linear_steering_vector(num_elements, spacing_lambda, theta, phi),
                polarizations[idx],
            )
            for idx, (theta, phi) in enumerate(zip(thetas, phis))
        ],
        axis=0,
    )
    return _simulate_from_steering_vectors(manifold, num_snapshots=num_snapshots, source_snr_db=snr_db, random_seed=random_seed)[
        "snapshots"
    ]


def beamform_iq(iq_snapshots: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(iq_snapshots, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.complex128).reshape(-1)
    if x.ndim != 2:
        raise ValueError("iq_snapshots must have shape (num_elements, num_snapshots)")
    if x.shape[0] != w.size:
        raise ValueError("weights length must match iq_snapshots num_elements")
    return np.conj(w) @ x


def compare_sim_vs_measurement(simulated: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    sim = np.asarray(simulated, dtype=np.complex128).reshape(-1)
    meas = np.asarray(measured, dtype=np.complex128).reshape(-1)
    if sim.size != meas.size:
        raise ValueError("simulated and measured vectors must have the same length")

    error = sim - meas
    mse = float(np.mean(np.abs(error) ** 2))
    ref_power = float(np.mean(np.abs(meas) ** 2) + 1e-15)
    nmse = mse / ref_power
    correlation = float(np.abs(np.vdot(sim, meas)) / (np.linalg.norm(sim) * np.linalg.norm(meas) + 1e-15))
    return {
        "mse": mse,
        "nmse_db": float(10.0 * np.log10(max(nmse, 1e-15))),
        "correlation": correlation,
    }
