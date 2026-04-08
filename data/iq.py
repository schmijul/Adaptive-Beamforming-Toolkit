from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithms.adaptive import linear_steering_vector


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


def simulate_array_iq(
    num_elements: int,
    num_snapshots: int,
    spacing_lambda: float,
    source_thetas_deg: np.ndarray,
    source_phis_deg: np.ndarray,
    source_snr_db: np.ndarray,
    random_seed: int = 0,
) -> np.ndarray:
    thetas = np.asarray(source_thetas_deg, dtype=float).reshape(-1)
    phis = np.asarray(source_phis_deg, dtype=float).reshape(-1)
    snr_db = np.asarray(source_snr_db, dtype=float).reshape(-1)
    if not (thetas.size == phis.size == snr_db.size):
        raise ValueError("source_thetas_deg, source_phis_deg, and source_snr_db must have equal length")

    rng = np.random.default_rng(random_seed)
    snapshots = np.zeros((num_elements, num_snapshots), dtype=np.complex128)
    for theta, phi, snr in zip(thetas, phis, snr_db):
        steering = linear_steering_vector(num_elements, spacing_lambda, float(theta), float(phi)).reshape(-1, 1)
        signal = (
            rng.normal(0.0, 1.0, num_snapshots) + 1j * rng.normal(0.0, 1.0, num_snapshots)
        ) / np.sqrt(2.0)
        amplitude = 10.0 ** (snr / 20.0)
        snapshots += amplitude * steering @ signal[None, :]

    noise = (
        rng.normal(0.0, 1.0, (num_elements, num_snapshots))
        + 1j * rng.normal(0.0, 1.0, (num_elements, num_snapshots))
    ) / np.sqrt(2.0)
    return snapshots + noise


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
