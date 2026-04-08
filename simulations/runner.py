from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from algorithms.adaptive import estimate_covariance_matrix, linear_steering_vector, mvdr_weights
from core.beamforming import array_factor_linear_from_weights
from data.iq import simulate_array_iq
from simulations.config import ScenarioConfig
from visualize.plots import build_elevation_cut, build_heatmap, build_pattern_3d, build_weights_plot


def _build_grids(config: ScenarioConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(config.sweep.theta_start_deg, config.sweep.theta_stop_deg, config.sweep.theta_num)
    phi = np.linspace(config.sweep.phi_start_deg, config.sweep.phi_stop_deg, config.sweep.phi_num)
    theta_grid = theta[:, None] * np.ones((1, phi.size))
    phi_grid = np.ones((theta.size, 1)) * phi[None, :]
    return theta, phi, theta_grid, phi_grid


def _select_weights(config: ScenarioConfig, snapshots: np.ndarray) -> np.ndarray:
    steer = linear_steering_vector(
        num_elements=config.array.num_elements,
        spacing_lambda=config.array.spacing_lambda,
        theta_deg=config.desired_source.theta_deg,
        phi_deg=config.desired_source.phi_deg,
    )

    if config.algorithm.name == "conventional":
        denom = np.vdot(steer, steer)
        return steer / denom

    covariance = estimate_covariance_matrix(snapshots, diagonal_loading=0.0)
    return mvdr_weights(
        covariance_matrix=covariance,
        steering_vector=steer,
        diagonal_loading=config.algorithm.diagonal_loading,
    )


def _calculate_sinr_db(config: ScenarioConfig, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.complex128).reshape(-1)

    desired_vec = linear_steering_vector(
        num_elements=config.array.num_elements,
        spacing_lambda=config.array.spacing_lambda,
        theta_deg=config.desired_source.theta_deg,
        phi_deg=config.desired_source.phi_deg,
    )
    desired_power = 10.0 ** (config.desired_source.snr_db / 10.0)
    signal = (np.abs(np.vdot(w, desired_vec)) ** 2) * desired_power

    interference = 0.0
    for src in config.interference_sources:
        interferer_vec = linear_steering_vector(
            num_elements=config.array.num_elements,
            spacing_lambda=config.array.spacing_lambda,
            theta_deg=src.theta_deg,
            phi_deg=src.phi_deg,
        )
        interference += (np.abs(np.vdot(w, interferer_vec)) ** 2) * (10.0 ** (src.snr_db / 10.0))

    noise = float(np.real(np.vdot(w, w)))
    return float(10.0 * np.log10(max(signal / (interference + noise + 1e-15), 1e-15)))


def _run_once(config: ScenarioConfig, seed: int) -> dict[str, object]:
    source_thetas = np.array([config.desired_source.theta_deg, *[src.theta_deg for src in config.interference_sources]], dtype=float)
    source_phis = np.array([config.desired_source.phi_deg, *[src.phi_deg for src in config.interference_sources]], dtype=float)
    source_snr = np.array([config.desired_source.snr_db, *[src.snr_db for src in config.interference_sources]], dtype=float)

    snapshots = simulate_array_iq(
        num_elements=config.array.num_elements,
        num_snapshots=config.snapshots,
        spacing_lambda=config.array.spacing_lambda,
        source_thetas_deg=source_thetas,
        source_phis_deg=source_phis,
        source_snr_db=source_snr,
        random_seed=seed,
    )
    weights = _select_weights(config, snapshots)

    theta, phi, theta_grid, phi_grid = _build_grids(config)
    pattern = array_factor_linear_from_weights(
        weights=weights,
        spacing_lambda=config.array.spacing_lambda,
        theta_grid_deg=theta_grid,
        phi_grid_deg=phi_grid,
    )
    sinr_db = _calculate_sinr_db(config, weights)

    phi0_idx = int(np.argmin(np.abs(phi)))
    cut_db = pattern["magnitude_db"][:, phi0_idx]

    return {
        "seed": seed,
        "sinr_db": sinr_db,
        "theta_deg": theta.tolist(),
        "phi_deg": phi.tolist(),
        "magnitude_db": np.asarray(pattern["magnitude_db"], dtype=float).tolist(),
        "elevation_cut_db": np.asarray(cut_db, dtype=float).tolist(),
        "weights_real": np.asarray(np.real(weights), dtype=float).tolist(),
        "weights_imag": np.asarray(np.imag(weights), dtype=float).tolist(),
        "weight_amplitudes": np.asarray(np.abs(weights), dtype=float).tolist(),
        "weight_phases_deg": np.asarray(np.rad2deg(np.angle(weights)), dtype=float).tolist(),
        "positions_lambda": np.asarray(pattern["positions_lambda"], dtype=float).tolist(),
    }


def _write_plots(config: ScenarioConfig, result: dict[str, object], output_dir: Path) -> None:
    theta = np.asarray(result["theta_deg"], dtype=float)
    phi = np.asarray(result["phi_deg"], dtype=float)
    magnitude_db = np.asarray(result["magnitude_db"], dtype=float)
    cut_db = np.asarray(result["elevation_cut_db"], dtype=float)
    positions = np.asarray(result["positions_lambda"], dtype=float)
    amps = np.asarray(result["weight_amplitudes"], dtype=float)
    phases = np.deg2rad(np.asarray(result["weight_phases_deg"], dtype=float))
    phase_weights = np.exp(1j * phases)

    build_elevation_cut(theta, cut_db, config.desired_source.theta_deg).write_html(output_dir / "cut.html")
    build_heatmap(theta, phi, magnitude_db).write_html(output_dir / "heatmap.html")
    build_pattern_3d(theta, phi, 10.0 ** (magnitude_db / 20.0)).write_html(output_dir / "pattern_3d.html")
    build_weights_plot(positions, amps, phase_weights).write_html(output_dir / "weights.html")


def run_single_simulation(config: ScenarioConfig) -> dict[str, object]:
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = _run_once(config, seed=config.seed)
    payload: dict[str, object] = {
        "mode": "single",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "result": run,
    }

    if config.output.save_plots:
        _write_plots(config, run, output_dir)

    (output_dir / "simulate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_monte_carlo(config: ScenarioConfig, runs: int) -> dict[str, object]:
    if runs < 1:
        raise ValueError("runs must be >= 1")

    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run = [_run_once(config, seed=config.seed + idx) for idx in range(runs)]
    sinr = np.array([entry["sinr_db"] for entry in per_run], dtype=float)

    summary = {
        "runs": runs,
        "seed_start": config.seed,
        "seed_end": config.seed + runs - 1,
        "sinr_mean_db": float(np.mean(sinr)),
        "sinr_std_db": float(np.std(sinr)),
        "sinr_min_db": float(np.min(sinr)),
        "sinr_max_db": float(np.max(sinr)),
    }

    payload: dict[str, object] = {
        "mode": "montecarlo",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "summary": summary,
        "runs": [{"seed": entry["seed"], "sinr_db": entry["sinr_db"]} for entry in per_run],
    }

    if config.output.save_plots:
        best_idx = int(np.argmax(sinr))
        _write_plots(config, per_run[best_idx], output_dir)

    (output_dir / "montecarlo.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
