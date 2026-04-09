from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from algorithms.adaptive import (
    estimate_covariance_matrix,
    linear_steering_vector,
    lms_weights,
    mvdr_weights,
    nlms_weights,
    planar_steering_vector,
    rls_weights,
)
from core import array_factor_linear_from_weights, array_factor_planar_from_weights
from data.iq import simulate_array_iq_components
from simulations.config import ScenarioConfig
from visualize.plots import build_elevation_cut, build_heatmap, build_pattern_3d, build_weights_plot


def _build_grids(config: ScenarioConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(config.sweep.theta_start_deg, config.sweep.theta_stop_deg, config.sweep.theta_num)
    phi = np.linspace(config.sweep.phi_start_deg, config.sweep.phi_stop_deg, config.sweep.phi_num)
    theta_grid = theta[:, None] * np.ones((1, phi.size))
    phi_grid = np.ones((theta.size, 1)) * phi[None, :]
    return theta, phi, theta_grid, phi_grid


def _steering_vector(config: ScenarioConfig, theta_deg: float, phi_deg: float) -> np.ndarray:
    if config.array.geometry == "ula":
        assert config.array.spacing_lambda is not None
        return linear_steering_vector(
            num_elements=config.array.num_elements,
            spacing_lambda=config.array.spacing_lambda,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
        )

    assert config.array.num_x is not None
    assert config.array.num_y is not None
    assert config.array.spacing_x_lambda is not None
    assert config.array.spacing_y_lambda is not None
    return planar_steering_vector(
        num_x=config.array.num_x,
        num_y=config.array.num_y,
        spacing_x_lambda=config.array.spacing_x_lambda,
        spacing_y_lambda=config.array.spacing_y_lambda,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )


def _simulate_components(config: ScenarioConfig, seed: int) -> dict[str, np.ndarray]:
    source_thetas = np.array([config.desired_source.theta_deg, *[src.theta_deg for src in config.interference_sources]], dtype=float)
    source_phis = np.array([config.desired_source.phi_deg, *[src.phi_deg for src in config.interference_sources]], dtype=float)
    source_snr = np.array([config.desired_source.snr_db, *[src.snr_db for src in config.interference_sources]], dtype=float)

    return simulate_array_iq_components(
        num_elements=config.array.num_elements,
        num_snapshots=config.snapshots,
        spacing_lambda=config.array.spacing_lambda,
        source_thetas_deg=source_thetas,
        source_phis_deg=source_phis,
        source_snr_db=source_snr,
        random_seed=seed,
        geometry=config.array.geometry,
        num_x=config.array.num_x,
        num_y=config.array.num_y,
        spacing_x_lambda=config.array.spacing_x_lambda,
        spacing_y_lambda=config.array.spacing_y_lambda,
    )


def _select_weights(config: ScenarioConfig, snapshots: np.ndarray, desired_signal: np.ndarray) -> np.ndarray:
    steer = _steering_vector(config, config.desired_source.theta_deg, config.desired_source.phi_deg)

    if config.algorithm.name == "conventional":
        denom = np.vdot(steer, steer)
        return steer / denom

    if config.algorithm.name == "mvdr":
        covariance = estimate_covariance_matrix(snapshots, diagonal_loading=0.0)
        return mvdr_weights(
            covariance_matrix=covariance,
            steering_vector=steer,
            diagonal_loading=config.algorithm.diagonal_loading,
        )

    if config.algorithm.name == "lms":
        return lms_weights(
            snapshots=snapshots,
            desired_signal=desired_signal,
            step_size=config.algorithm.step_size,
            leakage=config.algorithm.leakage,
        )["weights"]

    if config.algorithm.name == "nlms":
        return nlms_weights(
            snapshots=snapshots,
            desired_signal=desired_signal,
            step_size=config.algorithm.step_size,
            epsilon=config.algorithm.epsilon,
            leakage=config.algorithm.leakage,
        )["weights"]

    if config.algorithm.name == "rls":
        return rls_weights(
            snapshots=snapshots,
            desired_signal=desired_signal,
            forgetting_factor=config.algorithm.forgetting_factor,
            initialization_delta=config.algorithm.initialization_delta,
        )["weights"]

    raise ValueError(f"Unsupported algorithm: {config.algorithm.name}")


def _calculate_sinr_db(config: ScenarioConfig, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.complex128).reshape(-1)

    desired_vec = _steering_vector(config, config.desired_source.theta_deg, config.desired_source.phi_deg)
    desired_power = 10.0 ** (config.desired_source.snr_db / 10.0)
    signal = (np.abs(np.vdot(w, desired_vec)) ** 2) * desired_power

    interference = 0.0
    for src in config.interference_sources:
        interferer_vec = _steering_vector(config, src.theta_deg, src.phi_deg)
        interference += (np.abs(np.vdot(w, interferer_vec)) ** 2) * (10.0 ** (src.snr_db / 10.0))

    noise = float(np.real(np.vdot(w, w)))
    return float(10.0 * np.log10(max(signal / (interference + noise + 1e-15), 1e-15)))


def _evaluate_pattern(
    config: ScenarioConfig,
    weights: np.ndarray,
    theta_grid_deg: np.ndarray,
    phi_grid_deg: np.ndarray,
) -> dict[str, np.ndarray]:
    if config.array.geometry == "ula":
        assert config.array.spacing_lambda is not None
        return array_factor_linear_from_weights(
            weights=weights,
            spacing_lambda=config.array.spacing_lambda,
            theta_grid_deg=theta_grid_deg,
            phi_grid_deg=phi_grid_deg,
        )

    assert config.array.num_x is not None
    assert config.array.num_y is not None
    assert config.array.spacing_x_lambda is not None
    assert config.array.spacing_y_lambda is not None
    return array_factor_planar_from_weights(
        weights=weights,
        num_x=config.array.num_x,
        num_y=config.array.num_y,
        spacing_x_lambda=config.array.spacing_x_lambda,
        spacing_y_lambda=config.array.spacing_y_lambda,
        theta_grid_deg=theta_grid_deg,
        phi_grid_deg=phi_grid_deg,
    )


def _run_once(config: ScenarioConfig, seed: int) -> dict[str, object]:
    components = _simulate_components(config, seed=seed)
    snapshots = components["snapshots"]
    desired_signal = components["source_signals"][0]
    weights = _select_weights(config, snapshots, desired_signal=desired_signal)

    theta, phi, theta_grid, phi_grid = _build_grids(config)
    pattern = _evaluate_pattern(config, weights, theta_grid_deg=theta_grid, phi_grid_deg=phi_grid)
    sinr_db = _calculate_sinr_db(config, weights)

    phi0_idx = int(np.argmin(np.abs(phi)))
    cut_db = pattern["magnitude_db"][:, phi0_idx]

    result: dict[str, object] = {
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
    }

    if "positions_lambda" in pattern:
        result["positions_lambda"] = np.asarray(pattern["positions_lambda"], dtype=float).tolist()
    if "positions_xy_lambda" in pattern:
        result["positions_xy_lambda"] = np.asarray(pattern["positions_xy_lambda"], dtype=float).tolist()

    return result


def _write_plots(config: ScenarioConfig, result: dict[str, object], output_dir: Path) -> None:
    theta = np.asarray(result["theta_deg"], dtype=float)
    phi = np.asarray(result["phi_deg"], dtype=float)
    magnitude_db = np.asarray(result["magnitude_db"], dtype=float)
    cut_db = np.asarray(result["elevation_cut_db"], dtype=float)
    amps = np.asarray(result["weight_amplitudes"], dtype=float)
    phases = np.deg2rad(np.asarray(result["weight_phases_deg"], dtype=float))
    phase_weights = np.exp(1j * phases)

    if "positions_lambda" in result:
        weight_axis = np.asarray(result["positions_lambda"], dtype=float)
        axis_title = "Element Position (lambda)"
    else:
        weight_axis = np.arange(amps.size, dtype=float)
        axis_title = "Element Index"

    build_elevation_cut(theta, cut_db, config.desired_source.theta_deg).write_html(output_dir / "cut.html")
    build_heatmap(theta, phi, magnitude_db).write_html(output_dir / "heatmap.html")
    build_pattern_3d(theta, phi, 10.0 ** (magnitude_db / 20.0)).write_html(output_dir / "pattern_3d.html")
    build_weights_plot(weight_axis, amps, phase_weights, xaxis_title=axis_title).write_html(output_dir / "weights.html")


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


def run_monte_carlo(config: ScenarioConfig, runs: int, jobs: int = 1) -> dict[str, object]:
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if jobs < 1:
        raise ValueError("jobs must be >= 1")

    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [config.seed + idx for idx in range(runs)]
    if jobs == 1:
        per_run = [_run_once(config, seed=seed) for seed in seeds]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            per_run = list(executor.map(lambda seed: _run_once(config, seed=seed), seeds))

    sinr = np.array([entry["sinr_db"] for entry in per_run], dtype=float)

    summary = {
        "runs": runs,
        "jobs": jobs,
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
