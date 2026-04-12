from __future__ import annotations

from typing import Any

import numpy as np

from abf.algorithms import linear_steering_vector, mvdr_weights, planar_steering_vector
from simulations.config import ScenarioConfig


def _classification_edges(config: ScenarioConfig, task_params: dict[str, Any]) -> np.ndarray:
    edges = task_params.get("class_edges_deg")
    if edges is not None:
        arr = np.asarray(edges, dtype=float).reshape(-1)
        if arr.size < 2:
            raise ValueError("class_edges_deg must contain at least two edges")
        return arr

    num_classes = int(task_params.get("num_classes", 12))
    return np.linspace(config.sweep.theta_start_deg, config.sweep.theta_stop_deg, num_classes + 1)


def _beam_angles(config: ScenarioConfig, task_params: dict[str, Any]) -> np.ndarray:
    angles = task_params.get("beam_angles_deg")
    if angles is not None:
        arr = np.asarray(angles, dtype=float).reshape(-1)
        if arr.size < 2:
            raise ValueError("beam_angles_deg must contain at least two candidate beams")
        return arr
    return np.linspace(config.sweep.theta_start_deg, config.sweep.theta_stop_deg, max(config.sweep.theta_num, 9))


def _steering(config: ScenarioConfig, theta_deg: float, phi_deg: float = 0.0) -> np.ndarray:
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


def build_label(
    *,
    task_name: str,
    config: ScenarioConfig,
    components: dict[str, np.ndarray],
    task_params: dict[str, Any] | None = None,
) -> np.ndarray | float | int:
    task = str(task_name).lower()
    params = task_params or {}
    desired_theta = float(config.desired_source.theta_deg)
    covariance = components.get("covariance")
    if covariance is None:
        snapshots = np.asarray(components["snapshots"], dtype=np.complex128)
        covariance = (snapshots @ snapshots.conj().T) / snapshots.shape[1]

    if task == "doa_regression":
        return np.asarray([desired_theta], dtype=np.float64)

    if task == "doa_classification":
        edges = _classification_edges(config, params)
        return int(np.clip(np.digitize(desired_theta, edges[1:-1], right=False), 0, edges.size - 2))

    if task == "beam_selection":
        beam_angles = _beam_angles(config, params)
        return int(np.argmin(np.abs(beam_angles - desired_theta)))

    if task == "interference_detection":
        threshold_db = float(params.get("interference_threshold_db", -120.0))
        has_interference = any(src.snr_db >= threshold_db for src in config.interference_sources)
        return int(has_interference)

    if task == "weight_regression":
        algorithm = str(params.get("weight_algorithm", "mvdr")).lower()
        steer = _steering(config, desired_theta, float(config.desired_source.phi_deg))
        if algorithm == "mvdr":
            weights = mvdr_weights(covariance, steer, diagonal_loading=1e-3)
        elif algorithm == "conventional":
            weights = steer / np.vdot(steer, steer)
        else:
            raise ValueError(f"Unsupported weight_algorithm: {algorithm}")
        return np.concatenate([weights.real.reshape(-1), weights.imag.reshape(-1)]).astype(np.float64)

    raise ValueError(f"Unsupported task: {task_name}")
