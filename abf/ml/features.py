from __future__ import annotations

from typing import Any

import numpy as np

from abf.algorithms import estimate_covariance_matrix, linear_steering_vector, planar_steering_vector
from simulations.config import ScenarioConfig


def _scan_angles(config: ScenarioConfig, task_params: dict[str, Any]) -> np.ndarray:
    angles = task_params.get("beam_angles_deg") or task_params.get("theta_scan_deg")
    if angles is not None:
        return np.asarray(angles, dtype=float).reshape(-1)
    return np.linspace(config.sweep.theta_start_deg, config.sweep.theta_stop_deg, config.sweep.theta_num)


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


def build_feature_vector(
    *,
    feature_type: str,
    components: dict[str, np.ndarray],
    config: ScenarioConfig,
    task_params: dict[str, Any] | None = None,
) -> np.ndarray:
    feature_name = str(feature_type).lower()
    params = task_params or {}
    snapshots = np.asarray(components["snapshots"], dtype=np.complex128)

    if feature_name == "raw_iq":
        return snapshots.reshape(-1)

    if feature_name == "real_imag":
        return np.concatenate([snapshots.real.reshape(-1), snapshots.imag.reshape(-1)]).astype(np.float64)

    covariance = estimate_covariance_matrix(snapshots)
    if feature_name == "covariance":
        return covariance.reshape(-1)

    if feature_name == "covariance_real_imag":
        return np.concatenate([covariance.real.reshape(-1), covariance.imag.reshape(-1)]).astype(np.float64)

    if feature_name == "spectrum":
        theta_scan = _scan_angles(config, params)
        responses = []
        for theta_deg in theta_scan:
            steer = _steering(config, float(theta_deg), 0.0)
            power = np.real(np.vdot(steer, covariance @ steer))
            responses.append(float(power))
        return np.asarray(responses, dtype=np.float64)

    raise ValueError(f"Unsupported feature_type: {feature_type}")
