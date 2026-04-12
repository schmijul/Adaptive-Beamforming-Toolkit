from __future__ import annotations

from typing import Any

import numpy as np

from abf.algorithms import doa_music_linear

from .metrics import compute_metrics


def _music_predictions(records: list[dict[str, Any]], task_name: str, task_params: dict[str, Any]) -> np.ndarray:
    predictions = []
    theta_scan = np.asarray(task_params.get("theta_scan_deg"), dtype=float) if "theta_scan_deg" in task_params else None
    class_edges = np.asarray(task_params.get("class_edges_deg"), dtype=float) if "class_edges_deg" in task_params else None
    for record in records:
        snapshots = np.asarray(record["snapshots"], dtype=np.complex128)
        scan = theta_scan if theta_scan is not None else np.linspace(-90.0, 90.0, 181)
        spacing_lambda = float(record.get("spacing_lambda") or 0.5)
        result = doa_music_linear(
            snapshots=snapshots,
            spacing_lambda=spacing_lambda,
            theta_scan_deg=scan,
            num_sources=1,
        )
        theta_est = float(result["estimated_thetas_deg"][0])
        if task_name == "doa_classification":
            if class_edges is None:
                class_edges = np.linspace(scan.min(), scan.max(), 13)
            predictions.append(int(np.clip(np.digitize(theta_est, class_edges[1:-1], right=False), 0, class_edges.size - 2)))
        else:
            predictions.append(theta_est)
    return np.asarray(predictions)


def _beam_search_predictions(records: list[dict[str, Any]], task_params: dict[str, Any]) -> np.ndarray:
    beam_angles = np.asarray(task_params.get("beam_angles_deg"))
    if beam_angles.ndim != 1:
        raise ValueError("beam_angles_deg must be configured for beam_selection baselines")
    predictions = []
    for record in records:
        desired_theta = float(record["desired_theta_deg"])
        predictions.append(int(np.argmin(np.abs(beam_angles - desired_theta))))
    return np.asarray(predictions)


def _energy_detector_predictions(records: list[dict[str, Any]], task_params: dict[str, Any]) -> np.ndarray:
    threshold = float(task_params.get("energy_threshold", 0.0))
    values = []
    for record in records:
        covariance = np.asarray(record["covariance"])
        score = float(np.trace(np.real(covariance)))
        values.append(int(score > threshold))
    return np.asarray(values)


def evaluate_baselines(
    *,
    baseline_names: list[str] | tuple[str, ...],
    records: list[dict[str, Any]] | None,
    y_true: np.ndarray,
    metric_names: list[str] | tuple[str, ...],
    task_name: str,
    task_params: dict[str, Any],
) -> dict[str, dict[str, float | list[list[int]]]]:
    if records is None:
        return {}

    results: dict[str, dict[str, float | list[list[int]]]] = {}
    for name in baseline_names:
        baseline_name = str(name).lower()
        if baseline_name == "music":
            pred = _music_predictions(records, task_name=task_name, task_params=task_params)
        elif baseline_name == "conventional_beam_search":
            pred = _beam_search_predictions(records, task_params=task_params)
        elif baseline_name == "energy_detector":
            pred = _energy_detector_predictions(records, task_params=task_params)
        elif baseline_name == "configured_weights":
            continue
        else:
            raise ValueError(f"Unsupported baseline: {name}")
        results[baseline_name] = compute_metrics(metric_names, y_true, pred)
    return results
