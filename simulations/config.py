from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SourceConfig:
    theta_deg: float
    phi_deg: float
    snr_db: float


@dataclass(frozen=True)
class ArrayConfig:
    geometry: str
    num_elements: int
    spacing_lambda: float | None = None
    num_x: int | None = None
    num_y: int | None = None
    spacing_x_lambda: float | None = None
    spacing_y_lambda: float | None = None


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    diagonal_loading: float
    step_size: float
    leakage: float
    epsilon: float
    forgetting_factor: float
    initialization_delta: float


@dataclass(frozen=True)
class SweepConfig:
    theta_start_deg: float
    theta_stop_deg: float
    theta_num: int
    phi_start_deg: float
    phi_stop_deg: float
    phi_num: int


@dataclass(frozen=True)
class OutputConfig:
    directory: str
    save_plots: bool


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    seed: int
    snapshots: int
    array: ArrayConfig
    desired_source: SourceConfig
    interference_sources: tuple[SourceConfig, ...]
    algorithm: AlgorithmConfig
    sweep: SweepConfig
    output: OutputConfig


def _require_keys(container: dict[str, Any], required: tuple[str, ...], context: str) -> None:
    missing = [key for key in required if key not in container]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {', '.join(missing)}")


def _parse_source(data: dict[str, Any], context: str) -> SourceConfig:
    _require_keys(data, ("theta_deg", "phi_deg", "snr_db"), context)
    return SourceConfig(
        theta_deg=float(data["theta_deg"]),
        phi_deg=float(data["phi_deg"]),
        snr_db=float(data["snr_db"]),
    )


def _parse_array_config(array_data: dict[str, Any]) -> ArrayConfig:
    _require_keys(array_data, ("geometry",), "array")
    geometry = str(array_data["geometry"]).lower()
    if geometry == "ula":
        _require_keys(array_data, ("num_elements", "spacing_lambda"), "array")
        num_elements = int(array_data["num_elements"])
        spacing_lambda = float(array_data["spacing_lambda"])
        if num_elements < 1:
            raise ValueError("array.num_elements must be >= 1")
        if spacing_lambda <= 0.0:
            raise ValueError("array.spacing_lambda must be > 0")
        return ArrayConfig(
            geometry="ula",
            num_elements=num_elements,
            spacing_lambda=spacing_lambda,
        )

    if geometry in {"planar", "upa"}:
        _require_keys(array_data, ("num_x", "num_y", "spacing_x_lambda", "spacing_y_lambda"), "array")
        num_x = int(array_data["num_x"])
        num_y = int(array_data["num_y"])
        spacing_x_lambda = float(array_data["spacing_x_lambda"])
        spacing_y_lambda = float(array_data["spacing_y_lambda"])
        if num_x < 1 or num_y < 1:
            raise ValueError("array.num_x and array.num_y must be >= 1")
        if spacing_x_lambda <= 0.0 or spacing_y_lambda <= 0.0:
            raise ValueError("array planar spacings must be > 0")
        return ArrayConfig(
            geometry="planar",
            num_elements=num_x * num_y,
            num_x=num_x,
            num_y=num_y,
            spacing_x_lambda=spacing_x_lambda,
            spacing_y_lambda=spacing_y_lambda,
        )

    raise ValueError("array.geometry must be one of: ula, planar, upa")


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Config file does not exist: {source}")

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a top-level mapping")

    _require_keys(
        payload,
        ("name", "seed", "snapshots", "array", "desired_source", "interference_sources", "algorithm", "sweep", "output"),
        "scenario",
    )

    array = _parse_array_config(payload["array"])

    algorithm_data = payload["algorithm"]
    _require_keys(algorithm_data, ("name",), "algorithm")
    algorithm_name = str(algorithm_data["name"]).lower()
    if algorithm_name not in {"conventional", "mvdr", "lms", "nlms", "rls"}:
        raise ValueError("algorithm.name must be one of: conventional, mvdr, lms, nlms, rls")

    sweep_data = payload["sweep"]
    _require_keys(
        sweep_data,
        ("theta_start_deg", "theta_stop_deg", "theta_num", "phi_start_deg", "phi_stop_deg", "phi_num"),
        "sweep",
    )

    output_data = payload["output"]
    _require_keys(output_data, ("directory", "save_plots"), "output")

    interference = payload["interference_sources"]
    if not isinstance(interference, list):
        raise ValueError("interference_sources must be a list")

    config = ScenarioConfig(
        name=str(payload["name"]),
        seed=int(payload["seed"]),
        snapshots=int(payload["snapshots"]),
        array=array,
        desired_source=_parse_source(payload["desired_source"], "desired_source"),
        interference_sources=tuple(
            _parse_source(item, f"interference_sources[{idx}]") for idx, item in enumerate(interference)
        ),
        algorithm=AlgorithmConfig(
            name=algorithm_name,
            diagonal_loading=float(algorithm_data.get("diagonal_loading", 1e-3)),
            step_size=float(algorithm_data.get("step_size", 0.05)),
            leakage=float(algorithm_data.get("leakage", 0.0)),
            epsilon=float(algorithm_data.get("epsilon", 1e-6)),
            forgetting_factor=float(algorithm_data.get("forgetting_factor", 0.995)),
            initialization_delta=float(algorithm_data.get("initialization_delta", 1.0)),
        ),
        sweep=SweepConfig(
            theta_start_deg=float(sweep_data["theta_start_deg"]),
            theta_stop_deg=float(sweep_data["theta_stop_deg"]),
            theta_num=int(sweep_data["theta_num"]),
            phi_start_deg=float(sweep_data["phi_start_deg"]),
            phi_stop_deg=float(sweep_data["phi_stop_deg"]),
            phi_num=int(sweep_data["phi_num"]),
        ),
        output=OutputConfig(
            directory=str(output_data["directory"]),
            save_plots=bool(output_data["save_plots"]),
        ),
    )

    if config.snapshots < 2:
        raise ValueError("snapshots must be >= 2")
    if config.sweep.theta_num < 2 or config.sweep.phi_num < 2:
        raise ValueError("sweep.theta_num and sweep.phi_num must be >= 2")
    if config.algorithm.diagonal_loading < 0.0:
        raise ValueError("algorithm.diagonal_loading must be >= 0")
    if config.algorithm.step_size <= 0.0:
        raise ValueError("algorithm.step_size must be > 0")
    if config.algorithm.leakage < 0.0:
        raise ValueError("algorithm.leakage must be >= 0")
    if config.algorithm.epsilon <= 0.0:
        raise ValueError("algorithm.epsilon must be > 0")
    if not (0.0 < config.algorithm.forgetting_factor <= 1.0):
        raise ValueError("algorithm.forgetting_factor must be in (0, 1]")
    if config.algorithm.initialization_delta <= 0.0:
        raise ValueError("algorithm.initialization_delta must be > 0")

    return config
