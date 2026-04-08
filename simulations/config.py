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
    spacing_lambda: float


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    diagonal_loading: float


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

    array_data = payload["array"]
    _require_keys(array_data, ("geometry", "num_elements", "spacing_lambda"), "array")
    geometry = str(array_data["geometry"]).lower()
    if geometry != "ula":
        raise ValueError("Only 'ula' geometry is supported in the current simulation runner")

    algorithm_data = payload["algorithm"]
    _require_keys(algorithm_data, ("name",), "algorithm")
    algorithm_name = str(algorithm_data["name"]).lower()
    if algorithm_name not in {"conventional", "mvdr"}:
        raise ValueError("algorithm.name must be one of: conventional, mvdr")

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
        array=ArrayConfig(
            geometry=geometry,
            num_elements=int(array_data["num_elements"]),
            spacing_lambda=float(array_data["spacing_lambda"]),
        ),
        desired_source=_parse_source(payload["desired_source"], "desired_source"),
        interference_sources=tuple(
            _parse_source(item, f"interference_sources[{idx}]") for idx, item in enumerate(interference)
        ),
        algorithm=AlgorithmConfig(
            name=algorithm_name,
            diagonal_loading=float(algorithm_data.get("diagonal_loading", 1e-3)),
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

    if config.array.num_elements < 1:
        raise ValueError("array.num_elements must be >= 1")
    if config.array.spacing_lambda <= 0.0:
        raise ValueError("array.spacing_lambda must be > 0")
    if config.snapshots < 2:
        raise ValueError("snapshots must be >= 2")
    if config.sweep.theta_num < 2 or config.sweep.phi_num < 2:
        raise ValueError("sweep.theta_num and sweep.phi_num must be >= 2")

    return config
