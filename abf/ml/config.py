from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from simulations.config import ScenarioConfig, parse_scenario_config


@dataclass(frozen=True)
class SamplingRange:
    min: float
    max: float


@dataclass(frozen=True)
class SamplingConfig:
    desired_theta_deg: SamplingRange | None = None
    desired_snr_db: SamplingRange | None = None
    interference_count: tuple[int, int] = (0, 1)
    interference_theta_deg: SamplingRange | None = None
    interference_snr_db: SamplingRange | None = None


@dataclass(frozen=True)
class DatasetConfig:
    n_samples: int
    seed: int
    feature_type: str
    target_type: str
    export_path: str | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass(frozen=True)
class ModelConfig:
    family: str = "numpy"
    name: str = "auto"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "outputs/ml"
    save_dataset: bool = False
    save_model: bool = True


@dataclass(frozen=True)
class EnvironmentConfig:
    feature_type: str = "covariance_real_imag"
    beam_angles_deg: tuple[float, ...] = (-60.0, -30.0, 0.0, 30.0, 60.0)
    reward_mode: str = "gain"
    episode_length: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    task_name: str
    scenario: ScenarioConfig
    dataset: DatasetConfig
    split: dict[str, float]
    model: ModelConfig
    metrics: tuple[str, ...]
    baselines: tuple[str, ...]
    output: OutputConfig
    task_params: dict[str, Any] = field(default_factory=dict)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    source_path: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenario"] = asdict(self.scenario)
        return payload


def _parse_range(value: Any, name: str) -> SamplingRange | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "min" not in value or "max" not in value:
            raise ValueError(f"{name} must contain min and max")
        return SamplingRange(min=float(value["min"]), max=float(value["max"]))
    scalar = float(value)
    return SamplingRange(min=scalar, max=scalar)


def _parse_sampling(data: dict[str, Any] | None) -> SamplingConfig:
    payload = data or {}
    count = payload.get("interference_count", [0, 1])
    if isinstance(count, int):
        count_range = (count, count)
    else:
        values = list(count)
        if len(values) != 2:
            raise ValueError("interference_count must be an int or a two-element sequence")
        count_range = (int(values[0]), int(values[1]))
    return SamplingConfig(
        desired_theta_deg=_parse_range(payload.get("desired_theta_deg"), "desired_theta_deg"),
        desired_snr_db=_parse_range(payload.get("desired_snr_db"), "desired_snr_db"),
        interference_count=count_range,
        interference_theta_deg=_parse_range(payload.get("interference_theta_deg"), "interference_theta_deg"),
        interference_snr_db=_parse_range(payload.get("interference_snr_db"), "interference_snr_db"),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Config file does not exist: {source}")

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must contain a top-level mapping")

    scenario_payload = payload.get("scenario")
    if not isinstance(scenario_payload, dict):
        raise ValueError("Experiment config requires a 'scenario' mapping")
    scenario = parse_scenario_config(scenario_payload)

    dataset_payload = payload.get("dataset", {})
    target_type = str(dataset_payload.get("target_type", payload.get("task", {}).get("name", "doa_regression")))
    feature_type = str(dataset_payload.get("feature_type", "covariance_real_imag"))

    split_payload = payload.get("split", {})
    split = {
        "train": float(split_payload.get("train", 0.7)),
        "val": float(split_payload.get("val", 0.15)),
        "test": float(split_payload.get("test", 0.15)),
    }

    model_payload = payload.get("model", {})
    output_payload = payload.get("output", {})
    task_payload = payload.get("task", {})
    env_payload = payload.get("env", {})

    experiment_name = str(payload.get("experiment", {}).get("name", source.stem))
    task_name = str(task_payload.get("name", target_type))

    metrics = tuple(payload.get("metrics") or ())
    baselines = tuple(payload.get("baselines") or ())

    return ExperimentConfig(
        name=experiment_name,
        task_name=task_name,
        scenario=scenario,
        dataset=DatasetConfig(
            n_samples=int(dataset_payload.get("n_samples", 1024)),
            seed=int(dataset_payload.get("seed", scenario.seed)),
            feature_type=feature_type,
            target_type=target_type,
            export_path=dataset_payload.get("export_path"),
            sampling=_parse_sampling(dataset_payload.get("sampling")),
        ),
        split=split,
        model=ModelConfig(
            family=str(model_payload.get("family", "numpy")),
            name=str(model_payload.get("name", "auto")),
            params=dict(model_payload.get("params", {})),
        ),
        metrics=metrics,
        baselines=baselines,
        output=OutputConfig(
            dir=str(output_payload.get("dir", f"outputs/experiments/{experiment_name}")),
            save_dataset=bool(output_payload.get("save_dataset", False)),
            save_model=bool(output_payload.get("save_model", True)),
        ),
        task_params=dict(task_payload.get("params", {})),
        env=EnvironmentConfig(
            feature_type=str(env_payload.get("feature_type", "covariance_real_imag")),
            beam_angles_deg=tuple(float(v) for v in env_payload.get("beam_angles_deg", (-60.0, -30.0, 0.0, 30.0, 60.0))),
            reward_mode=str(env_payload.get("reward_mode", "gain")),
            episode_length=int(env_payload.get("episode_length", 1)),
        ),
        source_path=str(source),
    )
