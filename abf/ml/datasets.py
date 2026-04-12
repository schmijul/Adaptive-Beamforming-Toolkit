from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import json
import numpy as np

from abf.data import simulate_array_iq_components
from simulations.config import ScenarioConfig, SourceConfig, load_scenario_config

from .config import ExperimentConfig, SamplingConfig, load_experiment_config
from .features import build_feature_vector
from .labels import build_label
from .splits import create_split_indices
from .tasks import get_task_definition, validate_task_feature_pair


@dataclass
class MLDataset:
    X: np.ndarray
    y: np.ndarray
    meta: dict[str, Any]
    split_indices: dict[str, np.ndarray]
    records: list[dict[str, Any]] | None = None

    def subset(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        idx = self.split_indices[split]
        return self.X[idx], self.y[idx]

    def train(self) -> tuple[np.ndarray, np.ndarray]:
        return self.subset("train")

    def val(self) -> tuple[np.ndarray, np.ndarray]:
        return self.subset("val")

    def test(self) -> tuple[np.ndarray, np.ndarray]:
        return self.subset("test")

    def summary(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.X.shape[0]),
            "feature_shape": list(self.X.shape),
            "target_shape": list(self.y.shape),
            "task": self.meta["task"],
            "feature_type": self.meta["feature_type"],
        }


def _resolve_experiment(
    config: str | Path | ExperimentConfig | None,
    *,
    n_samples: int | None,
    feature_type: str | None,
    target_type: str | None,
    seed: int | None,
) -> tuple[ScenarioConfig, dict[str, Any], int, str, str, dict[str, Any]]:
    if isinstance(config, ExperimentConfig):
        exp = config
        return (
            exp.scenario,
            exp.split,
            int(n_samples or exp.dataset.n_samples),
            str(feature_type or exp.dataset.feature_type),
            str(target_type or exp.dataset.target_type),
            {
                "task": exp.task_name,
                "task_params": exp.task_params,
                "seed": int(seed if seed is not None else exp.dataset.seed),
                "config": exp.to_metadata(),
                "export_path": exp.dataset.export_path,
                "sampling": exp.dataset.sampling,
            },
        )

    if config is not None:
        path = Path(config)
        if path.suffix.lower() in {".yml", ".yaml"}:
            try:
                exp = load_experiment_config(path)
                return _resolve_experiment(exp, n_samples=n_samples, feature_type=feature_type, target_type=target_type, seed=seed)
            except Exception:
                scenario = load_scenario_config(path)
                if feature_type is None or target_type is None:
                    raise ValueError("Scenario-only configs require explicit feature_type and target_type")
                return (
                    scenario,
                    {"train": 0.7, "val": 0.15, "test": 0.15},
                    int(n_samples or 1024),
                    str(feature_type),
                    str(target_type),
                    {
                        "task": str(target_type),
                        "task_params": {},
                        "seed": int(seed if seed is not None else scenario.seed),
                        "config": {"scenario_path": str(path)},
                        "export_path": None,
                        "sampling": SamplingConfig(),
                    },
                )
    raise ValueError("config must be an ExperimentConfig or a YAML path")


def _sample_uniform(rng: np.random.Generator, bounds: SamplingConfig, value: str, fallback: float) -> float:
    range_obj = getattr(bounds, value)
    if range_obj is None:
        return float(fallback)
    return float(rng.uniform(range_obj.min, range_obj.max))


def _sample_scenario(base: ScenarioConfig, sampling: SamplingConfig, *, rng: np.random.Generator) -> ScenarioConfig:
    desired_source = replace(
        base.desired_source,
        theta_deg=_sample_uniform(rng, sampling, "desired_theta_deg", base.desired_source.theta_deg),
        snr_db=_sample_uniform(rng, sampling, "desired_snr_db", base.desired_source.snr_db),
    )

    min_count, max_count = sampling.interference_count
    num_interferers = int(rng.integers(min_count, max_count + 1))
    interference_sources: list[SourceConfig] = []
    for _ in range(num_interferers):
        interference_sources.append(
            SourceConfig(
                theta_deg=_sample_uniform(rng, sampling, "interference_theta_deg", desired_source.theta_deg + 20.0),
                phi_deg=base.desired_source.phi_deg,
                snr_db=_sample_uniform(rng, sampling, "interference_snr_db", 5.0),
            )
        )

    return replace(base, desired_source=desired_source, interference_sources=tuple(interference_sources))


def _simulate_components(config: ScenarioConfig, seed: int) -> dict[str, np.ndarray]:
    source_thetas = np.array([config.desired_source.theta_deg, *[src.theta_deg for src in config.interference_sources]], dtype=float)
    source_phis = np.array([config.desired_source.phi_deg, *[src.phi_deg for src in config.interference_sources]], dtype=float)
    source_snr = np.array([config.desired_source.snr_db, *[src.snr_db for src in config.interference_sources]], dtype=float)
    components = simulate_array_iq_components(
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
    snapshots = np.asarray(components["snapshots"], dtype=np.complex128)
    components["covariance"] = (snapshots @ snapshots.conj().T) / snapshots.shape[1]
    return components


def generate_dataset(
    config: str | Path | ExperimentConfig,
    *,
    n_samples: int | None = None,
    feature_type: str | None = None,
    target_type: str | None = None,
    seed: int | None = None,
) -> MLDataset:
    base_scenario, split, sample_count, chosen_feature, chosen_target, context = _resolve_experiment(
        config,
        n_samples=n_samples,
        feature_type=feature_type,
        target_type=target_type,
        seed=seed,
    )
    task_name = str(context["task"])
    validate_task_feature_pair(task_name, chosen_feature)
    task_def = get_task_definition(task_name)

    rng = np.random.default_rng(int(context["seed"]))
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    task_params = context.get("task_params", {})
    sampling = context.get("sampling", SamplingConfig())

    for sample_idx in range(sample_count):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        scenario = _sample_scenario(base_scenario, sampling, rng=rng)
        components = _simulate_components(scenario, seed=sample_seed)
        feature = build_feature_vector(
            feature_type=chosen_feature,
            components=components,
            config=scenario,
            task_params=task_params,
        )
        label = build_label(
            task_name=task_name,
            config=scenario,
            components=components,
            task_params=task_params,
        )
        features.append(np.asarray(feature))
        labels.append(np.asarray(label))
        records.append(
            {
                "seed": sample_seed,
                "desired_theta_deg": float(scenario.desired_source.theta_deg),
                "desired_snr_db": float(scenario.desired_source.snr_db),
                "num_interferers": len(scenario.interference_sources),
                "spacing_lambda": float(scenario.array.spacing_lambda) if scenario.array.spacing_lambda is not None else None,
                "snapshots": np.asarray(components["snapshots"]),
                "covariance": np.asarray(components["covariance"]),
            }
        )

    X = np.stack(features, axis=0)
    y = np.stack(labels, axis=0)
    if task_def.label_dtype == "int":
        y = y.astype(np.int64).reshape(-1)
    else:
        y = y.astype(np.float64)

    split_indices = create_split_indices(sample_count, seed=int(context["seed"]), **split)
    meta = {
        "task": task_name,
        "feature_type": chosen_feature,
        "target_type": chosen_target,
        "seed": int(context["seed"]),
        "config": context["config"],
        "samples": [{"seed": rec["seed"], "desired_theta_deg": rec["desired_theta_deg"], "num_interferers": rec["num_interferers"]} for rec in records],
    }
    return MLDataset(X=X, y=y, meta=meta, split_indices=split_indices, records=records)


def save_dataset(dataset: MLDataset, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        X=dataset.X,
        y=dataset.y,
        split_train=dataset.split_indices["train"],
        split_val=dataset.split_indices["val"],
        split_test=dataset.split_indices["test"],
        meta=np.array([json.dumps(dataset.meta)], dtype=object),
    )
    return target


def load_dataset(path: str | Path) -> MLDataset:
    container = np.load(Path(path), allow_pickle=True)
    meta = json.loads(str(container["meta"][0]))
    split_indices = {
        "train": np.asarray(container["split_train"], dtype=int),
        "val": np.asarray(container["split_val"], dtype=int),
        "test": np.asarray(container["split_test"], dtype=int),
    }
    return MLDataset(
        X=np.asarray(container["X"]),
        y=np.asarray(container["y"]),
        meta=meta,
        split_indices=split_indices,
        records=None,
    )
