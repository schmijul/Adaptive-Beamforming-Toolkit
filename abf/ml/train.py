from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, load_experiment_config
from .datasets import generate_dataset, save_dataset
from .evaluate import EvaluationResult, evaluate_model
from .io import ensure_dir, write_json
from .models import create_model, save_model
from .tasks import get_task_definition


@dataclass(frozen=True)
class ExperimentResult:
    config_name: str
    metrics: dict[str, Any]
    baselines: dict[str, dict[str, Any]]
    output_dir: str
    model_path: str | None
    dataset_path: str | None
    results_path: str


def run_experiment(config: str | Path | ExperimentConfig) -> ExperimentResult:
    exp = load_experiment_config(config) if not isinstance(config, ExperimentConfig) else config
    task = get_task_definition(exp.task_name)
    metric_names = list(exp.metrics or task.default_metrics)
    baseline_names = list(exp.baselines or task.baseline_names)

    dataset = generate_dataset(exp)
    output_dir = ensure_dir(exp.output.dir)

    dataset_path = None
    export_path = exp.dataset.export_path
    if exp.output.save_dataset or export_path:
        dataset_path = str(save_dataset(dataset, export_path or output_dir / "dataset.npz"))

    model = create_model(exp.model.family, exp.model.name, exp.model.params, exp.task_name)
    X_train, y_train = dataset.train()
    model.fit(X_train, y_train)

    val_result: EvaluationResult | None = None
    if dataset.split_indices["val"].size:
        val_result = evaluate_model(
            model,
            dataset,
            split="val",
            metric_names=metric_names,
            baseline_names=baseline_names,
            task_name=exp.task_name,
            task_params=exp.task_params,
        )

    test_result = evaluate_model(
        model,
        dataset,
        split="test",
        metric_names=metric_names,
        baseline_names=baseline_names,
        task_name=exp.task_name,
        task_params=exp.task_params,
    )

    model_path = None
    if exp.output.save_model:
        model_path = str(save_model(model, output_dir / "model.pkl"))

    payload = {
        "experiment": exp.name,
        "task": exp.task_name,
        "feature_type": exp.dataset.feature_type,
        "metrics": test_result.metrics,
        "baselines": test_result.baselines,
        "validation_metrics": val_result.metrics if val_result is not None else {},
        "config": exp.to_metadata(),
    }
    results_path = str(write_json(output_dir / "results.json", payload))
    return ExperimentResult(
        config_name=exp.name,
        metrics=test_result.metrics,
        baselines=test_result.baselines,
        output_dir=str(output_dir),
        model_path=model_path,
        dataset_path=dataset_path,
        results_path=results_path,
    )
