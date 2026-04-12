from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .baselines import evaluate_baselines
from .datasets import MLDataset
from .metrics import compute_metrics


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: np.ndarray
    baselines: dict[str, dict[str, Any]]


def evaluate_model(
    model: Any,
    dataset: MLDataset,
    *,
    split: str = "test",
    metric_names: list[str] | tuple[str, ...] = ("mae", "rmse"),
    baseline_names: list[str] | tuple[str, ...] = (),
    task_name: str,
    task_params: dict[str, Any] | None = None,
) -> EvaluationResult:
    X_split, y_split = dataset.subset(split)
    predictions = np.asarray(model.predict(X_split))
    metrics = compute_metrics(metric_names, y_split, predictions)

    baseline_records = None
    if dataset.records is not None:
        indices = dataset.split_indices[split]
        baseline_records = [dataset.records[int(idx)] for idx in indices]
    baselines = evaluate_baselines(
        baseline_names=baseline_names,
        records=baseline_records,
        y_true=y_split,
        metric_names=metric_names,
        task_name=task_name,
        task_params=task_params or {},
    )
    return EvaluationResult(metrics=metrics, predictions=predictions, baselines=baselines)
