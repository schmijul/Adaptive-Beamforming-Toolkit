from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    valid_feature_types: tuple[str, ...]
    default_metrics: tuple[str, ...]
    baseline_names: tuple[str, ...]
    label_dtype: str
    problem_type: str


_TASKS: dict[str, TaskDefinition] = {
    "doa_regression": TaskDefinition(
        name="doa_regression",
        valid_feature_types=("raw_iq", "real_imag", "covariance", "covariance_real_imag", "spectrum"),
        default_metrics=("mae", "rmse"),
        baseline_names=("music",),
        label_dtype="float",
        problem_type="regression",
    ),
    "doa_classification": TaskDefinition(
        name="doa_classification",
        valid_feature_types=("real_imag", "covariance_real_imag", "spectrum"),
        default_metrics=("accuracy", "balanced_accuracy"),
        baseline_names=("music",),
        label_dtype="int",
        problem_type="classification",
    ),
    "beam_selection": TaskDefinition(
        name="beam_selection",
        valid_feature_types=("real_imag", "covariance_real_imag", "spectrum"),
        default_metrics=("accuracy", "balanced_accuracy"),
        baseline_names=("conventional_beam_search",),
        label_dtype="int",
        problem_type="classification",
    ),
    "interference_detection": TaskDefinition(
        name="interference_detection",
        valid_feature_types=("real_imag", "covariance_real_imag", "spectrum"),
        default_metrics=("accuracy", "balanced_accuracy"),
        baseline_names=("energy_detector",),
        label_dtype="int",
        problem_type="classification",
    ),
    "weight_regression": TaskDefinition(
        name="weight_regression",
        valid_feature_types=("covariance", "covariance_real_imag", "spectrum"),
        default_metrics=("mae", "rmse"),
        baseline_names=("configured_weights",),
        label_dtype="float",
        problem_type="regression",
    ),
}


def get_task_definition(name: str) -> TaskDefinition:
    task_name = str(name).lower()
    if task_name not in _TASKS:
        raise ValueError(f"Unsupported task: {name}")
    return _TASKS[task_name]


def validate_task_feature_pair(task_name: str, feature_type: str) -> None:
    task = get_task_definition(task_name)
    if feature_type not in task.valid_feature_types:
        raise ValueError(
            f"feature_type '{feature_type}' is not supported for task '{task_name}'. "
            f"Expected one of: {', '.join(task.valid_feature_types)}"
        )


def list_supported_tasks() -> tuple[str, ...]:
    return tuple(_TASKS.keys())
