from __future__ import annotations

import math

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(truth - pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean((truth - pred) ** 2)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(truth == pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    scores = []
    for label in np.unique(truth):
        mask = truth == label
        scores.append(np.mean(pred[mask] == truth[mask]))
    return float(np.mean(scores))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    truth = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    labels = np.unique(np.concatenate([truth, pred]))
    matrix = np.zeros((labels.size, labels.size), dtype=int)
    for row, label_true in enumerate(labels):
        for col, label_pred in enumerate(labels):
            matrix[row, col] = int(np.sum((truth == label_true) & (pred == label_pred)))
    return matrix.tolist()


_METRICS = {
    "mae": mae,
    "rmse": rmse,
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
}


def compute_metric(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float | list[list[int]]:
    metric_name = str(name).lower()
    if metric_name == "confusion_matrix":
        return confusion_matrix(y_true, y_pred)
    if metric_name not in _METRICS:
        raise ValueError(f"Unsupported metric: {name}")
    return _METRICS[metric_name](y_true, y_pred)


def compute_metrics(names: list[str] | tuple[str, ...], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[list[int]]]:
    return {name: compute_metric(name, y_true, y_pred) for name in names}
