from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from ..tasks import get_task_definition
from .base import BaseEstimator


class RidgeRegressor(BaseEstimator):
    problem_type = "regression"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        x = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        if target.ndim == 1:
            target = target[:, None]
        x_mean = x.mean(axis=0, keepdims=True)
        y_mean = target.mean(axis=0, keepdims=True)
        x_centered = x - x_mean
        y_centered = target - y_mean
        gram = x_centered.T @ x_centered
        self.weights = np.linalg.solve(gram + self.alpha * np.eye(gram.shape[0]), x_centered.T @ y_centered)
        self.bias = y_mean - x_mean @ self.weights
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model is not fitted")
        pred = np.asarray(X, dtype=float) @ self.weights + self.bias
        if pred.shape[1] == 1:
            return pred[:, 0]
        return pred


class NearestCentroidClassifier(BaseEstimator):
    problem_type = "classification"

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestCentroidClassifier":
        x = np.asarray(X, dtype=float)
        target = np.asarray(y).reshape(-1)
        classes = np.unique(target)
        self.classes_ = classes
        self.centroids_ = np.stack([x[target == label].mean(axis=0) for label in classes], axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.centroids_ is None:
            raise RuntimeError("Model is not fitted")
        x = np.asarray(X, dtype=float)
        distances = np.sum((x[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        return self.classes_[np.argmin(distances, axis=1)]


class SklearnEstimatorAdapter(BaseEstimator):
    def __init__(self, estimator: Any, problem_type: str) -> None:
        self.estimator = estimator
        self.problem_type = problem_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnEstimatorAdapter":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X))


def _build_sklearn_model(name: str, params: dict[str, Any], problem_type: str) -> BaseEstimator:
    try:
        ensemble = import_module("sklearn.ensemble")
        linear_model = import_module("sklearn.linear_model")
        neighbors = import_module("sklearn.neighbors")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn support is optional. Install with: pip install 'adaptive-beamforming-toolkit[ml]'"
        ) from exc

    model_name = str(name).lower()
    if model_name == "random_forest":
        estimator_cls = ensemble.RandomForestRegressor if problem_type == "regression" else ensemble.RandomForestClassifier
    elif model_name == "ridge":
        estimator_cls = linear_model.Ridge if problem_type == "regression" else linear_model.LogisticRegression
    elif model_name == "knn":
        estimator_cls = neighbors.KNeighborsRegressor if problem_type == "regression" else neighbors.KNeighborsClassifier
    else:
        raise ValueError(f"Unsupported sklearn model name: {name}")
    return SklearnEstimatorAdapter(estimator_cls(**params), problem_type=problem_type)


def create_model(family: str, name: str, params: dict[str, Any], task_name: str) -> BaseEstimator:
    task = get_task_definition(task_name)
    family_name = str(family).lower()
    model_name = str(name).lower()
    if family_name == "numpy":
        if model_name in {"auto", "ridge", "ridge_regressor"}:
            if task.problem_type != "regression":
                return NearestCentroidClassifier()
            return RidgeRegressor(alpha=float(params.get("alpha", 1.0)))
        if model_name in {"nearest_centroid", "nearest_centroid_classifier", "auto_classifier"}:
            return NearestCentroidClassifier()
        raise ValueError(f"Unsupported numpy model name: {name}")
    if family_name == "sklearn":
        chosen = "ridge" if model_name == "auto" else model_name
        return _build_sklearn_model(chosen, params, task.problem_type)
    raise ValueError(f"Unsupported model family: {family}")
