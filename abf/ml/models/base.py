from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class BaseEstimator:
    problem_type: str = "regression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEstimator":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def save_model(model: BaseEstimator, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(model, handle)
    return target


def load_model(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
