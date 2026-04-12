from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15


def create_split_indices(
    n_samples: int,
    *,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    if n_samples < 3:
        raise ValueError("n_samples must be >= 3")

    ratios = np.array([train, val, test], dtype=float)
    if np.any(ratios < 0.0):
        raise ValueError("split ratios must be >= 0")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=int)
    rng.shuffle(indices)

    train_end = int(round(train * n_samples))
    val_end = train_end + int(round(val * n_samples))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("split produced an empty train or test partition")

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }
