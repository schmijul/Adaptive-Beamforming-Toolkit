from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

import abf
from abf.ml import generate_dataset, load_dataset, load_experiment_config, run_experiment, save_dataset
from abf.ml.envs import BeamSelectionEnv


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_config(payload: dict, path: Path) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_abf_namespace_exposes_ml_module() -> None:
    assert abf.ml is not None


def test_load_experiment_config_parses_ml_sections() -> None:
    config = load_experiment_config("config/ml/doa_regression.yaml")
    assert config.task_name == "doa_regression"
    assert config.dataset.feature_type == "covariance_real_imag"
    assert config.output.dir.endswith("doa_regression_baseline")


def test_generate_dataset_is_deterministic() -> None:
    first = generate_dataset("config/ml/doa_regression.yaml", n_samples=24, seed=5)
    second = generate_dataset("config/ml/doa_regression.yaml", n_samples=24, seed=5)
    assert np.allclose(first.X, second.X)
    assert np.allclose(first.y, second.y)
    assert np.array_equal(first.split_indices["train"], second.split_indices["train"])


def test_generate_dataset_shapes_and_splits() -> None:
    dataset = generate_dataset("config/ml/beam_selection.yaml", n_samples=30, seed=3)
    assert dataset.X.shape[0] == 30
    assert dataset.y.shape == (30,)
    assert dataset.split_indices["train"].size > 0
    assert dataset.split_indices["test"].size > 0


def test_dataset_serialization_roundtrip(tmp_path) -> None:
    dataset = generate_dataset("config/ml/doa_regression.yaml", n_samples=16, seed=9)
    path = save_dataset(dataset, tmp_path / "dataset.npz")
    loaded = load_dataset(path)
    assert np.allclose(dataset.X, loaded.X)
    assert np.allclose(dataset.y, loaded.y)
    assert loaded.meta["task"] == "doa_regression"


def test_run_experiment_writes_artifacts(tmp_path) -> None:
    payload = _load_yaml("config/ml/doa_regression.yaml")
    payload["dataset"]["n_samples"] = 40
    payload["dataset"]["export_path"] = str(tmp_path / "dataset.npz")
    payload["output"]["dir"] = str(tmp_path / "experiment")
    path = _write_config(payload, tmp_path / "experiment.yaml")

    result = run_experiment(path)
    assert "mae" in result.metrics
    assert Path(result.results_path).exists()
    assert Path(result.model_path or "").exists()
    assert Path(result.dataset_path or "").exists()


def test_environment_reset_and_step() -> None:
    env = BeamSelectionEnv("config/rl/beam_selection.yaml", seed=4)
    obs, info = env.reset()
    assert obs.ndim == 1
    assert "best_action" in info
    obs2, reward, terminated, truncated, step_info = env.step(0)
    assert obs2.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert "chosen_score" in step_info


def test_cli_dataset_train_and_env_demo(tmp_path) -> None:
    payload = _load_yaml("config/ml/doa_regression.yaml")
    payload["dataset"]["n_samples"] = 32
    payload["dataset"]["export_path"] = str(tmp_path / "dataset.npz")
    payload["output"]["dir"] = str(tmp_path / "train")
    exp_path = _write_config(payload, tmp_path / "doa.yaml")

    dataset_cmd = subprocess.run(
        [sys.executable, "-m", "abf", "dataset", "--config", str(exp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    train_cmd = subprocess.run(
        [sys.executable, "-m", "abf", "train", "--config", str(exp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    dataset_output = json.loads(dataset_cmd.stdout)
    train_output = json.loads(train_cmd.stdout)
    assert dataset_output["mode"] == "dataset"
    assert train_output["mode"] == "train"
    assert Path(train_output["results_path"]).exists()

    env_cmd = subprocess.run(
        [sys.executable, "-m", "abf", "env-demo", "--config", "config/rl/beam_selection.yaml", "--steps", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    env_output = json.loads(env_cmd.stdout)
    assert env_output["mode"] == "env-demo"
    assert env_output["transitions"]
