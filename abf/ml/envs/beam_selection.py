from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from abf.algorithms import estimate_covariance_matrix, linear_steering_vector, planar_steering_vector
from abf.data import simulate_array_iq_components
from simulations.config import SourceConfig

from ..config import EnvironmentConfig, ExperimentConfig, SamplingConfig, load_experiment_config
from ..features import build_feature_vector


@dataclass
class DiscreteActionSpace:
    n: int
    rng: np.random.Generator

    def sample(self) -> int:
        return int(self.rng.integers(0, self.n))


class BeamSelectionEnv:
    def __init__(self, config: str | ExperimentConfig, seed: int | None = None) -> None:
        self.experiment = load_experiment_config(config) if not isinstance(config, ExperimentConfig) else config
        self.seed = int(self.experiment.dataset.seed if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)
        self.env_config: EnvironmentConfig = self.experiment.env
        self.action_space = DiscreteActionSpace(n=len(self.env_config.beam_angles_deg), rng=self.rng)
        self._step_count = 0
        self._current_observation: np.ndarray | None = None
        self._current_scores: np.ndarray | None = None
        self._best_action = 0
        self._desired_theta_deg = 0.0

    def _steering(self, scenario, theta_deg: float) -> np.ndarray:
        if scenario.array.geometry == "ula":
            assert scenario.array.spacing_lambda is not None
            return linear_steering_vector(scenario.array.num_elements, float(scenario.array.spacing_lambda), theta_deg)
        assert scenario.array.num_x is not None
        assert scenario.array.num_y is not None
        assert scenario.array.spacing_x_lambda is not None
        assert scenario.array.spacing_y_lambda is not None
        return planar_steering_vector(
            scenario.array.num_x,
            scenario.array.num_y,
            float(scenario.array.spacing_x_lambda),
            float(scenario.array.spacing_y_lambda),
            theta_deg,
            0.0,
        )

    def _sample_scenario(self):
        sampling: SamplingConfig = self.experiment.dataset.sampling
        base = self.experiment.scenario
        theta = base.desired_source.theta_deg
        if sampling.desired_theta_deg is not None:
            theta = float(self.rng.uniform(sampling.desired_theta_deg.min, sampling.desired_theta_deg.max))
        count = int(self.rng.integers(sampling.interference_count[0], sampling.interference_count[1] + 1))
        interferers = []
        for _ in range(count):
            theta_i = theta + 20.0
            if sampling.interference_theta_deg is not None:
                theta_i = float(self.rng.uniform(sampling.interference_theta_deg.min, sampling.interference_theta_deg.max))
            snr_i = 5.0
            if sampling.interference_snr_db is not None:
                snr_i = float(self.rng.uniform(sampling.interference_snr_db.min, sampling.interference_snr_db.max))
            interferers.append(SourceConfig(theta_deg=theta_i, phi_deg=0.0, snr_db=snr_i))
        return self.experiment.scenario.__class__(
            name=base.name,
            seed=base.seed,
            snapshots=base.snapshots,
            array=base.array,
            desired_source=base.desired_source.__class__(theta_deg=theta, phi_deg=base.desired_source.phi_deg, snr_db=base.desired_source.snr_db),
            interference_sources=tuple(interferers),
            algorithm=base.algorithm,
            sweep=base.sweep,
            output=base.output,
        )

    def _build_episode(self) -> tuple[np.ndarray, dict[str, Any]]:
        scenario = self._sample_scenario()
        self._desired_theta_deg = float(scenario.desired_source.theta_deg)
        source_thetas = np.array([scenario.desired_source.theta_deg, *[src.theta_deg for src in scenario.interference_sources]], dtype=float)
        source_phis = np.array([scenario.desired_source.phi_deg, *[src.phi_deg for src in scenario.interference_sources]], dtype=float)
        source_snr = np.array([scenario.desired_source.snr_db, *[src.snr_db for src in scenario.interference_sources]], dtype=float)
        components = simulate_array_iq_components(
            num_elements=scenario.array.num_elements,
            num_snapshots=scenario.snapshots,
            spacing_lambda=scenario.array.spacing_lambda,
            source_thetas_deg=source_thetas,
            source_phis_deg=source_phis,
            source_snr_db=source_snr,
            random_seed=int(self.rng.integers(0, 2**31 - 1)),
            geometry=scenario.array.geometry,
            num_x=scenario.array.num_x,
            num_y=scenario.array.num_y,
            spacing_x_lambda=scenario.array.spacing_x_lambda,
            spacing_y_lambda=scenario.array.spacing_y_lambda,
        )
        obs = build_feature_vector(
            feature_type=self.env_config.feature_type,
            components=components,
            config=scenario,
            task_params={"beam_angles_deg": self.env_config.beam_angles_deg},
        )
        covariance = estimate_covariance_matrix(components["snapshots"])
        scores = []
        for angle in self.env_config.beam_angles_deg:
            steer = self._steering(scenario, float(angle))
            scores.append(float(np.real(np.vdot(steer, covariance @ steer))))
        self._current_scores = np.asarray(scores, dtype=float)
        self._best_action = int(np.argmax(self._current_scores))
        info = {
            "desired_theta_deg": self._desired_theta_deg,
            "best_action": self._best_action,
            "beam_angles_deg": list(self.env_config.beam_angles_deg),
        }
        return np.asarray(obs), info

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count = 0
        self._current_observation, info = self._build_episode()
        return np.asarray(self._current_observation), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._current_observation is None or self._current_scores is None:
            raise RuntimeError("Call reset() before step()")
        action_idx = int(action)
        if action_idx < 0 or action_idx >= self.action_space.n:
            raise ValueError(f"action must be in [0, {self.action_space.n - 1}]")

        self._step_count += 1
        chosen_score = float(self._current_scores[action_idx])
        best_score = float(np.max(self._current_scores))
        reward_mode = self.env_config.reward_mode.lower()
        if reward_mode == "hit":
            reward = float(action_idx == self._best_action)
        else:
            reward = float(chosen_score / max(best_score, 1e-12))
        terminated = self._step_count >= self.env_config.episode_length
        truncated = False
        info = {
            "desired_theta_deg": self._desired_theta_deg,
            "best_action": self._best_action,
            "chosen_score": chosen_score,
            "best_score": best_score,
        }
        return np.asarray(self._current_observation), reward, terminated, truncated, info
