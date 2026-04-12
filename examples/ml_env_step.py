from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abf.ml.envs import BeamSelectionEnv


def main() -> None:
    env = BeamSelectionEnv("config/rl/beam_selection.yaml", seed=1)
    observation, info = env.reset()
    print("reset", observation.shape, info)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, step_info = env.step(action)
    print("step", action, reward, terminated, truncated, step_info, observation.shape)


if __name__ == "__main__":
    main()
