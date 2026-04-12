# ML Environments

`abf.ml.envs.BeamSelectionEnv` wraps the simulator as a lightweight decision-making environment for beam-selection experiments.

## Interface

- `reset() -> observation, info`
- `step(action) -> observation, reward, terminated, truncated, info`
- `action_space.sample()`

## Observation, Action, Reward

- observation: simulator-derived feature vector, such as `covariance_real_imag`
- action: discrete beam index from `env.beam_angles_deg`
- reward:
  - `gain`: normalized beam response relative to the best configured beam
  - `hit`: `1.0` for the best beam and `0.0` otherwise

## Example

```python
from abf.ml.envs import BeamSelectionEnv

env = BeamSelectionEnv("config/rl/beam_selection.yaml", seed=1)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```
