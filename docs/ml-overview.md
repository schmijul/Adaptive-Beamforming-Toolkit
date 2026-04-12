# ML Overview

`abf.ml` adds a compact experimentation layer on top of the existing simulator and classical beamforming stack. It does not replace the current API. Instead, it wraps deterministic snapshot generation, feature building, task-specific labels, experiment splits, metrics, and baseline comparison in a repeatable workflow.

## Start Here in 5 Minutes

Install the base package plus the optional ML extra if you want scikit-learn models:

```bash
pip install -e .
pip install -e ".[ml]"
```

Generate a dataset:

```python
from abf.ml.datasets import generate_dataset

dataset = generate_dataset("config/ml/doa_regression.yaml")
print(dataset.X.shape, dataset.y.shape)
```

Run a supervised experiment:

```python
from abf.ml.train import run_experiment

result = run_experiment("config/ml/doa_regression.yaml")
print(result.metrics)
```

Step through an environment:

```python
from abf.ml.envs import BeamSelectionEnv

env = BeamSelectionEnv("config/rl/beam_selection.yaml", seed=1)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Included Scope

- deterministic dataset generation from simulator-backed scenarios
- explicit feature types such as IQ, covariance, and beamspace spectrum
- explicit task types such as DoA regression and beam selection
- lightweight training and evaluation helpers
- baseline comparison against classical methods where applicable
- a Gymnasium-style environment wrapper without making Gymnasium mandatory

## Intentional Non-Goals

- no deep-learning training framework
- no large model catalog
- no replacement for the existing simulation runner
- no requirement to install heavyweight ML libraries for base usage
