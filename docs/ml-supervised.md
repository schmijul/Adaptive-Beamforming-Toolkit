# ML Supervised Experiments

The supervised workflow centers on `abf.ml.train.run_experiment(...)`. It generates a dataset, creates split indices, fits a model, evaluates metrics, compares configured baselines, and writes artifacts.

## Public API

- `abf.ml.train.run_experiment(...)`
- `abf.ml.evaluate.evaluate_model(...)`

## Built-In Model Families

- `family: numpy`
  - `ridge_regressor`
  - `nearest_centroid`
- `family: sklearn`
  - `ridge`
  - `random_forest`
  - `knn`

The `sklearn` family is optional and requires `pip install -e ".[ml]"`.

## Example

```python
from abf.ml.train import run_experiment

result = run_experiment("config/ml/doa_regression.yaml")
print(result.metrics)
print(result.baselines)
```

## Results

The experiment runner writes `results.json` and, when enabled, `dataset.npz` and `model.pkl` under the configured output directory.
