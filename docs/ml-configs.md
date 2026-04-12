# ML Configs

The ML layer extends the existing YAML workflow instead of replacing it.

## Main Sections

- `experiment`
- `task`
- `dataset`
- `split`
- `model`
- `metrics`
- `baselines`
- `output`
- `env`
- `scenario`

## Example Shape

```yaml
experiment:
  name: doa_regression_baseline

task:
  name: doa_regression

dataset:
  n_samples: 1000
  seed: 42
  feature_type: covariance_real_imag
  target_type: doa_regression

split:
  train: 0.7
  val: 0.15
  test: 0.15

model:
  family: numpy
  name: ridge_regressor

metrics: [mae, rmse]
baselines: [music]
```

The `scenario` block reuses the existing simulation schema, so array geometry, sources, and sweep definitions stay consistent with the current toolkit.
