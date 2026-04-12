# ML Datasets

The dataset layer turns simulator runs into `(X, y, meta)` bundles with deterministic seeds and reproducible split indices.

## Public API

- `abf.ml.datasets.generate_dataset(...)`
- `abf.ml.datasets.save_dataset(...)`
- `abf.ml.datasets.load_dataset(...)`

## Feature Types

- `raw_iq`: flattened complex snapshots
- `real_imag`: real and imaginary snapshot channels stacked into one real vector
- `covariance`: flattened complex covariance matrix
- `covariance_real_imag`: real and imaginary covariance channels stacked into one real vector
- `spectrum`: beamspace spectrum over a configured angle scan

## Target Types

- `doa_regression`
- `doa_classification`
- `beam_selection`
- `interference_detection`
- `weight_regression`

## Reproducibility

- every dataset is seeded explicitly
- per-sample seeds are stored in metadata
- split indices are deterministic
- experiment configuration is captured in dataset metadata

## Minimal Example

```python
from abf.ml.datasets import generate_dataset, save_dataset

dataset = generate_dataset(
    "config/ml/doa_regression.yaml",
    n_samples=200,
    seed=7,
)
save_dataset(dataset, "outputs/datasets/example.npz")
```
