# Signal Model

## Array data

Most adaptive functions use snapshots shaped as:

```python
(num_elements, num_snapshots)
```

Each row is one sensor. Each column is one time sample or snapshot.

## Steering model

For a linear array, the steering vector is determined by:

- number of elements
- spacing in wavelengths
- arrival or steering direction `theta`, `phi`

The code uses a narrowband phase progression across the array.

## Covariance model

Adaptive methods estimate the spatial covariance as:

```python
R = XX^H / N
```

where:

- `X` is the snapshot matrix
- `X^H` is the conjugate transpose
- `N` is the number of snapshots

## Assumptions

- narrowband signal model for steering and MVDR/MUSIC
- complex baseband snapshots
- normalized array-pattern outputs for plotting
- optional diagonal loading for numerical stability
