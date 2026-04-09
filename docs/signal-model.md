# Signal Model

This page defines the mathematical model behind the adaptive parts of the repository and links that notation to the actual array and IQ helpers.

## Snapshot Representation

Most adaptive functions operate on a complex snapshot matrix with shape

```python
(num_elements, num_snapshots)
```

This means:

- each row corresponds to one array element
- each column corresponds to one time sample or snapshot

If `M` is the number of elements and `K` is the number of snapshots, then

```math
\mathbf{X} = [\mathbf{x}[1], \mathbf{x}[2], \dots, \mathbf{x}[K]] \in \mathbb{C}^{M \times K}
```

with `x[k]` the `M x 1` array observation at snapshot `k`.

## Narrowband Array Model

The narrowband complex-baseband receive model used by MVDR and MUSIC is

```math
\mathbf{x}[k] = \sum_{q=1}^{Q} \mathbf{a}(\theta_q,\phi_q)s_q[k] + \mathbf{n}[k]
```

where:

- `Q` is the number of sources
- `a(theta_q, phi_q)` is the steering vector of source `q`
- `s_q[k]` is the complex source amplitude at snapshot `k`
- `n[k]` is additive noise

This model assumes:

- narrowband signals
- phase coherence across the aperture
- a common carrier wavelength used to normalize spacing

## ULA Steering Vector

For a ULA centered at the origin with element positions `x_n` measured in wavelengths, the repository uses the direction cosine

```math
u(\theta,\phi) = \sin(\theta)\cos(\phi)
```

and the steering vector entries

```math
a_n(\theta,\phi) = e^{j 2 \pi x_n u(\theta,\phi)}
```

with centered positions

```math
x_n = \left(n - \frac{M-1}{2}\right)\frac{d}{\lambda}, \quad n=0,\dots,M-1
```

This is the exact model implemented by `algorithms.adaptive.linear_steering_vector(...)` and consistent with the independent references used in the test suite.

## Beamformer Output

Given weights `w`, the beamformer output is

```math
y[k] = \mathbf{w}^H \mathbf{x}[k]
```

For pattern evaluation, one often looks at the deterministic response toward a scan direction:

```math
H(\theta,\phi) = \mathbf{w}^H \mathbf{a}(\theta,\phi)
```

The plotted array factor is the normalized magnitude `|H(theta, phi)|`.

## Sample Covariance Matrix

Adaptive methods in this repository estimate the covariance matrix from snapshots as

```math
\hat{\mathbf{R}} = \frac{1}{K}\mathbf{X}\mathbf{X}^H
```

where `K` is the number of snapshots. In code, this is `estimate_covariance_matrix(...)`.

Optional diagonal loading adds numerical robustness:

```math
\hat{\mathbf{R}}_\delta = \hat{\mathbf{R}} + \delta \mathbf{I}
```

Diagonal loading is useful when:

- the covariance estimate is poorly conditioned
- the snapshot count is limited
- steering mismatch or model mismatch is expected

## Signal Simulation in `data.iq`

The helper `simulate_array_iq(...)` generates synthetic snapshots by summing several steering vectors, each excited by a random complex Gaussian source waveform, and then adding complex white noise.

At a high level, the simulator constructs

```math
\mathbf{X} = \sum_{q=1}^{Q} \alpha_q \mathbf{a}(\theta_q,\phi_q)\mathbf{s}_q^T + \mathbf{N}
```

where:

- `alpha_q = 10^(SNR_q / 20)` is the source amplitude scaling used by the current implementation
- `s_q` is a random source sequence
- `N` is complex Gaussian noise

This is a convenient stochastic model for testing covariance-based algorithms, even though it is not intended as a calibrated propagation model.

## Planar and Extended Models

The repository now includes both ULA and planar steering helpers for adaptive processing. The additional models in `core.advanced_models` extend the same general framework to:

- near-field focusing
- wideband response evaluation
- element-pattern weighting
- mutual coupling
- analog and hybrid architecture approximations

These models remain simplified and are meant for comparative simulation rather than full-wave electromagnetic prediction.

## Assumptions and Boundaries

The current signal model is intentionally compact. Important assumptions are:

- narrowband complex-baseband representation for adaptive processing
- optional per-frequency-bin wideband processing for frequency-domain MVDR
- stationary covariance over the snapshot batch
- no calibration errors unless impairments are explicitly added
- no coherent multipath or model-order estimation logic beyond the provided MUSIC input parameter
- no time-delay beamforming for true wideband waveform preservation

## Mapping to Code

- `algorithms.adaptive.linear_steering_vector(...)`: ULA steering vector
- `algorithms.adaptive.planar_steering_vector(...)`: planar steering vector
- `algorithms.adaptive.estimate_covariance_matrix(...)`: sample covariance estimator
- `algorithms.adaptive.mvdr_weights(...)`: MVDR/Capon weights from covariance plus steering vector
- `algorithms.adaptive.lms_weights(...)`: supervised LMS adaptation
- `algorithms.adaptive.nlms_weights(...)`: supervised normalized LMS adaptation
- `algorithms.adaptive.rls_weights(...)`: supervised RLS adaptation
- `algorithms.adaptive.wideband_mvdr_weights(...)`: per-bin MVDR for frequency-domain snapshots
- `algorithms.adaptive.doa_music_linear(...)`: MUSIC scan over a theta grid
- `data.iq.simulate_array_iq(...)`: synthetic snapshot generator
- `data.iq.simulate_mimo_iq(...)`: virtual-array MIMO snapshot generator
- `data.iq.simulate_polarimetric_array_iq(...)`: stacked polarimetric snapshot generator
- `data.iq.beamform_iq(...)`: direct weighted combination of snapshots

## References

- J. Capon, "High-resolution frequency-wavenumber spectrum analysis," *Proceedings of the IEEE*, 1969. https://ieeexplore.ieee.org/document/1449208
- R. O. Schmidt, "Multiple emitter location and signal parameter estimation," *IEEE Transactions on Antennas and Propagation*, 1986. https://ieeexplore.ieee.org/document/1143830/
- C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed., Wiley. https://bcs.wiley.com/he-bcs/Books?action=contents&bcsId=9777&itemId=1118642066
