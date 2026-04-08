# Algorithms

## Implemented in code

### LMS / NLMS / RLS

These adaptive update rules are not implemented in the current codebase.

### Frost

Frost beamforming is not implemented in the current codebase.

### LCMV

LCMV is not implemented as a dedicated solver. The closest related feature is linear null steering, which enforces simple directional constraints.

### MVDR

Implemented in `algorithms.adaptive.mvdr_weights(...)`.

- Input: sample covariance matrix and steering vector
- Goal: unit gain in the look direction with minimum output power elsewhere
- Note: diagonal loading is supported for stability

### MUSIC

Implemented in `algorithms.adaptive.doa_music_linear(...)`.

- Input: array snapshots and scan angles
- Goal: estimate arrival directions from the noise subspace
- Output: pseudo-spectrum and estimated peak angles

### Null Steering

Implemented in `core.beamforming.null_steering_weights_linear(...)`.

- Goal: keep gain in the desired direction while placing deep nulls at specified interferers
- Use case: deterministic interference suppression with known directions

## Supporting models

- Near-field focusing
- Digital / analog / hybrid weight synthesis
- Wideband beam-squint analysis
- Element-pattern and mutual-coupling impairments
