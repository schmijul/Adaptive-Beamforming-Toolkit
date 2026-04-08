# API Reference

This is a compact reference for the public Python entry points used throughout the repository.

## Supported Import Root

External users should prefer the `abf.*` namespace:

- `abf.core`
- `abf.algorithms`
- `abf.data`
- `abf.simulations`
- `abf.visualize`

The legacy roots `core`, `algorithms`, `data`, and `simulations` remain available for compatibility with older code and internal tests.

## `abf.core`

This module re-exports the native C++ extension and is the main entry point for idealized array geometry and pattern evaluation.

- `amplitude_taper(num_elements, taper_name)`
  Returns a real taper window such as uniform, Hamming, or Taylor.
- `element_positions_linear(num_elements, spacing_lambda)`
  Returns centered ULA element positions in wavelength units.
- `element_positions_planar(num_x, num_y, spacing_x_lambda, spacing_y_lambda)`
  Returns centered planar-array element positions.
- `steering_weights_linear(...)`
  Builds ULA steering weights and associated amplitude/phase components.
- `steering_weights_planar(...)`
  Builds planar-array steering weights.
- `null_steering_weights_linear(...)`
  Solves a linearly constrained deterministic null-steering problem for a ULA.
- `array_factor_linear(...)`
  Evaluates the normalized ULA response on a `(theta, phi)` grid.
- `array_factor_planar(...)`
  Evaluates the normalized planar-array response on a `(theta, phi)` grid.
- `array_factor_linear_from_weights(...)`
  Evaluates the ULA response for a user-supplied complex weight vector.

Typical outputs from the array-factor functions include:

- `response`
- `magnitude`
- `magnitude_db`
- `weights`
- `positions_lambda`

## `abf.core` advanced models

This module extends the ideal far-field model with simplified non-ideal or non-standard scenarios.

- `steering_weights_near_field_linear(...)`
  Builds a distance-based near-field focusing law for a ULA.
- `array_factor_linear_near_field(...)`
  Evaluates the near-field focused response.
- `array_factor_linear_field_mode(...)`
  Convenience wrapper that switches between `far` and `near` field modes.
- `ArchitectureWeights`
  Dataclass containing synthesized architecture weights and their RF/baseband components.
- `synthesize_beamforming_architecture(...)`
  Maps ideal weights to digital, analog, or hybrid approximations.
- `wideband_array_factor_linear(...)`
  Evaluates a fixed-phase-steered array over multiple frequencies to show beam squint.
- `build_mutual_coupling_matrix(...)`
  Constructs a simple distance-decay mutual-coupling matrix.
- `element_pattern_gain(...)`
  Returns isotropic, cosine, or cardioid element gain values over an angular grid.
- `array_factor_linear_with_impairments(...)`
  Applies element patterns and coupling to the ideal ULA response.

## `abf.algorithms`

This module contains covariance-based adaptive processing helpers.

- `estimate_covariance_matrix(snapshots, diagonal_loading=0.0)`
  Computes the sample covariance from a snapshot matrix with shape `(num_elements, num_snapshots)`.
- `linear_steering_vector(num_elements, spacing_lambda, theta_deg, phi_deg=0.0)`
  Returns the narrowband ULA steering vector used by the adaptive routines.
- `mvdr_weights(covariance_matrix, steering_vector, diagonal_loading=1e-3)`
  Computes MVDR/Capon weights.
- `music_spectrum(covariance_matrix, scan_manifold, num_sources)`
  Evaluates the MUSIC pseudospectrum for a scan manifold.
- `doa_music_linear(...)`
  Builds a ULA scan manifold over `theta_scan_deg`, computes the MUSIC spectrum, and returns peak estimates.

Typical output keys from `doa_music_linear(...)`:

- `theta_scan_deg`
- `spectrum`
- `estimated_thetas_deg`
- `covariance_matrix`

## `abf.data`

This module covers simple IQ ingest, synthesis, beamforming, and comparison.

- `load_iq_samples(path)`
  Loads IQ data from `.npy`, `.npz`, `.csv`, or `.txt`.
- `simulate_array_iq(...)`
  Generates complex array snapshots from one or more synthetic sources plus noise.
- `beamform_iq(iq_snapshots, weights)`
  Applies complex beamforming weights to snapshot data.
- `compare_sim_vs_measurement(simulated, measured)`
  Returns scalar comparison metrics such as MSE, NMSE in dB, and normalized correlation.

## `abf.simulations`

The config-driven simulation runner is the backend behind the CLI.

- `load_scenario_config(path)`
  Parses and validates a YAML scenario file.
- `run_single_simulation(config)`
  Runs one deterministic simulation and writes a JSON artifact.
- `run_monte_carlo(config, runs)`
  Repeats the scenario over multiple seeds and writes a Monte Carlo summary.

Current supported runner scope:

- `array.geometry`: `ula`
- `algorithm.name`: `conventional`, `mvdr`

Common top-level return keys:

- `run_single_simulation(...)`: `mode`, `created_at_utc`, `config`, `result`
- `run_monte_carlo(...)`: `mode`, `created_at_utc`, `config`, `summary`, `runs`

## `abf.visualize`

These functions build the Plotly figures used by the dashboard and optional simulation outputs.

- `build_elevation_cut(theta_deg, magnitude_db, theta_steer_deg)`
- `build_heatmap(theta_deg, phi_deg, magnitude_db)`
- `build_pattern_3d(theta_deg, phi_deg, magnitude)`
- `build_weights_plot(positions_lambda, amplitudes, phase_weights)`

## `ui.dash_app`

- `create_app()`
  Builds the Dash application and wires the interactive controls to the underlying array-factor functions.

## CLI Entry Point

The console script `abf` is defined in `pyproject.toml` and implemented in `abf_cli.py`.

Supported subcommands:

- `abf dashboard`
- `abf simulate --config <path>`
- `abf montecarlo --config <path> --runs <n>`
- `abf gallery --config <path>`
