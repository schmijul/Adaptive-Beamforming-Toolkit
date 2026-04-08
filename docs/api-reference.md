# API Reference

## `core.beamforming`

- `amplitude_taper(num_elements, taper_name)`
- `array_factor_linear(...)`
- `array_factor_linear_from_weights(...)`
- `array_factor_planar(...)`
- `element_positions_linear(...)`
- `element_positions_planar(...)`
- `steering_weights_linear(...)`
- `steering_weights_planar(...)`
- `null_steering_weights_linear(...)`

Purpose: ideal array geometry, steering, nulling, and normalized pattern evaluation.

## `core.advanced_models`

- `steering_weights_near_field_linear(...)`
- `array_factor_linear_near_field(...)`
- `array_factor_linear_field_mode(...)`
- `synthesize_beamforming_architecture(...)`
- `wideband_array_factor_linear(...)`
- `build_mutual_coupling_matrix(...)`
- `element_pattern_gain(...)`
- `array_factor_linear_with_impairments(...)`
- `ArchitectureWeights`

Purpose: non-ideal or extended beamforming models.

## `algorithms.adaptive`

- `estimate_covariance_matrix(...)`
- `linear_steering_vector(...)`
- `mvdr_weights(...)`
- `music_spectrum(...)`
- `doa_music_linear(...)`

Purpose: adaptive beamforming and direction-of-arrival estimation.

## `data.iq`

- `load_iq_samples(path)`
- `simulate_array_iq(...)`
- `beamform_iq(...)`
- `compare_sim_vs_measurement(...)`

Purpose: load or synthesize IQ data, beamform it, and compare simulation against measurements.

## `visualize.plots`

- `build_elevation_cut(...)`
- `build_heatmap(...)`
- `build_pattern_3d(...)`
- `build_weights_plot(...)`

Purpose: Plotly figures used by the dashboard.
