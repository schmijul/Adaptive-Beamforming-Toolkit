# ML Extension Architecture Plan

## Goals

- Add a thin `abf.ml` experimentation layer without disturbing the existing `abf.*` simulator and beamforming API.
- Reuse the current simulator, steering, covariance, and classical algorithm helpers instead of reimplementing them.
- Keep optional machine-learning dependencies out of the base install where possible.

## Existing Reuse Points

- `data.iq.simulate_array_iq_components()` already provides deterministic per-run snapshots, source signals, steering vectors, and noise.
- `algorithms.adaptive` already provides covariance estimation, steering vectors, MVDR weights, and MUSIC support needed for learned-method baselines.
- `simulations.config` already defines the scenario schema and validation rules for array geometry and source layout.
- `abf.cli` and `abf.services.runtime` already define the public CLI pattern and service boundary used for config-driven execution.

## Proposed ML Layer

- `abf.ml.config`
  - Parse YAML experiment files with sections for `experiment`, `scenario`, `dataset`, `split`, `model`, `metrics`, `baselines`, `output`, and optional `env`.
  - Reuse the scenario validator for embedded simulation settings.
- `abf.ml.datasets`
  - Main entry points: `generate_dataset()`, `save_dataset()`, `load_dataset()`.
  - Return a dataclass container with `X`, `y`, `meta`, and split indices/views.
  - Generate per-sample scenario variants by sampling configured angle/SNR/count ranges around a validated base scenario.
- `abf.ml.features`
  - Centralize explicit feature builders for raw IQ, real/imag snapshots, covariance, covariance real/imag, and beamspace spectrum.
- `abf.ml.labels`
  - Centralize label construction for DoA regression/classification, beam selection, interference detection, and weight regression.
- `abf.ml.tasks`
  - Register supported task names, valid feature/label pairings, default metrics, and baseline compatibility.
- `abf.ml.models`
  - Provide a small estimator protocol plus lightweight built-in models for regression/classification.
  - Add optional scikit-learn adapter support when the extra is installed.
- `abf.ml.train` and `abf.ml.evaluate`
  - Run repeatable supervised experiments, compute metrics, compare classical baselines, and export artifacts.
- `abf.ml.envs`
  - Provide a lightweight beam-selection environment with Gymnasium-like `reset()` and `step()` methods without making Gymnasium mandatory.

## Delivery Order

1. Implement config parsing plus dataset/task foundations.
2. Deliver one polished supervised vertical slice for `doa_regression`.
3. Extend task coverage to classification-style targets and baseline hooks.
4. Add the beam-selection environment wrapper and CLI demonstration entry point.
5. Update packaging, docs, examples, and tests around the final public surface.

## Compatibility Rules

- Preserve the existing simulation runner and CLI commands unchanged.
- Add new CLI subcommands instead of changing current semantics.
- Keep public imports intuitive through `abf.ml.*` and `abf.ml.envs`.
- Document any intentionally optional dependencies through extras rather than implicit imports.
