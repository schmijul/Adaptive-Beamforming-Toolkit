# Notes

## Current Scope

The repository is intentionally focused on compact, test-backed beamforming workflows:

- ideal ULA and planar-array factors
- deterministic steering and null steering
- selected advanced models for near-field, wideband, and impairments
- covariance-based MVDR and MUSIC utilities
- lightweight IQ simulation and comparison helpers
- a dashboard and config-driven CLI

## Simplifications

Several models are deliberately simplified for clarity and speed:

- the adaptive routines use a narrowband ULA steering model
- the wideband model demonstrates beam squint, not true time-delay beamforming
- the impairment model is first-order and not a full calibration framework
- the simulation runner currently supports only `ula` geometry
- the simulation runner currently supports only `conventional` and `mvdr`

## What The Toolkit Is Good For

This repository is a good fit for:

- learning and teaching array-processing fundamentals
- quick comparisons between classical beamforming methods
- building intuition about spacing, tapering, nulls, and squint
- generating reproducible simulation artifacts from YAML scenarios

## What It Is Not

The current codebase should not be mistaken for:

- a hardware-control stack
- a calibrated measurement-processing framework
- a full electromagnetic solver
- a comprehensive research benchmark suite covering modern adaptive methods

## Recommended Extensions

If the toolkit grows further, the next technically coherent additions would be:

- general LCMV constraints
- model-order selection for MUSIC
- planar adaptive steering helpers
- true-time-delay wideband beamforming
- calibration and array-error estimation utilities
