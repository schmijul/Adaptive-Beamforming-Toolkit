# Notes

## Current Scope

The repository is intentionally focused on compact, test-backed beamforming workflows:

- ideal ULA and planar-array factors
- deterministic steering and null steering
- selected advanced models for near-field, wideband, and impairments
- covariance-based MVDR, LMS, NLMS, RLS, and MUSIC utilities
- lightweight IQ simulation plus planar, MIMO, and polarimetric helpers
- a dashboard and config-driven CLI

## Simplifications

Several models are deliberately simplified for clarity and speed:

- the adaptive routines are still based on narrowband or per-frequency-bin models
- wideband support includes subband MVDR helpers, but not true time-delay beamforming or STAP
- the impairment model is first-order and not a full calibration framework
- the simulation runner currently supports `ula` and `planar` geometries
- the simulation runner currently supports `conventional`, `mvdr`, `lms`, `nlms`, and `rls`
- MIMO and polarimetric workflows are Python-API helpers rather than first-class CLI modes

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
- a real-time embedded beamforming runtime

## Recommended Extensions

If the toolkit grows further, the next technically coherent additions would be:

- general LCMV constraints
- model-order selection for MUSIC
- true-time-delay wideband beamforming
- calibration and array-error estimation utilities
