# Algorithms

This document describes the currently implemented methods, including the signal model and the underlying formulas.

## 1) Array and Signal Model (Narrowband)

For a linear array along the x-axis with element positions \(x_n\) measured in wavelengths:

\[
u(\theta,\phi)=\sin(\theta)\cos(\phi), \quad
a_n(\theta,\phi)=e^{j2\pi x_n u(\theta,\phi)}
\]

The array output with weights \(w_n\) is:

\[
y(\theta,\phi)=\sum_{n=1}^{N} w_n\,a_n(\theta,\phi), \quad
AF(\theta,\phi)=|y(\theta,\phi)|
\]

In code:
- `core.beamforming.array_factor_linear(...)`
- `core.beamforming.array_factor_planar(...)`

## 2) Conventional Steering and Tapering

For a look direction \((\theta_0,\phi_0)\):

\[
w_n = \alpha_n \, e^{-j2\pi x_n u(\theta_0,\phi_0)}
\]

\(\alpha_n\) is the amplitude taper (uniform, Hamming, Taylor).

In code:
- `core.beamforming.steering_weights_linear(...)`
- `core.beamforming.amplitude_taper(...)`

## 3) Null Steering (Deterministic Constraints)

The goal is to find weights satisfying:

\[
w^H a(\theta_0,\phi_0)=1, \quad
w^H a(\theta_k,\phi_k)=0 \;\; \forall k\in\{1,\dots,K\}
\]

The implementation builds a constraint matrix and solves a linear system in the constraint space.

In code:
- `core.beamforming.null_steering_weights_linear(...)`

## 4) MVDR (Capon)

Optimization problem:

\[
\min_w \; w^H R w
\quad \text{s.t.} \quad
w^H a_0 = 1
\]

Closed-form solution:

\[
w_{\text{MVDR}} = \frac{R^{-1}a_0}{a_0^H R^{-1} a_0}
\]

With optional diagonal loading:

\[
R_\delta = R + \delta I
\]

In code:
- `algorithms.adaptive.mvdr_weights(...)`
- `algorithms.adaptive.estimate_covariance_matrix(...)`

## 5) MUSIC (DoA Estimation)

Eigen-decomposition of the covariance matrix:

\[
R = E_s \Lambda_s E_s^H + E_n \Lambda_n E_n^H
\]

Pseudo-spectrum:

\[
P_{\text{MUSIC}}(\theta,\phi) =
\frac{1}{a^H(\theta,\phi)\,E_nE_n^H\,a(\theta,\phi)}
\]

Peaks of \(P_{\text{MUSIC}}\) provide the estimated directions of arrival.

In code:
- `algorithms.adaptive.music_spectrum(...)`
- `algorithms.adaptive.doa_music_linear(...)`

## 6) Near-Field Focusing

In the near field, the phase depends on the true distance \(r_n\) from each element to the focus point:

\[
r_n = \|p_{\text{focus}} - p_n\|, \quad
w_n \propto e^{-j2\pi r_n}
\]

In the far field, this is replaced by the plane-wave approximation, which produces a linear phase progression over \(x_n\).

In code:
- `core.advanced_models.steering_weights_near_field_linear(...)`
- `core.advanced_models.array_factor_linear_field_mode(...)`

## 7) Wideband Response and Beam Squint

Phase-shifter weights are designed for \(f_0\). For \(f \neq f_0\), the electrical spacing scales as:

\[
d_{\text{eff}}(f)=d\frac{f}{f_0}
\]

As a result, the main lobe shifts with frequency, which is the beam-squint effect.

In code:
- `core.advanced_models.wideband_array_factor_linear(...)`

## 8) Element Patterns and Mutual Coupling

The overall response can be written in simplified form as:

\[
y(\theta,\phi)=g(\theta,\phi)\sum_n \tilde{w}_n a_n(\theta,\phi),
\quad \tilde{w}=Cw
\]

- \(g(\theta,\phi)\): element pattern, for example isotropic, cosine, or cardioid
- \(C\): coupling matrix representing mutual coupling

In code:
- `core.advanced_models.element_pattern_gain(...)`
- `core.advanced_models.build_mutual_coupling_matrix(...)`
- `core.advanced_models.array_factor_linear_with_impairments(...)`

## 9) Architecture Models: Digital / Analog / Hybrid

- Digital: one complex weight \(w_n\) per element
- Analog: RF phase-shifter weights with constant magnitude
- Hybrid: \(w \approx F_{\text{RF}} w_{\text{BB}}\)

In code:
- `core.advanced_models.synthesize_beamforming_architecture(...)`

## 10) Not Implemented

The following methods are currently not implemented as dedicated solvers:
- LMS / NLMS / RLS
- Frost
- full generic LCMV
