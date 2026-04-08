# Algorithms

Dieses Dokument beschreibt die aktuell implementierten Verfahren mit Signalmodell und Formeln.

## 1) Array- und Signalmodell (Narrowband)

Für ein lineares Array entlang der x-Achse mit Elementpositionen \(x_n\) (in \(\lambda\)):

\[
u(\theta,\phi)=\sin(\theta)\cos(\phi), \quad
a_n(\theta,\phi)=e^{j2\pi x_n u(\theta,\phi)}
\]

Der Array-Output mit Gewichten \(w_n\):

\[
y(\theta,\phi)=\sum_{n=1}^{N} w_n\,a_n(\theta,\phi), \quad
AF(\theta,\phi)=|y(\theta,\phi)|
\]

Im Code:
- `core.beamforming.array_factor_linear(...)`
- `core.beamforming.array_factor_planar(...)`

## 2) Konventionelles Steering und Tapering

Für eine Look-Richtung \((\theta_0,\phi_0)\):

\[
w_n = \alpha_n \, e^{-j2\pi x_n u(\theta_0,\phi_0)}
\]

\(\alpha_n\) ist der Amplitudentaper (uniform, Hamming, Taylor).

Im Code:
- `core.beamforming.steering_weights_linear(...)`
- `core.beamforming.amplitude_taper(...)`

## 3) Null-Steering (deterministische Constraints)

Gesucht sind Gewichte mit:

\[
w^H a(\theta_0,\phi_0)=1, \quad
w^H a(\theta_k,\phi_k)=0 \;\; \forall k\in\{1,\dots,K\}
\]

Die Implementierung baut eine Constraint-Matrix und löst ein lineares Gleichungssystem im Constraint-Raum.

Im Code:
- `core.beamforming.null_steering_weights_linear(...)`

## 4) MVDR (Capon)

Optimierung:

\[
\min_w \; w^H R w
\quad \text{s.t.} \quad
w^H a_0 = 1
\]

Geschlossene Lösung:

\[
w_{\text{MVDR}} = \frac{R^{-1}a_0}{a_0^H R^{-1} a_0}
\]

Mit optionalem Diagonal Loading:

\[
R_\delta = R + \delta I
\]

Im Code:
- `algorithms.adaptive.mvdr_weights(...)`
- `algorithms.adaptive.estimate_covariance_matrix(...)`

## 5) MUSIC (DoA-Schätzung)

Eigenzerlegung der Kovarianz:

\[
R = E_s \Lambda_s E_s^H + E_n \Lambda_n E_n^H
\]

Pseudo-Spektrum:

\[
P_{\text{MUSIC}}(\theta,\phi) =
\frac{1}{a^H(\theta,\phi)\,E_nE_n^H\,a(\theta,\phi)}
\]

Peaks von \(P_{\text{MUSIC}}\) liefern DoA-Schätzungen.

Im Code:
- `algorithms.adaptive.music_spectrum(...)`
- `algorithms.adaptive.doa_music_linear(...)`

## 6) Near-Field Focusing

Im Nahfeld hängt die Phase von der echten Distanz \(r_n\) zum Fokuspunkt ab:

\[
r_n = \|p_{\text{focus}} - p_n\|, \quad
w_n \propto e^{-j2\pi r_n}
\]

Im Fernfeld wird stattdessen die ebene Welle (lineare Phase über \(x_n\)) verwendet.

Im Code:
- `core.advanced_models.steering_weights_near_field_linear(...)`
- `core.advanced_models.array_factor_linear_field_mode(...)`

## 7) Wideband und Beam Squint

Phase-Shifter-Gewichte sind für \(f_0\) optimiert. Für \(f\neq f_0\) skaliert die elektrische Distanz:

\[
d_{\text{eff}}(f)=d\frac{f}{f_0}
\]

Dadurch wandert die Hauptkeule mit der Frequenz (Beam Squint).

Im Code:
- `core.advanced_models.wideband_array_factor_linear(...)`

## 8) Elementpattern und Mutual Coupling

Gesamtantwort (vereinfacht):

\[
y(\theta,\phi)=g(\theta,\phi)\sum_n \tilde{w}_n a_n(\theta,\phi),
\quad \tilde{w}=Cw
\]

- \(g(\theta,\phi)\): Elementpattern (z. B. isotropic, cosine, cardioid)
- \(C\): Kopplungsmatrix (Mutual Coupling)

Im Code:
- `core.advanced_models.element_pattern_gain(...)`
- `core.advanced_models.build_mutual_coupling_matrix(...)`
- `core.advanced_models.array_factor_linear_with_impairments(...)`

## 9) Architekturmodelle: Digital / Analog / Hybrid

- Digital: je Element komplexes Gewicht \(w_n\)
- Analog: phasenbasierte RF-Gewichte (konstante Magnitude)
- Hybrid: \(w \approx F_{\text{RF}} w_{\text{BB}}\)

Im Code:
- `core.advanced_models.synthesize_beamforming_architecture(...)`

## 10) Nicht implementiert

Folgende Verfahren sind aktuell nicht als Solver implementiert:
- LMS / NLMS / RLS
- Frost
- vollständiges generisches LCMV
