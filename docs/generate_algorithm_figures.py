from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from algorithms.adaptive import doa_music_linear
from core.advanced_models import (
    array_factor_linear_field_mode,
    array_factor_linear_with_impairments,
    build_mutual_coupling_matrix,
    wideband_array_factor_linear,
)
from core.beamforming import array_factor_linear, array_factor_linear_from_weights, null_steering_weights_linear
from data.iq import simulate_array_iq


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "imgs"

THETA = np.linspace(0.0, 180.0, 721)
PHI = np.linspace(-180.0, 180.0, 361)
THETA_GRID = THETA[:, None] * np.ones((1, PHI.size))
PHI_GRID = np.ones((THETA.size, 1)) * PHI[None, :]
PHI_ZERO_INDEX = int(np.argmin(np.abs(PHI)))


def _save(fig: plt.Figure, name: str) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_null_steering_figure() -> None:
    baseline = array_factor_linear(
        num_elements=12,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=0.0,
        phi_steer_deg=0.0,
        taper_name="uniform",
    )
    weights = null_steering_weights_linear(
        num_elements=12,
        spacing_lambda=0.5,
        theta_steer_deg=0.0,
        phi_steer_deg=0.0,
        null_thetas_deg=np.array([25.0]),
        null_phis_deg=np.array([0.0]),
        taper_name="uniform",
    )
    nulled = array_factor_linear_from_weights(
        weights=np.asarray(weights),
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(THETA, baseline["magnitude_db"][:, PHI_ZERO_INDEX], label="Conventional steering", lw=2.2)
    ax.plot(THETA, nulled["magnitude_db"][:, PHI_ZERO_INDEX], label="Null steering at 25 deg", lw=2.2)
    ax.axvline(25.0, color="tab:red", ls="--", lw=1.2)
    ax.set_title("Steering vs Null-Steering")
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Normalized gain (dB)")
    ax.set_ylim(-65.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "alg-null-steering.png")


def generate_music_figure() -> None:
    theta_scan = np.linspace(0.0, 90.0, 721)
    snapshots = simulate_array_iq(
        num_elements=12,
        num_snapshots=4096,
        spacing_lambda=0.5,
        source_thetas_deg=np.array([18.0, 42.0]),
        source_phis_deg=np.array([0.0, 0.0]),
        source_snr_db=np.array([18.0, 16.0]),
        random_seed=12,
    )
    music = doa_music_linear(
        snapshots=snapshots,
        spacing_lambda=0.5,
        theta_scan_deg=theta_scan,
        num_sources=2,
    )

    spectrum_db = 10.0 * np.log10(music["spectrum"] / np.max(music["spectrum"]))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(theta_scan, spectrum_db, lw=2.2, color="#0f766e")
    for theta in [18.0, 42.0]:
        ax.axvline(theta, color="tab:red", ls="--", lw=1.2)
    ax.set_title("MUSIC Pseudospectrum")
    ax.set_xlabel("Theta scan (deg)")
    ax.set_ylabel("Relative spectrum (dB)")
    ax.set_ylim(-50.0, 1.0)
    ax.grid(True, alpha=0.25)
    _save(fig, "alg-music-spectrum.png")


def generate_near_far_figure() -> None:
    far = array_factor_linear_field_mode(
        num_elements=16,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=35.0,
        phi_steer_deg=0.0,
        field_mode="far",
    )
    near = array_factor_linear_field_mode(
        num_elements=16,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=35.0,
        phi_steer_deg=0.0,
        field_mode="near",
        focus_range_lambda=6.0,
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(THETA, far["magnitude_db"][:, PHI_ZERO_INDEX], label="Far field", lw=2.2)
    ax.plot(THETA, near["magnitude_db"][:, PHI_ZERO_INDEX], label="Near field focus at 6 lambda", lw=2.2)
    ax.set_title("Near-Field vs Far-Field Focusing")
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Normalized gain (dB)")
    ax.set_ylim(-65.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "alg-near-vs-far.png")


def generate_wideband_figure() -> None:
    result = wideband_array_factor_linear(
        num_elements=16,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=50.0,
        phi_steer_deg=0.0,
        center_frequency_hz=10e9,
        frequency_hz=np.array([8e9, 10e9, 12e9]),
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for idx, freq in enumerate(result["frequency_hz"]):
        label = f"{freq / 1e9:.0f} GHz"
        ax.plot(THETA, result["magnitude_db"][idx, :, PHI_ZERO_INDEX], label=label, lw=2.2)
    ax.set_title("Wideband Beam Squint")
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Normalized gain (dB)")
    ax.set_ylim(-65.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "alg-wideband-squint.png")


def generate_impairments_figure() -> None:
    ideal = array_factor_linear(
        num_elements=14,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=0.0,
        phi_steer_deg=0.0,
    )
    impaired = array_factor_linear_with_impairments(
        num_elements=14,
        spacing_lambda=0.5,
        theta_grid_deg=THETA_GRID,
        phi_grid_deg=PHI_GRID,
        theta_steer_deg=0.0,
        phi_steer_deg=0.0,
        element_pattern_name="cosine",
        element_pattern_exponent=2.0,
        coupling_matrix=build_mutual_coupling_matrix(14, nearest_neighbor_db=-12.0, phase_deg=-25.0),
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(THETA, ideal["magnitude_db"][:, PHI_ZERO_INDEX], label="Ideal isotropic array", lw=2.2)
    ax.plot(THETA, impaired["magnitude_db"][:, PHI_ZERO_INDEX], label="Cosine element pattern + coupling", lw=2.2)
    ax.set_title("Pattern Deformation from Element Pattern and Coupling")
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Normalized gain (dB)")
    ax.set_ylim(-65.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, "alg-impairments.png")


def main() -> None:
    generate_null_steering_figure()
    generate_music_figure()
    generate_near_far_figure()
    generate_wideband_figure()
    generate_impairments_figure()


if __name__ == "__main__":
    main()
