from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_elevation_cut(theta_deg: np.ndarray, magnitude_db: np.ndarray, theta_steer_deg: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theta_deg,
            y=magnitude_db,
            mode="lines",
            name="Elevation Cut",
            line={"width": 3, "color": "#0f766e"},
        )
    )
    fig.add_vline(x=theta_steer_deg, line_dash="dash", line_color="#b91c1c")
    fig.update_layout(
        template="plotly_white",
        title="2D Elevation Cut",
        xaxis_title="Theta (deg)",
        yaxis_title="Normalized Gain (dB)",
        yaxis_range=[-60, 1],
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    return fig


def build_heatmap(theta_deg: np.ndarray, phi_deg: np.ndarray, magnitude_db: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            x=phi_deg,
            y=theta_deg,
            z=magnitude_db,
            colorscale="Viridis",
            zmin=-40,
            zmax=0,
            colorbar={"title": "dB"},
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Theta/Phi Heatmap",
        xaxis_title="Phi (deg)",
        yaxis_title="Theta (deg)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    return fig


def build_pattern_3d(theta_deg: np.ndarray, phi_deg: np.ndarray, magnitude: np.ndarray) -> go.Figure:
    theta_rad = np.deg2rad(theta_deg)[:, None]
    phi_rad = np.deg2rad(phi_deg)[None, :]

    radius = np.maximum(magnitude, 0.02)
    x = radius * np.sin(theta_rad) * np.cos(phi_rad)
    y = radius * np.sin(theta_rad) * np.sin(phi_rad)
    z = radius * np.cos(theta_rad)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=20.0 * np.log10(np.maximum(magnitude, 1e-6)),
                colorscale="Turbo",
                cmin=-40,
                cmax=0,
                colorbar={"title": "dB"},
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title="3D Radiation Pattern",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


def build_weights_plot(
    positions_lambda: np.ndarray,
    amplitudes: np.ndarray,
    phase_weights: np.ndarray,
) -> go.Figure:
    phase_deg = np.rad2deg(np.angle(phase_weights))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=positions_lambda,
            y=amplitudes,
            name="Amplitude",
            marker_color="#2563eb",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=positions_lambda,
            y=phase_deg,
            mode="lines+markers",
            name="Phase",
            line={"color": "#dc2626", "width": 2},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_white",
        title="Separated Amplitude / Phase Control",
        xaxis_title="Element Position (lambda)",
        margin={"l": 50, "r": 50, "t": 50, "b": 50},
    )
    fig.update_yaxes(title_text="Amplitude", secondary_y=False, range=[0, 1.1])
    fig.update_yaxes(title_text="Phase (deg)", secondary_y=True)
    return fig
