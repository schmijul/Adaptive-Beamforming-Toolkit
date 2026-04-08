from __future__ import annotations

import numpy as np
from dash import Dash, Input, Output, dcc, html

from core.beamforming import array_factor_linear
from visualize.plots import (
    build_elevation_cut,
    build_heatmap,
    build_pattern_3d,
    build_weights_plot,
)


THETA_DEG = np.linspace(0.0, 180.0, 181)
PHI_DEG = np.linspace(-180.0, 180.0, 241)


def create_app() -> Dash:
    app = Dash(__name__)
    app.title = "Beamforming Simulator"

    app.layout = html.Div(
        style={
            "fontFamily": "ui-sans-serif, sans-serif",
            "padding": "24px",
            "background": "linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%)",
            "minHeight": "100vh",
        },
        children=[
            html.H1("Adaptive Beamforming Toolkit"),
            html.P("MVP: linear array, steering, 2D/3D plots, grating lobe intuition."),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "340px 1fr",
                    "gap": "20px",
                    "alignItems": "start",
                },
                children=[
                    _controls_panel(),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
                        children=[
                            dcc.Graph(id="cut-plot"),
                            dcc.Graph(id="weights-plot"),
                            dcc.Graph(id="heatmap-plot"),
                            dcc.Graph(id="pattern-plot"),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("cut-plot", "figure"),
        Output("weights-plot", "figure"),
        Output("heatmap-plot", "figure"),
        Output("pattern-plot", "figure"),
        Output("status-text", "children"),
        Input("num-elements", "value"),
        Input("spacing", "value"),
        Input("theta-steer", "value"),
        Input("phi-steer", "value"),
        Input("taper", "value"),
    )
    def update_plots(num_elements: int, spacing: float, theta_steer: float, phi_steer: float, taper: str):
        result = array_factor_linear(
            num_elements=num_elements,
            spacing_lambda=spacing,
            theta_grid_deg=THETA_DEG[:, None] * np.ones((1, PHI_DEG.size)),
            phi_grid_deg=np.ones((THETA_DEG.size, 1)) * PHI_DEG[None, :],
            theta_steer_deg=theta_steer,
            phi_steer_deg=phi_steer,
            taper_name=taper,
        )

        phi0_index = int(np.argmin(np.abs(PHI_DEG)))
        cut_db = result["magnitude_db"][:, phi0_index]

        grating_lobe_hint = "visible" if spacing > 0.5 else "suppressed"
        status_text = (
            f"N={num_elements}, d/lambda={spacing:.2f}, steer=({theta_steer:.1f} deg, {phi_steer:.1f} deg), "
            f"taper={taper}, grating lobes {grating_lobe_hint}."
        )

        return (
            build_elevation_cut(THETA_DEG, cut_db, theta_steer),
            build_weights_plot(result["positions_lambda"], result["amplitudes"], result["phase_weights"]),
            build_heatmap(THETA_DEG, PHI_DEG, result["magnitude_db"]),
            build_pattern_3d(THETA_DEG, PHI_DEG, result["magnitude"]),
            status_text,
        )

    return app


def _controls_panel() -> html.Div:
    card_style = {
        "background": "white",
        "padding": "18px",
        "borderRadius": "16px",
        "boxShadow": "0 12px 30px rgba(15, 23, 42, 0.08)",
        "position": "sticky",
        "top": "24px",
    }

    return html.Div(
        style=card_style,
        children=[
            html.H3("Controls"),
            _slider("num-elements", "Elements N", 2, 32, 1, 8, marks={2: "2", 8: "8", 16: "16", 32: "32"}),
            _slider("spacing", "Spacing d/lambda", 0.25, 2.0, 0.05, 0.5, marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5", 2.0: "2.0"}),
            _slider("theta-steer", "Steering Theta", 0, 180, 1, 30, marks={0: "0", 30: "30", 60: "60", 90: "90", 120: "120", 150: "150", 180: "180"}),
            _slider("phi-steer", "Steering Phi", -180, 180, 1, 0, marks={-180: "-180", -90: "-90", 0: "0", 90: "90", 180: "180"}),
            html.Label("Amplitude Taper"),
            dcc.Dropdown(
                id="taper",
                options=[
                    {"label": "Uniform", "value": "uniform"},
                    {"label": "Hamming", "value": "hamming"},
                    {"label": "Taylor", "value": "taylor"},
                ],
                value="uniform",
                clearable=False,
                style={"marginBottom": "16px"},
            ),
            html.Div(
                id="status-text",
                style={
                    "padding": "12px",
                    "borderRadius": "12px",
                    "background": "#ecfeff",
                    "color": "#155e75",
                    "fontWeight": 600,
                },
            ),
        ],
    )


def _slider(component_id: str, label: str, min_value: float, max_value: float, step: float, value: float, marks: dict) -> html.Div:
    return html.Div(
        style={"marginBottom": "18px"},
        children=[
            html.Label(label),
            dcc.Slider(
                id=component_id,
                min=min_value,
                max=max_value,
                step=step,
                value=value,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
    )
