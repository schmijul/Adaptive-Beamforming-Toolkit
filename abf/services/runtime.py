from __future__ import annotations

from collections.abc import Mapping

from abf.simulations import load_scenario_config, run_monte_carlo, run_single_simulation


def run_dashboard(*, host: str, port: int, debug: bool) -> None:
    """Run the dashboard, requiring the optional UI dependency extra."""
    try:
        from ui.dash_app import create_app
    except ModuleNotFoundError as exc:
        if exc.name in {"dash", "plotly"}:
            raise RuntimeError(
                "Dashboard dependencies are optional. Install with: pip install 'adaptive-beamforming-toolkit[ui]'"
            ) from exc
        raise

    app = create_app()
    app.run(host=host, port=port, debug=debug)


def run_simulate_command(config_path: str) -> Mapping[str, object]:
    config = load_scenario_config(config_path)
    payload = run_single_simulation(config)
    return {"mode": payload["mode"], "out_dir": config.output.directory}


def run_montecarlo_command(config_path: str, *, runs: int, jobs: int) -> Mapping[str, object]:
    config = load_scenario_config(config_path)
    payload = run_monte_carlo(config, runs=runs, jobs=jobs)
    return {"mode": payload["mode"], "summary": payload["summary"], "out_dir": config.output.directory}


def run_gallery_command(config_path: str) -> Mapping[str, object]:
    config = load_scenario_config(config_path)
    payload = run_single_simulation(config)
    return {"mode": payload["mode"], "plots_written": config.output.save_plots, "out_dir": config.output.directory}
