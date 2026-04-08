from __future__ import annotations

import argparse
import json

from simulations import load_scenario_config, run_monte_carlo, run_single_simulation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abf", description="Adaptive Beamforming Toolkit CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    dashboard = sub.add_parser("dashboard", help="Run the interactive Dash dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8050)
    dashboard.add_argument("--debug", action="store_true")

    simulate = sub.add_parser("simulate", help="Run one deterministic simulation from a YAML config")
    simulate.add_argument("--config", default="config/default.yaml")

    monte = sub.add_parser("montecarlo", help="Run a Monte-Carlo simulation from a YAML config")
    monte.add_argument("--config", default="config/default.yaml")
    monte.add_argument("--runs", type=int, default=30)

    gallery = sub.add_parser("gallery", help="Generate reproducible gallery plots from a YAML config")
    gallery.add_argument("--config", default="config/default.yaml")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "dashboard":
        from ui.dash_app import create_app

        app = create_app()
        app.run(host=args.host, port=args.port, debug=args.debug)
        return

    config = load_scenario_config(args.config)

    if args.command == "simulate":
        payload = run_single_simulation(config)
        print(json.dumps({"mode": payload["mode"], "out_dir": config.output.directory}, indent=2))
        return

    if args.command == "montecarlo":
        payload = run_monte_carlo(config, runs=args.runs)
        print(json.dumps({"mode": payload["mode"], "summary": payload["summary"], "out_dir": config.output.directory}, indent=2))
        return

    if args.command == "gallery":
        payload = run_single_simulation(config)
        print(json.dumps({"mode": payload["mode"], "plots_written": config.output.save_plots, "out_dir": config.output.directory}, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
