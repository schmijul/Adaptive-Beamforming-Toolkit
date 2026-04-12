from __future__ import annotations

import argparse
import json
import sys

from abf.services import (
    run_dashboard,
    run_dataset_command,
    run_env_demo_command,
    run_evaluate_command,
    run_gallery_command,
    run_montecarlo_command,
    run_simulate_command,
    run_train_command,
)


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
    monte.add_argument("--jobs", type=int, default=1)

    gallery = sub.add_parser("gallery", help="Generate reproducible gallery plots from a YAML config")
    gallery.add_argument("--config", default="config/default.yaml")

    dataset = sub.add_parser("dataset", help="Generate an ML-ready dataset from an experiment config")
    dataset.add_argument("--config", default="config/ml/doa_regression.yaml")

    train = sub.add_parser("train", help="Run a supervised ML experiment from an experiment config")
    train.add_argument("--config", default="config/ml/doa_regression.yaml")

    evaluate = sub.add_parser("evaluate", help="Evaluate a supervised ML experiment config")
    evaluate.add_argument("--config", default="config/ml/doa_regression.yaml")

    env_demo = sub.add_parser("env-demo", help="Step through the beam-selection environment")
    env_demo.add_argument("--config", default="config/rl/beam_selection.yaml")
    env_demo.add_argument("--steps", type=int, default=3)

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    try:
        if args.command == "dashboard":
            run_dashboard(host=args.host, port=args.port, debug=args.debug)
            return

        if args.command == "simulate":
            print(json.dumps(run_simulate_command(args.config), indent=2))
            return

        if args.command == "montecarlo":
            print(json.dumps(run_montecarlo_command(args.config, runs=args.runs, jobs=args.jobs), indent=2))
            return

        if args.command == "gallery":
            print(json.dumps(run_gallery_command(args.config), indent=2))
            return

        if args.command == "dataset":
            print(json.dumps(run_dataset_command(args.config), indent=2))
            return

        if args.command == "train":
            print(json.dumps(run_train_command(args.config), indent=2))
            return

        if args.command == "evaluate":
            print(json.dumps(run_evaluate_command(args.config), indent=2))
            return

        if args.command == "env-demo":
            print(json.dumps(run_env_demo_command(args.config, steps=args.steps), indent=2))
            return

        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
