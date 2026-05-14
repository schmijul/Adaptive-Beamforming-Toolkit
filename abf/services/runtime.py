from __future__ import annotations

from collections.abc import Mapping

from abf.ml import generate_dataset, load_experiment_config, run_experiment
from abf.ml.envs import BeamSelectionEnv
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
    result = payload["result"]
    summary: dict[str, object] = {
        "mode": payload["mode"],
        "algorithm": config.algorithm.name,
        "out_dir": config.output.directory,
        "result_path": f"{config.output.directory}/simulate.json",
    }
    if "estimated_thetas_deg" in result:
        summary["estimated_thetas_deg"] = result["estimated_thetas_deg"]
    if "sinr_db" in result:
        summary["sinr_db"] = result["sinr_db"]
    if "support_indices" in result:
        summary["support_indices"] = result["support_indices"]
    if "source_power" in result:
        summary["source_power"] = result["source_power"]
    if config.output.save_plots:
        if "music_spectrum_db" in result:
            summary["plot_path"] = f"{config.output.directory}/music_spectrum.html"
        if "sparse_spectrum_db" in result:
            summary["plot_path"] = f"{config.output.directory}/sparse_spectrum.html"
    return summary


def run_montecarlo_command(config_path: str, *, runs: int, jobs: int) -> Mapping[str, object]:
    config = load_scenario_config(config_path)
    payload = run_monte_carlo(config, runs=runs, jobs=jobs)
    return {"mode": payload["mode"], "summary": payload["summary"], "out_dir": config.output.directory}


def run_gallery_command(config_path: str) -> Mapping[str, object]:
    config = load_scenario_config(config_path)
    payload = run_single_simulation(config)
    return {"mode": payload["mode"], "plots_written": config.output.save_plots, "out_dir": config.output.directory}


def run_dataset_command(config_path: str) -> Mapping[str, object]:
    exp = load_experiment_config(config_path)
    dataset = generate_dataset(exp)
    dataset_path = exp.dataset.export_path
    if dataset_path:
        from abf.ml import save_dataset

        save_dataset(dataset, dataset_path)
    return {
        "mode": "dataset",
        "task": exp.task_name,
        "feature_type": exp.dataset.feature_type,
        "summary": dataset.summary(),
        "dataset_path": dataset_path,
    }


def run_train_command(config_path: str) -> Mapping[str, object]:
    result = run_experiment(config_path)
    return {
        "mode": "train",
        "metrics": result.metrics,
        "baselines": result.baselines,
        "out_dir": result.output_dir,
        "model_path": result.model_path,
        "dataset_path": result.dataset_path,
        "results_path": result.results_path,
    }


def run_evaluate_command(config_path: str) -> Mapping[str, object]:
    result = run_experiment(config_path)
    return {
        "mode": "evaluate",
        "metrics": result.metrics,
        "baselines": result.baselines,
        "out_dir": result.output_dir,
        "results_path": result.results_path,
    }


def run_env_demo_command(config_path: str, *, steps: int) -> Mapping[str, object]:
    env = BeamSelectionEnv(config_path)
    observation, info = env.reset()
    transitions = []
    for _ in range(max(1, steps)):
        action = env.action_space.sample()
        _, reward, terminated, truncated, step_info = env.step(action)
        transitions.append(
            {
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": step_info,
            }
        )
        if terminated or truncated:
            break
    return {
        "mode": "env-demo",
        "observation_shape": list(observation.shape),
        "reset_info": info,
        "transitions": transitions,
    }
