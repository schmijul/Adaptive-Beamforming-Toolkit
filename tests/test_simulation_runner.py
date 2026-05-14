from __future__ import annotations

from dataclasses import replace

from simulations.config import AlgorithmConfig
from simulations.config import load_scenario_config
from simulations.runner import run_monte_carlo, run_single_simulation


def test_load_default_config() -> None:
    config = load_scenario_config("config/default.yaml")
    assert config.name == "baseline_mvdr"
    assert config.algorithm.name == "mvdr"
    assert config.array.geometry == "ula"


def test_single_and_montecarlo_runs_write_outputs(tmp_path) -> None:
    base = load_scenario_config("config/conventional.yaml")
    config = replace(base, output=replace(base.output, directory=str(tmp_path), save_plots=False), snapshots=256)

    single = run_single_simulation(config)
    assert single["mode"] == "single"
    assert "result" in single

    mc = run_monte_carlo(config, runs=3, jobs=2)
    assert mc["mode"] == "montecarlo"
    assert mc["summary"]["runs"] == 3
    assert mc["summary"]["jobs"] == 2
    assert len(mc["runs"]) == 3

    assert (tmp_path / "simulate.json").exists()
    assert (tmp_path / "montecarlo.json").exists()


def test_default_conventional_scenario_has_stable_sinr_regression(tmp_path) -> None:
    base = load_scenario_config("config/conventional.yaml")
    config = replace(base, output=replace(base.output, directory=str(tmp_path), save_plots=False), snapshots=256)

    single = run_single_simulation(config)
    assert abs(single["result"]["sinr_db"] - 15.797460622963762) <= 1e-9


def test_planar_mvdr_scenario_runs(tmp_path) -> None:
    base = load_scenario_config("config/default.yaml")
    planar_array = replace(
        base.array,
        geometry="planar",
        num_elements=6,
        spacing_lambda=None,
        num_x=3,
        num_y=2,
        spacing_x_lambda=0.5,
        spacing_y_lambda=0.5,
    )
    config = replace(base, array=planar_array, output=replace(base.output, directory=str(tmp_path), save_plots=False), snapshots=256)

    single = run_single_simulation(config)
    assert single["mode"] == "single"
    assert len(single["result"]["positions_xy_lambda"]) == 6
    assert single["config"]["array"]["geometry"] == "planar"


def test_music_scenario_estimates_source_direction(tmp_path) -> None:
    base = load_scenario_config("config/default.yaml")
    algorithm = AlgorithmConfig(
        name="music",
        diagonal_loading=1e-3,
        num_sources=1,
        model_order="fixed",
        step_size=0.05,
        leakage=0.0,
        epsilon=1e-6,
        forgetting_factor=0.995,
        initialization_delta=1.0,
    )
    config = replace(
        base,
        algorithm=algorithm,
        interference_sources=(),
        output=replace(base.output, directory=str(tmp_path), save_plots=True),
        snapshots=1024,
        sweep=replace(base.sweep, theta_stop_deg=90.0, theta_num=181),
    )

    single = run_single_simulation(config)
    result = single["result"]

    assert single["mode"] == "single"
    assert abs(result["estimated_thetas_deg"][0] - base.desired_source.theta_deg) <= 2.0
    assert len(result["music_spectrum_db"]) == 181
    assert (tmp_path / "simulate.json").exists()
    assert (tmp_path / "music_spectrum.html").exists()


def test_music_example_config_runs(tmp_path) -> None:
    base = load_scenario_config("config/music_doa.yaml")
    config = replace(base, output=replace(base.output, directory=str(tmp_path), save_plots=False))

    single = run_single_simulation(config)
    result = single["result"]

    assert single["config"]["algorithm"]["name"] == "music"
    assert abs(result["estimated_thetas_deg"][0] - config.desired_source.theta_deg) <= 1.0
    assert len(result["theta_scan_deg"]) == 361
    assert result["num_sources"] == 1
    assert result["model_order"] == "mdl"
    assert result["model_order_candidates"] == [0, 1, 2, 3, 4]


def test_lcmv_example_config_runs_with_null_constraints(tmp_path) -> None:
    base = load_scenario_config("config/lcmv_nulls.yaml")
    config = replace(base, output=replace(base.output, directory=str(tmp_path), save_plots=False), snapshots=512)

    single = run_single_simulation(config)
    result = single["result"]

    assert single["config"]["algorithm"]["name"] == "lcmv"
    assert result["sinr_db"] > 10.0
    assert len(result["weight_amplitudes"]) == config.array.num_elements
