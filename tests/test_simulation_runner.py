from __future__ import annotations

from dataclasses import replace

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
