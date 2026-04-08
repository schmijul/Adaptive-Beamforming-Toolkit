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

    mc = run_monte_carlo(config, runs=3)
    assert mc["mode"] == "montecarlo"
    assert mc["summary"]["runs"] == 3
    assert len(mc["runs"]) == 3

    assert (tmp_path / "simulate.json").exists()
    assert (tmp_path / "montecarlo.json").exists()
