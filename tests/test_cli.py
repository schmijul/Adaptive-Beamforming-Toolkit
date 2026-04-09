from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_cli_simulate_returns_json_and_writes_output(tmp_path) -> None:
    payload = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    payload["output"]["directory"] = str(tmp_path)
    payload["output"]["save_plots"] = False
    config_path = tmp_path / "scenario.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "abf_cli", "simulate", "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    command_output = json.loads(completed.stdout)
    simulate_json = json.loads((tmp_path / "simulate.json").read_text(encoding="utf-8"))

    assert command_output["mode"] == "single"
    assert Path(command_output["out_dir"]) == tmp_path
    assert simulate_json["mode"] == "single"


def test_cli_rejects_invalid_config(tmp_path) -> None:
    invalid_payload = {
        "name": "invalid",
        "seed": 1,
        "snapshots": 128,
        "array": {"geometry": "circular", "num_elements": 8, "spacing_lambda": 0.5},
        "desired_source": {"theta_deg": 0.0, "phi_deg": 0.0, "snr_db": 10.0},
        "interference_sources": [],
        "algorithm": {"name": "mvdr"},
        "sweep": {
            "theta_start_deg": 0.0,
            "theta_stop_deg": 90.0,
            "theta_num": 21,
            "phi_start_deg": -90.0,
            "phi_stop_deg": 90.0,
            "phi_num": 21,
        },
        "output": {"directory": str(tmp_path / "out"), "save_plots": False},
    }
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(invalid_payload, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "abf_cli", "simulate", "--config", str(config_path)],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "array.geometry must be one of" in completed.stderr


def test_cli_supports_planar_scenario(tmp_path) -> None:
    payload = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    payload["array"] = {
        "geometry": "planar",
        "num_x": 3,
        "num_y": 2,
        "spacing_x_lambda": 0.5,
        "spacing_y_lambda": 0.5,
    }
    payload["algorithm"]["name"] = "mvdr"
    payload["output"]["directory"] = str(tmp_path)
    payload["output"]["save_plots"] = False
    config_path = tmp_path / "planar.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "abf_cli", "simulate", "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    command_output = json.loads(completed.stdout)
    simulate_json = json.loads((tmp_path / "simulate.json").read_text(encoding="utf-8"))

    assert command_output["mode"] == "single"
    assert simulate_json["config"]["array"]["geometry"] == "planar"
    assert len(simulate_json["result"]["positions_xy_lambda"]) == 6
