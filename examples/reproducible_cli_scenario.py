from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base_config = root / "config" / "default.yaml"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
        payload["output"]["directory"] = str(tmp_path / "artifacts")
        payload["output"]["save_plots"] = False

        temp_config = tmp_path / "scenario.yaml"
        temp_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, "-m", "abf_cli", "simulate", "--config", str(temp_config)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        command_output = json.loads(completed.stdout)
        result_path = Path(command_output["out_dir"]) / "simulate.json"
        simulation = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"CLI mode: {command_output['mode']}")
        print(f"SINR dB: {simulation['result']['sinr_db']:.2f}")


if __name__ == "__main__":
    main()
