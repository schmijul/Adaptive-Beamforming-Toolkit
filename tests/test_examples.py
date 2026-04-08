from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_example(path: str) -> str:
    completed = subprocess.run(
        [sys.executable, path],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_linear_array_pattern_example_runs() -> None:
    output = _run_example("examples/linear_array_pattern.py")
    assert "Peak steering angle" in output


def test_adaptive_mvdr_music_example_runs() -> None:
    output = _run_example("examples/adaptive_mvdr_music.py")
    assert "MUSIC estimate" in output


def test_reproducible_cli_scenario_example_runs() -> None:
    output = _run_example("examples/reproducible_cli_scenario.py")
    assert "CLI mode: single" in output
