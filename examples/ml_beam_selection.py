from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abf.ml.train import run_experiment


def main() -> None:
    result = run_experiment("config/ml/beam_selection.yaml")
    print(result.metrics)


if __name__ == "__main__":
    main()
