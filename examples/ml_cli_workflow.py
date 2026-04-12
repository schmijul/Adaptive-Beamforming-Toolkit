from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    subprocess.run([sys.executable, "-m", "abf", "dataset", "--config", "config/ml/doa_regression.yaml"], check=True)
    subprocess.run([sys.executable, "-m", "abf", "train", "--config", "config/ml/doa_regression.yaml"], check=True)


if __name__ == "__main__":
    main()
