from __future__ import annotations

import warnings

from abf.cli import main as _main

def main() -> None:
    warnings.warn(
        "`abf_cli` is deprecated and will be removed in the next release. Use `abf.cli` or `python -m abf`.",
        DeprecationWarning,
        stacklevel=2,
    )
    _main()


if __name__ == "__main__":
    main()
