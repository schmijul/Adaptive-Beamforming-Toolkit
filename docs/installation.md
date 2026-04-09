# Installation

## Requirements

The package builds a native extension, so installation is partly a Python task and partly a local compiler-toolchain task.

- Python 3.10 or newer
- `pip` and virtual-environment support
- A C++17-capable compiler available in the active shell

Typical Linux and macOS setups work as long as the compiler can build the `core._beamforming_cpp` extension during `pip install -e .`.

## Recommended Editable Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Install optional extras only when needed:

```bash
pip install -e ".[ui]"
pip install -e ".[dev]"
```

`requirements.txt` remains available as a convenience for a full local development environment.

Why editable mode:

- the Python modules stay linked to your working tree
- the CLI entry point `abf` is installed
- the C++ extension is compiled once into the environment and reused during development

## What Gets Installed

An editable install exposes:

- the Python packages in `core/`, `algorithms/`, `data/`, `visualize/`, `simulations/`, and `ui/`
- the console script `abf`
- the compiled extension module `core._beamforming_cpp`

## Verification

Run the automated tests:

```bash
pytest -q
```

Run the dashboard directly:

```bash
python -m abf dashboard
```

Or use the installed CLI:

```bash
abf dashboard
```

The default dashboard endpoint is `http://127.0.0.1:8050`.

## Build Troubleshooting

If `pip install -e .` fails while compiling the extension:

1. Confirm that the virtual environment is active.
2. Confirm that your shell can locate a C++ compiler.
3. Retry the install after upgrading `pip`, `setuptools`, and `wheel` if your environment is old.

The extension is defined in `setup.py` and compiled from `core/beamforming_cpp.cpp` using C++17 flags. A missing or misconfigured compiler is the most common installation problem.

## Optional Next Step

Once installation succeeds, continue with [Getting Started](getting-started.md) for the first dashboard run, Python example, and CLI simulation workflow.
