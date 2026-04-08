# Installation

## Requirements

- Python 3.10+
- A C++17-capable compiler
- `pip` and virtual environment support

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Verify

Run the test suite:

```bash
pytest -q
```

Run the dashboard:

```bash
python app.py
```

Then open `http://127.0.0.1:8050`.
