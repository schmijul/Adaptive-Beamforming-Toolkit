# Examples

## Current state

This repository does not currently ship a dedicated `examples/` directory or standalone example scripts.

## Real runnable entrypoints

### `app.py`

Run:

```bash
python app.py
```

Expected output:

- a local Dash server starts
- the browser UI shows 2D cut, heatmap, 3D pattern, and weight plots

### `pytest -q`

Run:

```bash
pytest -q
```

Expected output:

- the test suite passes
- coverage includes steering, planar arrays, nulling, MVDR, MUSIC, wideband effects, impairments, and IQ utilities

## Code examples already present in the repo

- `README.md`: package overview and a short Python usage snippet
- `tests/test_ground_truth.py`: reference-backed examples for array-factor behavior
- `tests/test_next_steps.py`: examples for adaptive algorithms and advanced models
