# cogwar-bakeoff

Model selection bake-off for CIB detection in Hindi/English/Urdu.

## Structure

```
src/               # project source code
data/raw/          # untouched source data (gitignored — populate locally)
data/processed/    # cleaned/featurized data (gitignored — populate locally)
tests/             # test suite
configs/           # experiment/model configs
results/           # bake-off outputs, metrics (json outputs gitignored)
```

## Setup

```
uv sync
```

This creates a `.venv` pinned to Python 3.11 with all dev dependencies (pytest, pytest-cov, ruff, mypy) installed.

## Data

The `data/` folder is not tracked in git. Populate `data/raw/` and `data/processed/` locally before running anything.
