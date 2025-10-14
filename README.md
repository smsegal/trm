# Tiny Reasoning Models (TRM)

This repository contains a reimplementation of the Tiny Reasoning Models paper (Jolicoeur-Martineau, Alexia. “Less Is More: Recursive Reasoning with Tiny Networks.” arXiv:2510.04871. Preprint, arXiv, October 6, 2025. https://doi.org/10.48550/arXiv.2510.04871.) 

## Environment Setup

1. Install [uv](https://github.com/astral-sh/uv) (once per machine):
   ```bash
   curl -sSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment and install the project in editable mode with development extras:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

   Alternatively, Nix users can enter the pinned toolchain:
   ```bash
   nix develop
   ```

## Key Commands

- Run the full test suite (coverage enabled):
  ```bash
  uv run pytest
  ```

- Run static analysis:
  ```bash
  uv run basedpyright
  ```

- Lint and format:
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```

## Training

The `trm.train` module bundles encoding helpers, configuration, and executable entry points for Sudoku training.
To launch a training run against the default `sapientinc/sudoku-extreme` dataset:

```bash
uv run python -m trm.train --epochs 3 --batch-size 128 --latent-steps 4 --recurse 6
```

Notable CLI options:

- `--streaming` enables iterable datasets for large-scale runs.
- `--eval-split` chooses the dataset split to evaluate after each epoch (defaults to `test`).
- `--device` selects the compute target (`auto`, `cpu`, `cuda`, or `mps`).

Use `uv run python -m trm.train --help` to inspect the full argument list.

The core helpers are also available as a library:

- `encode_board` / `encode_example` convert Sudoku strings into token IDs.
- `run_training` orchestrates data loading, optimization, and evaluation driven by a `TrainingConfig`.

## Project Structure

```
├── src/
│   └── trm/
│       ├── __init__.py          # Public package API
│       ├── model.py             # TRM architecture
│       └── train.py             # Training utilities and CLI
├── tests/                       # PyTest suites mirroring src/trm modules
├── pyproject.toml               # Project metadata and tooling configuration
├── flake.nix                    # Nix development shell definition
├── uv.lock                      # Locked Python dependencies
└── README.md
```

## Reference Materials

The repository ships with `paper.pdf` and `paper.txt` for model context. Update these sources only when
the implementation diverges or new derivations are required.
