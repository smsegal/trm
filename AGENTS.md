# Repository Guidelines

## Project Structure & Module Organization
- `src/trm/` holds the library code; start with `model.py` for architecture references.
- `tests/` mirrors public APIs; add a new `test_<module>.py` when introducing modules.
- `paper.pdf` and `paper.txt` are reference materials; avoid modifying them unless updating the derived implementation.
- Lockfiles (`uv.lock`, `flake.lock`) and `flake.nix` pin dependencies; regenerate only when dependency versions change, and mention it in your pull request.

## Environment Setup
- Use Python 3.12 via `uv`. Create an environment with `uv venv` and install extras with `uv pip install -e ".[dev]"`.
- When invoking Python directly, prefer the project venv binary at `/Users/shanesegal/projects/trm/.venv/bin/python` to match CI.
- Nix users can run `nix develop` to enter the pinned toolchain; commit updated lockfiles whenever they change.

## Build, Test, and Development Commands
- `uv run pytest` — executes the full test suite with verbose output and coverage against `src`.
- `uv run ruff check .` — lint the codebase using the shared `pyproject.toml` configuration.
- `uv run ruff format .` — auto-format; run before committing.
- `/Users/shanesegal/projects/trm/.venv/bin/python -m basedpyright` — perform type checks with the stricter fork we standardize on.

## Coding Style & Naming Conventions
- Ruff enforces line length ≤ 120, explicit imports, and allows `torch.nn.functional as F`; keep modules importable and lightweight.
- Use `snake_case` for functions, variables, and tests; `PascalCase` for classes; UPPER_SNAKE for constants.
- Include type hints throughout; prefer `jaxtyping` annotations when expressing tensor shapes.
- Expose new public APIs via `src/trm/__init__.py` and document non-obvious tensor semantics inline.

## Testing Guidelines
- Place tests under `tests/` using the filename pattern `test_*.py`, mirroring module names.
- Maintain coverage by honoring the default `pytest --cov=src --cov-report=term-missing` options; add regression tests for tensor shapes and edge cases.
- Use PyTest fixtures for sample tensors and avoid GPU-only assumptions in baseline tests unless guarded.

## Commit & Pull Request Guidelines
- Follow the existing history’s style: a single capitalized, imperative sentence (e.g., `Add shape-typed TRM RoPE implementation`).
- Reference related issues in the body (`Fixes #123`) and summarize architectural decisions or deviations from the paper.
- Include in every PR: purpose, key changes, test results (paste command output), and performance notes when model behavior changes.
- Request review before merging and wait for CI to pass, or explain any required skips explicitly.

## Version Control (jj)
- We manage history with [Jujutsu (`jj`)](https://github.com/martinvonz/jj); avoid direct `git` commands to keep the changeop stack clean.
- `jj status` reveals working-copy changes, and `jj diff` shows the staged diff before committing.
- Create commits with `jj commit -m "Descriptive message"`; ensure the working copy (`@`) is clean before handing off.
- The project history expects short, imperative commit messages—mirror the existing style when crafting new commits.
