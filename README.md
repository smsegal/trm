# PyTorch Project

A modern Python project with PyTorch, managed with `uv` and linted with `ruff`.

## Setup

1. Install [uv](https://github.com/astral-sh/uv) if you haven't already:
   ```bash
   curl -sSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install --upgrade pip
   uv pip install -e ".[dev]"
   ```

## Development

- Run tests:
  ```bash
  pytest
  ```

- Lint code:
  ```bash
  ruff check .
  ```

- Format code:
  ```bash
  ruff format .
  ```

## Project Structure

```
├── src/
│   └── pytorch_project/  # Main package
│       ├── __init__.py
│       └── model.py      # Example PyTorch model
├── tests/               # Test files
├── .gitignore
├── pyproject.toml
└── README.md
```
