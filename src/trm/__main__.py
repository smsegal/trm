"""Command-line entry point for ``python -m trm``."""

from .train import main as _train_main

def main() -> None:
    """Delegate to the training CLI defined in ``trm.train``."""
    _train_main()


if __name__ == "__main__":  # pragma: no cover
    main()
