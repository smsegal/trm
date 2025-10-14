"""CLI entrypoint for running TRM training jobs."""

import tyro
from collections.abc import Sequence

from trm.config import CONFIG_PRESETS, TrainingConfig, apply_overrides
from trm.training import run_training


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """Parse command-line arguments using tyro."""
    # Use tyro to parse the TrainingConfig dataclass
    config = tyro.cli(
        TrainingConfig,
        description="Train TRM on the sudoku-extreme dataset.",
    )
    # Apply the preset if specified
    if config.config in CONFIG_PRESETS:
        preset_config = CONFIG_PRESETS[config.config]()
        # Override with user-specified values
        config = apply_overrides(preset_config, {k: v for k, v in config.__dict__.items() if k != 'config'})
    return config


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    run_training(config)


if __name__ == "__main__":  # pragma: no cover
    main()
