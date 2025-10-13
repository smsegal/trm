"""CLI entrypoint for running TRM training jobs."""

from collections.abc import Sequence
import sys

import tyro

from trm.config import TrainingConfig, iter_cli_config_options
from trm.training import run_training

_CONFIG_PREFIX_FLAGS = {"--config", "-c"}


def _normalize_cli_args(argv: Sequence[str] | None) -> list[str]:
    """Normalize legacy flags into Tyro subcommand arguments."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        return ["default"]

    config_name: str | None = None
    remaining: list[str] = []
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg in _CONFIG_PREFIX_FLAGS:
            if i + 1 >= len(raw_args):
                raise SystemExit("--config flag requires a value.")
            config_name = raw_args[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            config_name = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1

    if config_name is not None:
        remaining.insert(0, config_name)

    if not remaining or remaining[0].startswith("-"):
        remaining.insert(0, "default")

    return remaining


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """Parse CLI arguments into a TrainingConfig using Tyro."""
    config_options = iter_cli_config_options()
    tyro_args = _normalize_cli_args(argv)
    return tyro.extras.overridable_config_cli(
        config_options,
        prog="trm-train",
        description="Train TRM on the sudoku-extreme dataset.",
        args=tyro_args,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    run_training(config)


if __name__ == "__main__":  # pragma: no cover
    main()
