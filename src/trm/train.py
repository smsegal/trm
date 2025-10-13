"""CLI entrypoint for running TRM training jobs."""

import argparse
from collections.abc import Sequence
from dataclasses import fields, replace

from trm.config import CONFIG_PRESETS, TrainingConfig
from trm.training import run_training


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train TRM on the sudoku-extreme dataset.")
    parser.add_argument(
        "--config",
        type=str,
        choices=sorted(CONFIG_PRESETS),
        help="Choose a preset configuration (default, macbook, gpu).",
    )
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--latent-steps", type=int)
    parser.add_argument("--recurse", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--eval-split", type=str)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", type=str)

    args = parser.parse_args(argv)

    if args.config is not None:
        base_config = CONFIG_PRESETS[args.config]()
    else:
        base_config = TrainingConfig()

    overrides = {}
    for field in fields(TrainingConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            overrides[field.name] = value

    config = replace(base_config, **overrides)
    if args.streaming is not None:
        config = replace(config, streaming=args.streaming)
    return config


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    run_training(config)


if __name__ == "__main__":  # pragma: no cover
    main()
