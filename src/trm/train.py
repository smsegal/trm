"""CLI entrypoint for running TRM training jobs."""

import argparse
from collections.abc import Sequence

from trm.config import CONFIG_PRESETS, TrainingConfig, apply_overrides
from trm.training import run_training


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train TRM on the sudoku-extreme dataset.")
    parser.add_argument(
        "--config",
        type=str,
        choices=sorted(CONFIG_PRESETS),
        default="macbook",
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

    overrides: dict[str, object] = {}

    data_arg_map = {
        "dataset_name": "dataset_name",
        "split": "train_split",
        "eval_split": "eval_split",
    }
    model_arg_map = {
        "hidden_dim": "hidden_dim",
        "n_layers": "n_layers",
        "num_heads": "num_heads",
        "dropout": "dropout",
    }
    training_arg_map = {
        "batch_size": "batch_size",
        "epochs": "epochs",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "max_grad_norm": "max_grad_norm",
        "latent_steps": "latent_steps",
        "recurse": "recurse",
        "log_every": "log_every",
        "max_train_samples": "max_train_samples",
        "max_steps_per_epoch": "max_steps_per_epoch",
        "max_eval_samples": "max_eval_samples",
        "eval_batches": "eval_batches",
        "seed": "seed",
        "num_workers": "num_workers",
        "device": "device",
    }

    for arg_name, override_key in data_arg_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[override_key] = value

    for arg_name, override_key in model_arg_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[override_key] = value

    for arg_name, override_key in training_arg_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[override_key] = value

    if args.streaming is not None:
        overrides["streaming"] = args.streaming

    config = apply_overrides(base_config, overrides)
    return config


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    run_training(config)


if __name__ == "__main__":  # pragma: no cover
    main()
