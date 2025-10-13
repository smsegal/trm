"""Interactive CLI for viewing sudoku data and model predictions."""

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Sequence

import torch

from .config import CONFIG_PRESETS, TrainingConfig
from .data import decode_board, format_board
from .inspection import SudokuSample, pull_sudoku_samples
from .training import build_model, resolve_device


def _build_config(args: argparse.Namespace) -> TrainingConfig:
    if args.config is not None:
        base = CONFIG_PRESETS[args.config]()
    else:
        base = TrainingConfig()

    overrides = {}
    for field in fields(TrainingConfig):
        if field.name in {"epochs", "max_train_samples", "max_steps_per_epoch", "eval_split", "max_eval_samples", "eval_batches"}:
            continue
        if hasattr(args, field.name):
            value = getattr(args, field.name)
            if value is not None:
                overrides[field.name] = value
    config = replace(base, **overrides)
    return config


def _load_model(config: TrainingConfig, checkpoint: Path | None, device: torch.device) -> torch.nn.Module:
    model = build_model(config).to(device)
    if checkpoint is None:
        model.eval()
        return model
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_solution(
    model: torch.nn.Module,
    sample: SudokuSample,
    *,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[str, float]:
    model.eval()
    with torch.no_grad():
        puzzle = torch.tensor(sample.encoded_puzzle, dtype=torch.long, device=device).unsqueeze(0)
        x_hidden = model.embedding(puzzle)
        zeros = torch.zeros_like(x_hidden)
        _, _, lm_logits, _ = model.deep_recursion(
            x_hidden,
            zeros,
            zeros,
            n=config.latent_steps,
            recurse=config.recurse,
        )
        preds = lm_logits.argmax(dim=-1).squeeze(0).tolist()
    predicted_board = decode_board(preds)
    matches = sum(1 for truth, pred in zip(sample.answer, predicted_board) if truth == pred)
    accuracy = matches / len(preds)
    return predicted_board, accuracy


def _print_sample(sample: SudokuSample) -> None:
    print(f"Source : {sample.source}")
    print(f"Rating : {sample.rating}")
    print("Puzzle :")
    print(format_board(sample.question))


def _print_solution(sample: SudokuSample) -> None:
    print("Solution:")
    print(format_board(sample.answer))


def _print_prediction(predicted: str, sample: SudokuSample, accuracy: float) -> None:
    print("Model Prediction:")
    print(format_board(predicted))
    print(f"Cell accuracy: {accuracy * 100:.2f}%")

    print("Ground Truth:")
    print(format_board(sample.answer))


def _interactive_loop(
    samples: list[SudokuSample],
    model: torch.nn.Module,
    *,
    device: torch.device,
    config: TrainingConfig,
) -> None:
    if not samples:
        print("No samples available to display.")
        return
    index = 0
    print("Interactive viewer controls: [Enter]/n next, p previous, s show solution, m run model, q quit.")
    while True:
        sample = samples[index]
        print()
        print(f"=== Sample {index + 1} of {len(samples)} ===")
        _print_sample(sample)

        command = input("Command [n/p/s/m/q]: ").strip().lower()
        if command in {"", "n"}:
            index = (index + 1) % len(samples)
            continue
        if command == "p":
            index = (index - 1) % len(samples)
            continue
        if command == "s":
            _print_solution(sample)
            continue
        if command == "m":
            predicted, accuracy = _predict_solution(model, sample, device=device, config=config)
            _print_prediction(predicted, sample, accuracy)
            continue
        if command == "q":
            print("Exiting viewer.")
            break
        print("Unrecognised command. Use n, p, s, m, or q.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive viewer for sudoku-extreme data and model outputs.")
    parser.add_argument("--config", type=str, choices=sorted(CONFIG_PRESETS), help="Base configuration preset to use.")
    parser.add_argument("--dataset_name", dest="dataset_name", type=str, help="Dataset identifier.")
    parser.add_argument("--split", type=str, help="Dataset split to sample.")
    parser.add_argument("--seed", type=int, help="Random seed for sampling.")
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, help="Transformer hidden dimension.")
    parser.add_argument("--n_layers", dest="n_layers", type=int, help="Number of transformer layers.")
    parser.add_argument("--num_heads", dest="num_heads", type=int, help="Number of attention heads.")
    parser.add_argument("--latent_steps", dest="latent_steps", type=int, help="Latent recursion steps.")
    parser.add_argument("--recurse", type=int, help="Number of recursion passes.")
    parser.add_argument("--dropout", type=float, help="Dropout probability.")
    parser.add_argument("--device", type=str, help="Device override (cpu, cuda, mps, auto).")
    parser.add_argument("--count", type=int, default=5, help="How many samples to load.")
    parser.add_argument("--checkpoint", type=Path, help="Optional path to a model checkpoint.")
    parser.add_argument("--no_shuffle", action="store_true", help="Keep dataset order instead of shuffling.")
    args = parser.parse_args(argv)

    config = _build_config(args)

    seed = None if args.no_shuffle else config.seed
    samples = pull_sudoku_samples(
        dataset_name=config.dataset_name,
        split=config.split,
        count=args.count,
        seed=seed,
    )

    device = resolve_device(config.device)
    model = _load_model(config, args.checkpoint, device)
    _interactive_loop(samples, model, device=device, config=config)


if __name__ == "__main__":  # pragma: no cover
    main()
