from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader

from trm.model import TRM

BOARD_VOCAB = ".123456789"
TOKEN_LOOKUP = {token: index for index, token in enumerate(BOARD_VOCAB)}
SUDOKU_SEQ_LEN = 81


def encode_board(board: str) -> list[int]:
    """Convert a flattened sudoku board string into token ids."""
    cleaned = board.strip().replace("\n", "").replace(" ", "")
    if len(cleaned) != SUDOKU_SEQ_LEN:
        raise ValueError(f"Sudoku board must be length {SUDOKU_SEQ_LEN}, received {len(cleaned)}.")
    try:
        tokens = [TOKEN_LOOKUP[char] for char in cleaned]
    except KeyError as exc:
        raise ValueError(f"Unexpected character {exc.args[0]!r} in sudoku board.") from exc
    return tokens


def encode_example(example: dict) -> dict:
    """Encode a single example's puzzle and solution fields."""
    return {
        "puzzle": encode_board(example["puzzle"]),
        "solution": encode_board(example["solution"]),
    }


def sudoku_collate_fn(batch: list[dict]) -> tuple[Tensor, Tensor]:
    """Collate function to convert lists of token ids into batched tensors."""
    puzzles = torch.tensor([ex["puzzle"] for ex in batch], dtype=torch.long)
    solutions = torch.tensor([ex["solution"] for ex in batch], dtype=torch.long)
    return puzzles, solutions


@dataclass
class TrainingConfig:
    dataset_name: str = "sapientinc/sudoku-extreme"
    split: str = "train"
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hidden_dim: int = 256
    n_layers: int = 4
    num_heads: int | None = None
    dropout: float = 0.1
    latent_steps: int = 4
    recurse: int = 6
    log_every: int = 50
    max_train_samples: int | None = None
    max_steps_per_epoch: int | None = None
    eval_split: str | None = "test"
    max_eval_samples: int | None = 4096
    eval_batches: int | None = None
    streaming: bool = False
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


def resolve_device(device_option: str) -> torch.device:
    if device_option == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_option)


def build_model(config: TrainingConfig) -> TRM:
    vocab_size = len(BOARD_VOCAB)
    model = TRM(
        n_layers=config.n_layers,
        vocab_size=vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        max_position_embeddings=SUDOKU_SEQ_LEN,
        dropout=config.dropout,
    )
    return model


def prepare_dataset(config: TrainingConfig):
    """Load and prepare the training dataset."""
    dataset = load_dataset(config.dataset_name, split=config.split, streaming=config.streaming)
    if not config.streaming:
        if config.max_train_samples is not None:
            dataset = dataset.select(range(config.max_train_samples))
        dataset = dataset.shuffle(seed=config.seed)
    dataset = dataset.map(encode_example, remove_columns=["puzzle", "solution"])
    return dataset


def build_dataloader(dataset, *, config: TrainingConfig, is_streaming: bool) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a DataLoader from a HuggingFace dataset."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not is_streaming,
        num_workers=config.num_workers,
        collate_fn=sudoku_collate_fn,
    )


def training_step(
    model: TRM,
    optimizer: torch.optim.Optimizer,
    puzzles: Tensor,
    solutions: Tensor,
    *,
    config: TrainingConfig,
) -> float:
    x_hidden = model.embedding(puzzles)
    zeros = torch.zeros_like(x_hidden)
    _, _, lm_logits, halting_logits = model.deep_recursion(
        x_hidden,
        zeros,
        zeros,
        n=config.latent_steps,
        recurse=config.recurse,
    )
    loss = F.cross_entropy(
        lm_logits.reshape(-1, lm_logits.size(-1)),
        solutions.view(-1),
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
    optimizer.step()
    return float(loss.item()), halting_logits


@torch.no_grad()
def evaluate(
    model: TRM,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
    config: TrainingConfig,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_tokens = 0

    for batch_index, (puzzles, solutions) in enumerate(dataloader):
        puzzles = puzzles.to(device)
        solutions = solutions.to(device)
        x_hidden = model.embedding(puzzles)
        zeros = torch.zeros_like(x_hidden)
        _, _, lm_logits, _ = model.deep_recursion(
            x_hidden,
            zeros,
            zeros,
            n=config.latent_steps,
            recurse=config.recurse,
        )
        loss = F.cross_entropy(
            lm_logits.reshape(-1, lm_logits.size(-1)),
            solutions.view(-1),
        )
        preds = lm_logits.argmax(dim=-1)
        total_correct += (preds == solutions).sum().item()
        total_tokens += solutions.numel()
        total_loss += loss.item()
        total_batches += 1
        if max_batches is not None and (batch_index + 1) >= max_batches:
            break

    mean_loss = total_loss / max(1, total_batches)
    accuracy = total_correct / max(1, total_tokens)
    return {"loss": mean_loss, "cell_accuracy": accuracy}


def prepare_eval_dataset(config: TrainingConfig):
    """Load and prepare the evaluation dataset."""
    if config.eval_split is None:
        raise ValueError("eval_split must be set before preparing evaluation data.")
    dataset = load_dataset(config.dataset_name, split=config.eval_split, streaming=config.streaming)
    if not config.streaming and config.max_eval_samples is not None:
        dataset = dataset.select(range(config.max_eval_samples))
    dataset = dataset.map(encode_example, remove_columns=["puzzle", "solution"])
    return dataset


def run_training(config: TrainingConfig) -> None:
    torch.manual_seed(config.seed)
    dataset = prepare_dataset(config)
    device = resolve_device(config.device)
    model = build_model(config).to(device)
    dataloader = build_dataloader(dataset, config=config, is_streaming=config.streaming)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    eval_loader: DataLoader[tuple[Tensor, Tensor]] | None = None
    if config.eval_split is not None and not config.streaming:
        eval_dataset = prepare_eval_dataset(config)
        eval_loader = build_dataloader(eval_dataset, config=config, is_streaming=False)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        for puzzles, solutions in dataloader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            loss, halting_logits = training_step(
                model,
                optimizer,
                puzzles,
                solutions,
                config=config,
            )
            running_loss += loss
            steps += 1
            if steps % config.log_every == 0:
                avg_loss = running_loss / config.log_every
                print(f"[Epoch {epoch + 1}] Step {steps} - mean loss {avg_loss:.4f}")
                running_loss = 0.0
            if halting_logits > 0 and config.max_steps_per_epoch is not None and steps >= config.max_steps_per_epoch:
                break
        if steps and steps % config.log_every != 0:
            avg_loss = running_loss / (steps % config.log_every)
            print(f"[Epoch {epoch + 1}] Step {steps} - mean loss {avg_loss:.4f}")

        if config.eval_split is not None:
            if config.streaming:
                eval_dataset = prepare_eval_dataset(config)
                eval_loader = build_dataloader(eval_dataset, config=config, is_streaming=True)
            assert eval_loader is not None
            metrics = evaluate(
                model,
                eval_loader,
                device=device,
                config=config,
                max_batches=config.eval_batches,
            )
            print(
                f"[Epoch {epoch + 1}] Evaluation - loss {metrics['loss']:.4f}, "
                f"cell accuracy {metrics['cell_accuracy'] * 100:.2f}%"
            )


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train TRM on the sudoku-extreme dataset.")
    parser.add_argument("--dataset-name", type=str, default="sapientinc/sudoku-extreme")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--latent-steps", type=int, default=4)
    parser.add_argument("--recurse", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--max-eval-samples", type=int, default=4096)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args(argv)
    return TrainingConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        latent_steps=args.latent_steps,
        recurse=args.recurse,
        log_every=args.log_every,
        max_train_samples=args.max_train_samples,
        max_steps_per_epoch=args.max_steps_per_epoch,
        eval_split=args.eval_split,
        max_eval_samples=args.max_eval_samples,
        eval_batches=args.eval_batches,
        streaming=args.streaming,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    run_training(config)


__all__ = [
    "BOARD_VOCAB",
    "SUDOKU_SEQ_LEN",
    "TrainingConfig",
    "encode_board",
    "encode_example",
    "main",
    "run_training",
    "sudoku_collate_fn",
]


if __name__ == "__main__":  # pragma: no cover
    main()
