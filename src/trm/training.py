"""Core training utilities for TRM models."""

from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import TrainingConfig
from .data import (
    BOARD_VOCAB,
    SUDOKU_SEQ_LEN,
    build_dataloader,
    decode_board,
    format_board,
    prepare_dataset,
    prepare_eval_dataset,
)
from .model import TRM


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


class CheckpointManager:
    """Manage persistence of recent and best-performing checkpoints."""

    def __init__(
        self,
        directory: Path,
        keep_last: int,
        *,
        metric_name: str,
        maximize: bool,
    ) -> None:
        self.directory = directory
        self.keep_last = max(0, keep_last)
        self.metric_name = metric_name
        self.maximize = maximize
        self.recent: deque[Path] = deque()
        self.best_metric: float | None = None
        self.best_path = self.directory / "best.pt"
        self.directory.mkdir(parents=True, exist_ok=True)

    def _is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.maximize:
            return metric > self.best_metric
        return metric < self.best_metric

    def save(
        self,
        *,
        epoch: int,
        model: TRM,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, Any] | None,
    ) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
        }

        if self.keep_last > 0:
            checkpoint_path = self.directory / f"epoch-{epoch:04d}.pt"
            torch.save(payload, checkpoint_path)
            self.recent.append(checkpoint_path)
            while len(self.recent) > self.keep_last:
                old_path = self.recent.popleft()
                if old_path.exists():
                    old_path.unlink()

        metric_value = None
        if metrics is not None and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]

        if metric_value is None:
            return

        metric_float = float(metric_value)
        if self._is_better(metric_float):
            torch.save(payload, self.best_path)
            self.best_metric = metric_float




def training_step(
    model: TRM,
    optimizer: torch.optim.Optimizer,
    puzzles: Tensor,
    solutions: Tensor,
    *,
    config: TrainingConfig,
) -> tuple[float, Tensor | float]:
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
    dataloader,
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

    preview_examples: list[tuple[list[int], list[int], list[int]]] = []
    preview_limit = max(0, config.eval_preview_examples)

    for batch_puzzles, batch_solutions in dataloader:
        puzzles = batch_puzzles.to(device)
        solutions = batch_solutions.to(device)
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
        if preview_limit > 0 and len(preview_examples) < preview_limit:
            puzzle_tokens = puzzles.detach().cpu().tolist()
            solution_tokens = solutions.detach().cpu().tolist()
            prediction_tokens = preds.detach().cpu().tolist()
            for puzzle_item, solution_item, prediction_item in zip(puzzle_tokens, solution_tokens, prediction_tokens):
                if len(preview_examples) >= preview_limit:
                    break
                preview_examples.append(
                    (
                        puzzle_item,
                        solution_item,
                        prediction_item,
                    )
                )
        if max_batches is not None and total_batches >= max_batches:
            break

    mean_loss = total_loss / max(1, total_batches)
    accuracy = total_correct / max(1, total_tokens)
    metrics = {"loss": mean_loss, "cell_accuracy": accuracy}

    if preview_examples:
        print("Evaluation samples:")
        for index, (puzzle_tokens, solution_tokens, prediction_tokens) in enumerate(preview_examples, start=1):
            puzzle_board = decode_board(puzzle_tokens)
            prediction_board = decode_board(prediction_tokens)
            solution_board = decode_board(solution_tokens)
            print(f"Sample {index}:")
            print("Puzzle:")
            print(format_board(puzzle_board))
            print("Predicted:")
            print(format_board(prediction_board))
            print("Solution:")
            print(format_board(solution_board))
            print()
    return metrics




def run_training(config: TrainingConfig) -> None:
    torch.manual_seed(config.seed)
    dataset = prepare_dataset(config)
    device = resolve_device(config.device)
    model = build_model(config).to(device)
    dataloader = build_dataloader(dataset, config=config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_manager: CheckpointManager | None = None
    if config.checkpoint_dir is not None:
        checkpoint_manager = CheckpointManager(
            Path(config.checkpoint_dir),
            config.keep_last_n_checkpoints,
            metric_name=config.checkpoint_metric,
            maximize=config.checkpoint_metric_higher_is_better,
        )

    eval_loader = None
    if config.eval_split is not None and not config.streaming:
        eval_dataset = prepare_eval_dataset(config)
        eval_loader = build_dataloader(eval_dataset, config=config)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        epoch_metrics: dict[str, float] | None = None
        for batch_puzzles, batch_solutions in dataloader:
            puzzles = batch_puzzles.to(device)
            solutions = batch_solutions.to(device)
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
            if isinstance(halting_logits, torch.Tensor):
                halting_metric = float(halting_logits.detach().max().item())
            else:
                halting_metric = float(halting_logits)
            if halting_metric > 0 and config.max_steps_per_epoch is not None and steps >= config.max_steps_per_epoch:
                break
        if steps and steps % config.log_every != 0:
            avg_loss = running_loss / (steps % config.log_every)
            print(f"[Epoch {epoch + 1}] Step {steps} - mean loss {avg_loss:.4f}")

        if config.eval_split is not None:
            if config.streaming:
                eval_dataset = prepare_eval_dataset(config)
                eval_loader = build_dataloader(eval_dataset, config=config)
            assert eval_loader is not None
            epoch_metrics = evaluate(
                model,
                eval_loader,
                device=device,
                config=config,
                max_batches=config.eval_batches,
            )
            print(
                f"[Epoch {epoch + 1}] Evaluation - loss {epoch_metrics['loss']:.4f}, "
                f"cell accuracy {epoch_metrics['cell_accuracy'] * 100:.2f}%"
            )
        if checkpoint_manager is not None:
            checkpoint_manager.save(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                metrics=epoch_metrics,
            )
