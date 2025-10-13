"""Core training utilities for TRM models."""

import math
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from .config import TrainingConfig
from .data import (
    BOARD_VOCAB,
    SUDOKU_SEQ_LEN,
    build_dataloader,
    decode_board,
    prepare_ds,
)
from .display import console, render_sudoku
from .model import TRM

EMPTY_TOKEN_ID = BOARD_VOCAB.index(".")


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
    model_config = config.model
    model = TRM(
        n_layers=model_config.n_layers,
        vocab_size=vocab_size,
        hidden_dim=model_config.hidden_dim,
        num_heads=model_config.num_heads,
        max_position_embeddings=SUDOKU_SEQ_LEN,
        dropout=model_config.dropout,
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
        self._history_limit = max(1, self.keep_last)
        self.metric_name = metric_name
        self.maximize = maximize
        self.recent: deque[Path] = deque()
        self.best_metric: float | None = None
        self.best_path = self.directory / "best.pt"
        self.latest_path = self.directory / "latest.pt"
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

        checkpoint_path = self.directory / f"epoch-{epoch:04d}.pt"
        torch.save(payload, checkpoint_path)
        self.recent.append(checkpoint_path)
        while len(self.recent) > self._history_limit:
            old_path = self.recent.popleft()
            if old_path.exists():
                old_path.unlink()

        if self.latest_path.exists() or self.latest_path.is_symlink():
            self.latest_path.unlink()
        try:
            self.latest_path.symlink_to(checkpoint_path.name)
        except OSError:
            torch.save(payload, self.latest_path)

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
    x: Tensor,
    y_true: Tensor,
    *,
    config: TrainingConfig,
) -> tuple[float, Tensor | float]:
    x_hidden = model.embedding(x)
    zeros = torch.zeros_like(x_hidden)
    y, z, y_hat, q_hat = model.deep_recursion(
        x_hidden,
        zeros,
        zeros,
        n=config.model.latent_steps,
        recurse=config.model.recurse,
    )
    loss_class = F.cross_entropy(
        y_hat.reshape(-1, y_hat.size(-1)),
        y_true.view(-1),
    )
    loss_halt = F.binary_cross_entropy(q_hat.squeeze(dim=-1), (y_hat.argmax(dim=-1) == y_true))
    loss = loss_class + loss_halt
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return float(loss.item()), q_hat


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
            n=config.model.latent_steps,
            recurse=config.model.recurse,
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
        console.print("[bold]Evaluation samples:[/bold]")
        for index, (puzzle_tokens, solution_tokens, prediction_tokens) in enumerate(preview_examples, start=1):
            puzzle_board = decode_board(puzzle_tokens)
            prediction_display_tokens = prediction_tokens.copy()
            given_indices = {idx for idx, token in enumerate(puzzle_tokens) if token != EMPTY_TOKEN_ID}
            for idx in given_indices:
                prediction_display_tokens[idx] = puzzle_tokens[idx]
            prediction_board = decode_board(prediction_display_tokens)
            solution_board = decode_board(solution_tokens)
            correct_indices = {
                idx
                for idx, (pred_token, sol_token, puzzle_token) in enumerate(
                    zip(prediction_tokens, solution_tokens, puzzle_tokens)
                )
                if puzzle_token == EMPTY_TOKEN_ID and pred_token == sol_token
            }
            incorrect_indices = {
                idx
                for idx, (pred_token, sol_token, puzzle_token) in enumerate(
                    zip(prediction_tokens, solution_tokens, puzzle_tokens)
                )
                if puzzle_token == EMPTY_TOKEN_ID and pred_token != sol_token
            }
            prediction_styles: dict[int, str] = {}
            for idx in given_indices:
                prediction_styles[idx] = "dim cyan"
            for idx in correct_indices:
                prediction_styles[idx] = "green"
            for idx in incorrect_indices:
                prediction_styles[idx] = "bold red"

            console.print(f"[dim]Sample {index}[/dim]")
            render_sudoku(puzzle_board, title="Puzzle", puzzle=True, border_style="cyan")
            render_sudoku(
                prediction_board,
                title="Prediction",
                border_style="magenta",
                cell_styles=prediction_styles,
            )
            render_sudoku(solution_board, title="Solution", border_style="green")
    return metrics


def run_training(config: TrainingConfig) -> None:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    dataset = prepare_ds(
        name=config.data.dataset_name, split="train", columns=config.data.columns, stream=config.data.streaming
    )
    model = build_model(config).to(device)

    steps_per_epoch_estimate: int | None = None
    if config.max_steps_per_epoch is not None:
        steps_per_epoch_estimate = config.max_steps_per_epoch
    else:
        dataset_length = len(dataset)
        steps_per_epoch_estimate = math.ceil(dataset_length / config.batch_size)

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

    eval_dataset = None
    if config.data.eval_split is not None and not config.data.streaming:
        eval_dataset = prepare_ds(config.data.dataset_name, config.data.eval_split, columns=config.data.columns)

    for epoch in range(config.epochs):
        model.train()
        dataloader = build_dataloader(dataset, batch_size=config.batch_size, device=device)
        loss_window: deque[float] = deque(maxlen=max(1, config.log_every))
        time_window: deque[float] = deque(maxlen=max(1, config.log_every))
        steps = 0
        epoch_metrics: dict[str, float] | None = None
        progress_total = config.max_steps_per_epoch or steps_per_epoch_estimate
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            unit="step",
            total=progress_total,
            dynamic_ncols=True,
        )
        try:
            for x_input, y_labels in progress_bar:
                step_start = perf_counter()
                loss, halting_logits = training_step(
                    model,
                    optimizer,
                    x_input,
                    y_labels,
                    config=config,
                )
                step_duration = perf_counter() - step_start
                steps += 1
                loss_window.append(loss)
                time_window.append(step_duration)
                mean_loss = sum(loss_window) / len(loss_window)
                mean_step_time = sum(time_window) / len(time_window)
                progress_bar.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    step_time=f"{mean_step_time * 1000:.1f} ms",
                )
                if isinstance(halting_logits, torch.Tensor):
                    halting_metric = float(halting_logits.detach().max().item())
                else:
                    halting_metric = float(halting_logits)
                if (
                    halting_metric > 0
                    and config.max_steps_per_epoch is not None
                    and steps >= config.max_steps_per_epoch
                ):
                    break
        finally:
            progress_bar.close()

        if config.data.eval_split is not None:
            assert eval_dataset is not None
            eval_loader = build_dataloader(
                eval_dataset,
                batch_size=config.batch_size,
                device=device,
            )
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
