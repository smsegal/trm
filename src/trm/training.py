"""Core training utilities for TRM models."""

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import TrainingConfig
from .data import (
    BOARD_VOCAB,
    SUDOKU_SEQ_LEN,
    build_dataloader,
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
        if max_batches is not None and total_batches >= max_batches:
            break

    mean_loss = total_loss / max(1, total_batches)
    accuracy = total_correct / max(1, total_tokens)
    return {"loss": mean_loss, "cell_accuracy": accuracy}




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

    eval_loader = None
    if config.eval_split is not None and not config.streaming:
        eval_dataset = prepare_eval_dataset(config)
        eval_loader = build_dataloader(eval_dataset, config=config)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
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
