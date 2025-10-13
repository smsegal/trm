"""Core training utilities for TRM models."""

from typing import cast

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from torch import Tensor

from .config import TrainingConfig
from .model import TRM

BOARD_VOCAB = ".123456789"
TOKEN_LOOKUP = {token: index for index, token in enumerate(BOARD_VOCAB)}
SUDOKU_SEQ_LEN = 81
PUZZLE_COLUMN_CANDIDATES = ("puzzle", "question", "input", "board")
SOLUTION_COLUMN_CANDIDATES = ("solution", "answer", "target", "output")


def _resolve_split(
    dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict,
    split: str,
) -> Dataset | IterableDataset:
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        return cast(Dataset | IterableDataset, dataset[split])
    return cast(Dataset | IterableDataset, dataset)


def _determine_remove_columns(dataset: Dataset | IterableDataset) -> list[str] | None:
    column_names = getattr(dataset, "column_names", None)
    if column_names is None:
        return None
    return [column for column in column_names if column not in ("puzzle", "solution")]


def _extract_board(example: dict, *, candidates: tuple[str, ...], field_name: str) -> str:
    for key in candidates:
        if key in example:
            value = example[key]
            if isinstance(value, str):
                return value
            raise TypeError(f"{field_name} field {key!r} must be a string, received {type(value).__name__}.")
    available = ", ".join(sorted(example.keys()))
    raise KeyError(f"Could not find {field_name} field in example. Available keys: {available}.")


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
    puzzle_text = _extract_board(example, candidates=PUZZLE_COLUMN_CANDIDATES, field_name="puzzle")
    solution_text = _extract_board(example, candidates=SOLUTION_COLUMN_CANDIDATES, field_name="solution")
    return {
        "puzzle": encode_board(puzzle_text),
        "solution": encode_board(solution_text),
    }


def sudoku_collate_fn(batch: list[dict]) -> tuple[Tensor, Tensor]:
    """Collate function to convert lists of token ids into batched tensors."""
    puzzles = torch.tensor([ex["puzzle"] for ex in batch], dtype=torch.long)
    solutions = torch.tensor([ex["solution"] for ex in batch], dtype=torch.long)
    return puzzles, solutions


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
    raw_dataset = load_dataset(config.dataset_name, split=config.split, streaming=config.streaming)
    dataset = _resolve_split(raw_dataset, config.split)

    if config.streaming:
        iterable_dataset = cast(IterableDataset, dataset)
        return iterable_dataset.map(encode_example)

    hf_dataset = cast(Dataset, dataset)
    if config.max_train_samples is not None:
        hf_dataset = hf_dataset.select(range(config.max_train_samples))
    hf_dataset = hf_dataset.shuffle(seed=config.seed)
    remove_columns = _determine_remove_columns(hf_dataset)
    if remove_columns is None:
        return hf_dataset.map(encode_example)
    return hf_dataset.map(encode_example, remove_columns=remove_columns)


def build_dataloader(dataset, *, config: TrainingConfig):
    """Create a batch iterator from a HuggingFace dataset using its native batching."""

    if not hasattr(dataset, "batch"):
        msg = "build_dataloader expects a Hugging Face dataset with a batch() method."
        raise TypeError(msg)

    batched_dataset = dataset.batch(config.batch_size)

    def _create_batches():
        for batch in batched_dataset:
            puzzles = torch.tensor(batch["puzzle"], dtype=torch.long)
            solutions = torch.tensor(batch["solution"], dtype=torch.long)
            yield puzzles, solutions

    return _create_batches()


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


def prepare_eval_dataset(config: TrainingConfig):
    """Load and prepare the evaluation dataset."""
    if config.eval_split is None:
        raise ValueError("eval_split must be set before preparing evaluation data.")
    raw_dataset = load_dataset(config.dataset_name, split=config.eval_split, streaming=config.streaming)
    dataset = _resolve_split(raw_dataset, config.eval_split)

    if config.streaming:
        iterable_dataset = cast(IterableDataset, dataset)
        return iterable_dataset.map(encode_example)

    hf_dataset = cast(Dataset, dataset)
    if config.max_eval_samples is not None:
        hf_dataset = hf_dataset.select(range(config.max_eval_samples))
    remove_columns = _determine_remove_columns(hf_dataset)
    if remove_columns is None:
        return hf_dataset.map(encode_example)
    return hf_dataset.map(encode_example, remove_columns=remove_columns)


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
