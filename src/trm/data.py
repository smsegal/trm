"""Data processing utilities for TRM models."""

from typing import cast

import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

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


def sudoku_collate_fn(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function to convert lists of token ids into batched tensors."""
    puzzles = torch.tensor([ex["puzzle"] for ex in batch], dtype=torch.long)
    solutions = torch.tensor([ex["solution"] for ex in batch], dtype=torch.long)
    return puzzles, solutions


def prepare_dataset(config):
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


def build_dataloader(dataset, *, config):
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


def prepare_eval_dataset(config):
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
