"""Data processing utilities for TRM models."""

from collections.abc import Sequence
from typing import cast

import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

BOARD_VOCAB = ".123456789"
TOKEN_LOOKUP = {token: index for index, token in enumerate(BOARD_VOCAB)}
TOKEN_REVERSE_LOOKUP = {index: token for token, index in TOKEN_LOOKUP.items()}
SUDOKU_SEQ_LEN = 81
RAW_PUZZLE_COLUMN = "question"
RAW_SOLUTION_COLUMN = "answer"
ENCODED_PUZZLE_COLUMN = "puzzle"
ENCODED_SOLUTION_COLUMN = "solution"


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
    return [column for column in column_names if column not in (ENCODED_PUZZLE_COLUMN, ENCODED_SOLUTION_COLUMN)]


def _extract_board(example: dict, *, raw_column: str, encoded_column: str) -> str:
    if raw_column in example:
        value = example[raw_column]
    elif encoded_column in example:
        value = example[encoded_column]
    else:
        available = ", ".join(sorted(example.keys()))
        raise KeyError(f"Could not find {raw_column!r} or {encoded_column!r} in example. Available keys: {available}.")
    if not isinstance(value, str):
        raise TypeError(
            f"{raw_column if raw_column in example else encoded_column} must be a string, "
            f"received {type(value).__name__}."
        )
    return value


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


def format_board(board: str) -> str:
    """Pretty-print a flattened sudoku board as a 9x9 grid."""
    cleaned = board.strip().replace("\n", "").replace(" ", "")
    if len(cleaned) != SUDOKU_SEQ_LEN:
        raise ValueError(f"Sudoku board must be length {SUDOKU_SEQ_LEN}, received {len(cleaned)}.")
    rows = [cleaned[i : i + 9] for i in range(0, SUDOKU_SEQ_LEN, 9)]
    return "\n".join(rows)


def encode_example(example: dict) -> dict:
    """Encode a single example's puzzle and solution fields."""
    puzzle_text = _extract_board(
        example,
        raw_column=RAW_PUZZLE_COLUMN,
        encoded_column=ENCODED_PUZZLE_COLUMN,
    )
    solution_text = _extract_board(
        example,
        raw_column=RAW_SOLUTION_COLUMN,
        encoded_column=ENCODED_SOLUTION_COLUMN,
    )
    return {
        ENCODED_PUZZLE_COLUMN: encode_board(puzzle_text),
        ENCODED_SOLUTION_COLUMN: encode_board(solution_text),
    }


def decode_board(tokens: Sequence[int]) -> str:
    """Convert token ids back into a flattened sudoku board string."""
    try:
        characters = [TOKEN_REVERSE_LOOKUP[int(token)] for token in tokens]
    except KeyError as exc:
        raise ValueError(f"Unexpected token id {exc.args[0]!r} in sudoku encoding.") from exc
    return "".join(characters)


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
