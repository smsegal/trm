"""Data processing utilities for TRM models."""

import os
from collections.abc import Callable, Sequence

import torch
from datasets import Dataset, load_dataset

BOARD_VOCAB = ".123456789"
TOKEN_LOOKUP = {token: index for index, token in enumerate(BOARD_VOCAB)}
TOKEN_REVERSE_LOOKUP = {index: token for token, index in TOKEN_LOOKUP.items()}
SUDOKU_SEQ_LEN = 81


def format_board(board: str) -> str:
    """Pretty-print a flattened sudoku board as a 9x9 grid."""
    cleaned = board.strip().replace("\n", "").replace(" ", "")
    if len(cleaned) != SUDOKU_SEQ_LEN:
        raise ValueError(f"Sudoku board must be length {SUDOKU_SEQ_LEN}, received {len(cleaned)}.")
    rows = [cleaned[i : i + 9] for i in range(0, SUDOKU_SEQ_LEN, 9)]
    return "\n".join(rows)


def decode_board(tokens: Sequence[int]) -> str:
    """Convert token ids back into a flattened sudoku board string."""
    try:
        characters = [TOKEN_REVERSE_LOOKUP[int(token)] for token in tokens]
    except KeyError as exc:
        raise ValueError(f"Unexpected token id {exc.args[0]!r} in sudoku encoding.") from exc
    return "".join(characters)


def tokenize(examples: dict[str, list[str]], columns: dict[str, str]) -> dict[str, list[list[int]]]:
    return {
        new_col: [[TOKEN_LOOKUP[t] for t in text] for text in examples[original_col]]
        for new_col, original_col in columns.items()
    }


def prepare_ds(
    name: str,
    split: str,
    columns: dict[str, str],
    mapper: Callable,
    stream: bool = False,
    num_proc: int = -1,
    seed: int | None = None,
):
    dataset = load_dataset(name, split=split, streaming=stream)
    assert isinstance(dataset, Dataset)
    if num_proc == -1:
        # if os.cpu_count can't return a count, just assume 0?
        num_proc = os.cpu_count() or 0
    dataset = dataset.map(
        mapper, fn_kwargs={"columns": columns}, num_proc=num_proc, batched=True, remove_columns=dataset.column_names
    ).with_format("torch")
    return dataset


def build_dataloader(dataset: Dataset, batch_size: int, device: torch.device = torch.device("cpu"), num_proc: int = -1):
    """num_proc==-1 means autodetect"""
    if num_proc == -1:
        # if os.cpu_count can't return a count, just assume 0?
        num_proc = os.cpu_count() or 0
    return dataset.batch(batch_size=batch_size, num_proc=num_proc).shuffle().with_format("torch")
