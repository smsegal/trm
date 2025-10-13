"""Data processing utilities for TRM models."""

import os
from collections.abc import Sequence

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


def prepare_ds(
    name: str,
    split: str,
    columns: dict[str, str],
    stream: bool = False,
    seed: int | None = None,
):
    dataset = load_dataset(name, split=split, streaming=stream)
    assert isinstance(dataset, Dataset)
    
    # Keep only columns that are in the mapping
    cols_to_remove = [col for col in dataset.column_names if col not in columns]
    dataset = dataset.remove_columns(cols_to_remove).rename_columns(columns).shuffle(seed)
    
    return dataset


def build_dataloader(dataset: Dataset, batch_size: int, device: torch.device = torch.device("cpu"), num_proc: int = -1):
    """num_proc==-1 means autodetect"""
    if num_proc == -1:
        # if os.cpu_count can't return a count, just assume 0?
        num_proc = os.cpu_count() or 0
    return dataset.batch(batch_size=batch_size, num_proc=num_proc).with_format("torch", device=device)
