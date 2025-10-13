import itertools

import pytest
import torch
from torch.utils.data import IterableDataset

from trm.data import (
    BOARD_VOCAB,
    SUDOKU_SEQ_LEN,
    build_dataloader,
    encode_board,
    encode_example,
    prepare_dataset,
    prepare_eval_dataset,
    sudoku_collate_fn,
)


def make_board(pattern: str) -> str:
    repeat = (SUDOKU_SEQ_LEN + len(pattern) - 1) // len(pattern)
    board = (pattern * repeat)[:SUDOKU_SEQ_LEN]
    return board


def make_sudoku_rows(count: int) -> list[dict[str, str]]:
    return [
        {
            "puzzle": make_board("." * 9),
            "solution": make_board("123456789"),
        }
        for _ in range(count)
    ]


@pytest.fixture
def sudoku_rows() -> list[dict[str, str]]:
    return make_sudoku_rows(5)


class DummyHFDataset:
    def __init__(self, rows: list[dict[str, str | list[int]]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, str | list[int]]:
        return self._rows[idx]

    def shuffle(self, seed: int) -> "DummyHFDataset":
        import random

        random.seed(seed)
        rows = self._rows.copy()
        random.shuffle(rows)
        return DummyHFDataset(rows)

    def select(self, indices: range) -> "DummyHFDataset":
        return DummyHFDataset([self._rows[i] for i in indices])

    def map(self, func, remove_columns=None) -> "DummyHFDataset":
        mapped_rows = [func(row) for row in self._rows]
        return DummyHFDataset(mapped_rows)

    def batch(self, batch_size: int, drop_last_batch: bool = False) -> "DummyBatchedDataset":
        return DummyBatchedDataset(self, batch_size, drop_last_batch)


class DummyBatchedDataset:
    def __init__(self, dataset: DummyHFDataset, batch_size: int, drop_last_batch: bool = False) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_last_batch = drop_last_batch

    def __iter__(self):
        batch = []
        for i in range(len(self._dataset)):
            batch.append(self._dataset[i])
            if len(batch) >= self._batch_size:
                yield {
                    "puzzle": [item["puzzle"] for item in batch],
                    "solution": [item["solution"] for item in batch]
                }
                batch = []
        if batch and not self._drop_last_batch:
            yield {
                "puzzle": [item["puzzle"] for item in batch],
                "solution": [item["solution"] for item in batch]
            }


class DummyStreamingDataset(IterableDataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows

    def __iter__(self):
        yield from self._rows

    def map(self, func, remove_columns=None) -> "DummyStreamingMappedDataset":
        return DummyStreamingMappedDataset(self._rows, func)


class DummyStreamingMappedDataset(IterableDataset):
    def __init__(self, rows: list[dict[str, str]], func) -> None:
        super().__init__()
        self._rows = rows
        self._func = func

    def __iter__(self):
        for row in self._rows:
            yield self._func(row)

    def batch(self, batch_size: int, drop_last_batch: bool = False) -> "DummyStreamingBatchedDataset":
        return DummyStreamingBatchedDataset(self, batch_size, drop_last_batch)


class DummyStreamingBatchedDataset:
    def __init__(self, dataset: DummyStreamingMappedDataset, batch_size: int, drop_last_batch: bool = False) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_last_batch = drop_last_batch

    def __iter__(self):
        batch = []
        for item in self._dataset:
            batch.append(item)
            if len(batch) >= self._batch_size:
                yield {
                    "puzzle": [item["puzzle"] for item in batch],
                    "solution": [item["solution"] for item in batch]
                }
                batch = []
        if batch and not self._drop_last_batch:
            yield {
                "puzzle": [item["puzzle"] for item in batch],
                "solution": [item["solution"] for item in batch]
            }


def test_encode_board_valid() -> None:
    board = "123456789" * 9  # 81 characters
    tokens = encode_board(board)
    assert len(tokens) == SUDOKU_SEQ_LEN
    assert tokens[0] == 1  # "1" maps to 1
    assert tokens[80] == 9  # "9" maps to 9


def test_encode_board_invalid_length() -> None:
    board = "123"  # Too short
    with pytest.raises(ValueError, match="Sudoku board must be length 81"):
        encode_board(board)


def test_encode_board_invalid_character() -> None:
    board = ".123456788" * 8 + "X"  # 81 characters with invalid X
    with pytest.raises(ValueError, match="Unexpected character"):
        encode_board(board)


def test_encode_example() -> None:
    example = {
        "puzzle": make_board(".123456789"),
        "solution": make_board("123456789."),
        "extra": "ignore_me",
    }
    encoded = encode_example(example)
    assert "puzzle" in encoded
    assert "solution" in encoded
    assert "extra" not in encoded
    assert len(encoded["puzzle"]) == SUDOKU_SEQ_LEN
    assert len(encoded["solution"]) == SUDOKU_SEQ_LEN


def test_encode_example_alternative_columns() -> None:
    example = {
        "question": make_board(".123456789"),
        "answer": make_board("123456789."),
    }
    encoded = encode_example(example)
    assert "puzzle" in encoded
    assert "solution" in encoded
    assert len(encoded["puzzle"]) == SUDOKU_SEQ_LEN
    assert len(encoded["solution"]) == SUDOKU_SEQ_LEN


def test_encode_example_missing_puzzle() -> None:
    example = {"solution": make_board("123456789.")}
    with pytest.raises(KeyError, match="Could not find puzzle field"):
        encode_example(example)


def test_encode_example_missing_solution() -> None:
    example = {"puzzle": make_board(".123456789")}
    with pytest.raises(KeyError, match="Could not find solution field"):
        encode_example(example)


def test_sudoku_collate_fn() -> None:
    batch = [
        {"puzzle": [0, 1, 2], "solution": [1, 2, 3]},
        {"puzzle": [1, 2, 3], "solution": [2, 3, 4]},
    ]
    puzzles, solutions = sudoku_collate_fn(batch)
    assert isinstance(puzzles, torch.Tensor)
    assert isinstance(solutions, torch.Tensor)
    assert puzzles.shape == (2, 3)
    assert solutions.shape == (2, 3)


def test_prepare_dataset_map_style(monkeypatch: pytest.MonkeyPatch, sudoku_rows: list[dict[str, str]]) -> None:
    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        assert name == "sapientinc/sudoku-extreme"
        assert split == "train"
        assert not streaming
        return DummyHFDataset(sudoku_rows)

    monkeypatch.setattr("trm.data.load_dataset", fake_load_dataset)
    from trm.config import TrainingConfig
    
    config = TrainingConfig(max_train_samples=2, seed=0)
    dataset = prepare_dataset(config)
    assert len(dataset) == 2
    example = dataset[0]
    assert isinstance(example["puzzle"], list)
    assert isinstance(example["solution"], list)
    assert len(example["puzzle"]) == SUDOKU_SEQ_LEN
    assert len(example["solution"]) == SUDOKU_SEQ_LEN


def test_prepare_dataset_streaming(monkeypatch: pytest.MonkeyPatch, sudoku_rows: list[dict[str, str]]) -> None:
    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        return DummyStreamingDataset(sudoku_rows)

    monkeypatch.setattr("trm.data.load_dataset", fake_load_dataset)
    from trm.config import TrainingConfig
    
    config = TrainingConfig(streaming=True, max_train_samples=2)
    dataset = prepare_dataset(config)
    items = list(itertools.islice(dataset, 3))
    assert len(items) == 3  # Streaming cycles indefinitely


def test_prepare_eval_dataset(monkeypatch: pytest.MonkeyPatch, sudoku_rows: list[dict[str, str]]) -> None:
    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        assert split == "validation"
        if streaming:
            return DummyStreamingDataset(sudoku_rows)
        return DummyHFDataset(sudoku_rows)

    monkeypatch.setattr("trm.data.load_dataset", fake_load_dataset)
    from trm.config import TrainingConfig
    
    config = TrainingConfig(eval_split="validation", max_eval_samples=1)
    dataset = prepare_eval_dataset(config)
    assert len(dataset) == 1


def test_prepare_eval_dataset_streaming(monkeypatch: pytest.MonkeyPatch, sudoku_rows: list[dict[str, str]]) -> None:
    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        return DummyStreamingDataset(sudoku_rows)

    monkeypatch.setattr("trm.data.load_dataset", fake_load_dataset)
    from trm.config import TrainingConfig
    
    config = TrainingConfig(streaming=True, eval_split="validation", max_eval_samples=2)
    dataset = prepare_eval_dataset(config)
    items = list(itertools.islice(dataset, 3))
    assert len(items) == 3  # Streaming cycles indefinitely


def test_prepare_eval_dataset_requires_split() -> None:
    from trm.config import TrainingConfig
    
    config = TrainingConfig(eval_split=None)
    with pytest.raises(ValueError):
        prepare_eval_dataset(config)


def test_build_dataloader_batches_items(monkeypatch: pytest.MonkeyPatch, sudoku_rows: list[dict[str, str]]) -> None:
    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        return DummyHFDataset(sudoku_rows)

    monkeypatch.setattr("trm.data.load_dataset", fake_load_dataset)
    from trm.config import TrainingConfig
    
    config = TrainingConfig(max_train_samples=4, batch_size=2)
    dataset = prepare_dataset(config)
    dataloader = build_dataloader(dataset, config=config)
    
    batches = list(dataloader)
    assert len(batches) == 2
    puzzles, solutions = batches[0]
    assert puzzles.shape[0] == 2
    assert solutions.shape[0] == 2


def test_board_vocab_constants() -> None:
    assert BOARD_VOCAB == ".123456789"
    assert SUDOKU_SEQ_LEN == 81
