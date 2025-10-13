import itertools

import pytest
import torch
from torch.utils.data import IterableDataset

from trm.config import TrainingConfig
from trm.train import main, parse_args
from trm.training import (
    build_model,
    evaluate,
    resolve_device,
    run_training,
    training_step,
)


def make_board(pattern: str) -> str:
    repeat = (81 + len(pattern) - 1) // len(pattern)
    board = (pattern * repeat)[:81]
    return board


def encode_board(board: str) -> list[int]:
    """Simple encode_board function for testing."""
    from trm.data import encode_board as real_encode_board
    return real_encode_board(board)


@pytest.fixture()
def sudoku_rows() -> list[dict[str, str]]:
    return [
        {"puzzle": make_board("."), "solution": make_board("1")},
        {"puzzle": make_board(".2"), "solution": make_board("2")},
        {"puzzle": make_board(".3"), "solution": make_board("3")},
    ]


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

    def batch(self, batch_size: int, drop_last_batch: bool = False) -> "DummyBatchDataset":
        return DummyBatchDataset(self._rows, batch_size, drop_last_batch)


class DummyBatchDataset:
    def __init__(self, rows: list[dict[str, str | list[int]]], batch_size: int, drop_last_batch: bool = False) -> None:
        self._rows = rows
        self._batch_size = batch_size
        self._drop_last_batch = drop_last_batch

    def __iter__(self):
        batch = []
        for row in self._rows:
            batch.append(row)
            if len(batch) >= self._batch_size:
                yield self._collate(batch)
                batch = []
        if batch and not self._drop_last_batch:
            yield self._collate(batch)

    @staticmethod
    def _collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        puzzles = torch.tensor([item["puzzle"] for item in batch], dtype=torch.long)
        solutions = torch.tensor([item["solution"] for item in batch], dtype=torch.long)
        return {"puzzle": puzzles, "solution": solutions}


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


def test_resolve_device_auto_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device = resolve_device("auto")
    assert device.type == "cpu"


def test_resolve_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device = resolve_device("auto")
    assert device.type == "cuda"


def test_resolve_device_uses_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device = resolve_device("auto")
    assert device.type == "mps"


def test_build_model_uses_vocab_size() -> None:
    config = TrainingConfig(hidden_dim=32, n_layers=2, num_heads=4, dropout=0.0)
    model = build_model(config)
    from trm.data import BOARD_VOCAB
    assert model.embedding.num_embeddings == len(BOARD_VOCAB)
    assert model.embedding.embedding_dim == config.hidden_dim


def test_training_step_runs_backward() -> None:
    config = TrainingConfig(
        hidden_dim=32,
        n_layers=2,
        num_heads=4,
        latent_steps=1,
        recurse=1,
        max_grad_norm=1.0,
    )
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create dummy data
    puzzles = torch.zeros(2, 81, dtype=torch.long)
    solutions = torch.ones(2, 81, dtype=torch.long)
    
    loss, halting_logits = training_step(
        model,
        optimizer,
        puzzles,
        solutions,
        config=config,
    )
    assert isinstance(loss, float)
    assert isinstance(halting_logits, torch.Tensor)


def test_evaluate_returns_metrics() -> None:
    config = TrainingConfig(
        hidden_dim=32,
        n_layers=2,
        num_heads=4,
        latent_steps=1,
        recurse=1,
    )
    model = build_model(config)
    
    # Create dummy dataset with tensors
    puzzles = torch.zeros(2, 81, dtype=torch.long)
    solutions = torch.ones(2, 81, dtype=torch.long)
    
    def mock_dataloader():
        yield puzzles, solutions
        yield puzzles, solutions
    
    metrics = evaluate(
        model,
        mock_dataloader(),
        device=torch.device("cpu"),
        config=config,
        max_batches=1,
    )
    assert "loss" in metrics and "cell_accuracy" in metrics


def test_build_dataloader_batches_items() -> None:
    from trm.data import build_dataloader
    
    config = TrainingConfig(batch_size=2)
    dataset = DummyHFDataset([
        {"puzzle": encode_board(make_board(".")), "solution": encode_board(make_board("1"))},
        {"puzzle": encode_board(make_board(".")), "solution": encode_board(make_board("1"))},
    ])
    dataloader = build_dataloader(dataset, config=config)
    puzzles, _ = next(iter(dataloader))
    assert puzzles.shape[0] == 2


def test_run_training_executes(monkeypatch: pytest.MonkeyPatch, sudoku_rows: list[dict[str, str]]) -> None:
    class FiniteStreamingDataset:
        def __init__(self, rows: list[dict[str, str | list[int]]]) -> None:
            self._rows = rows

        def __iter__(self):
            yield from self._rows

        def map(self, func, remove_columns=None) -> "FiniteStreamingMappedDataset":
            return FiniteStreamingMappedDataset(self._rows, func)

    class FiniteStreamingMappedDataset(IterableDataset):
        def __init__(self, rows: list[dict[str, str | list[int]]], func) -> None:
            super().__init__()
            self._rows = rows
            self._func = func

        def __iter__(self):
            for row in self._rows:
                yield self._func(row)

        def batch(self, batch_size: int, drop_last_batch: bool = False) -> "DummyStreamingBatchedDataset":
            return DummyStreamingBatchedDataset(self, batch_size, drop_last_batch)

    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        if streaming:
            return FiniteStreamingDataset(sudoku_rows)
        return DummyHFDataset(sudoku_rows)

    monkeypatch.setattr("trm.data.load_dataset", fake_load_dataset)
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        hidden_dim=32,
        n_layers=2,
        num_heads=4,
        latent_steps=1,
        recurse=1,
        max_train_samples=2,
        max_steps_per_epoch=1,
        log_every=1,
        eval_split="train",
        max_eval_samples=2,
        eval_batches=1,
        device="cpu",
    )
    run_training(config)


def test_run_training_streaming(monkeypatch: pytest.MonkeyPatch, sudoku_rows: list[dict[str, str]]) -> None:
    class FiniteStreamingDataset:
        def __init__(self, rows: list[dict[str, str | list[int]]]) -> None:
            self._rows = rows

        def __iter__(self):
            yield from self._rows

        def map(self, func, remove_columns=None) -> "FiniteStreamingMappedDataset":
            return FiniteStreamingMappedDataset(self._rows, func)

    class FiniteStreamingMappedDataset(IterableDataset):
        def __init__(self, rows: list[dict[str, str | list[int]]], func) -> None:
            super().__init__()
            self._rows = rows
            self._func = func

        def __iter__(self):
            for row in self._rows:
                yield self._func(row)

        def batch(self, batch_size: int, drop_last_batch: bool = False):
            class FiniteStreamingBatchedDataset:
                def __init__(self, dataset, batch_size: int, drop_last_batch: bool = False) -> None:
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
            
            return FiniteStreamingBatchedDataset(self, batch_size, drop_last_batch)

    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        if streaming:
            return FiniteStreamingDataset(sudoku_rows)
        return DummyHFDataset(sudoku_rows)

    monkeypatch.setattr("trm.data.load_dataset", fake_load_dataset)
    config = TrainingConfig(
        streaming=True,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        hidden_dim=32,
        n_layers=2,
        num_heads=4,
        latent_steps=1,
        recurse=1,
        max_train_samples=2,
        max_steps_per_epoch=1,
        log_every=1,
        eval_split="train",
        max_eval_samples=2,
        eval_batches=1,
        device="cpu",
    )
    run_training(config)


def test_run_training_saves_checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path, sudoku_rows: list[dict[str, str]]) -> None:
    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        return DummyHFDataset(sudoku_rows)

    monkeypatch.setattr("trm.data.load_dataset", fake_load_dataset)
    checkpoint_dir = tmp_path / "checkpoints"
    config = TrainingConfig(
        epochs=3,
        batch_size=2,
        learning_rate=1e-3,
        hidden_dim=32,
        n_layers=2,
        num_heads=4,
        latent_steps=1,
        recurse=1,
        max_train_samples=2,
        max_steps_per_epoch=1,
        log_every=1,
        eval_split="train",
        max_eval_samples=2,
        eval_batches=1,
        device="cpu",
        checkpoint_dir=str(checkpoint_dir),
        keep_last_n_checkpoints=2,
        eval_preview_examples=0,
    )
    run_training(config)

    epoch_checkpoints = sorted(checkpoint_dir.glob("epoch-*.pt"))
    assert len(epoch_checkpoints) == 2
    assert epoch_checkpoints[0].name.endswith("0002.pt")
    assert epoch_checkpoints[1].name.endswith("0003.pt")
    assert (checkpoint_dir / "best.pt").exists()


def test_parse_args_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, TrainingConfig] = {}

    def fake_run_training(config: TrainingConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr("trm.train.run_training", fake_run_training)
    config = parse_args(["--epochs", "2", "--batch-size", "8", "--streaming"])
    assert config.epochs == 2
    assert config.batch_size == 8
    assert config.streaming
    
    main(["--epochs", "3"])
    assert captured["config"].epochs == 3


def test_parse_args_supports_presets() -> None:
    config = parse_args(["--config", "macbook"])
    assert config.batch_size == 16
    assert config.hidden_dim == 128
    assert config.device == "auto"

    config = parse_args(["--config", "gpu", "--batch-size", "64"])
    assert config.device == "cuda"
    assert config.batch_size == 64
