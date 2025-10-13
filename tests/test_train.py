import itertools

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from trm.train import (
    BOARD_VOCAB,
    SUDOKU_SEQ_LEN,
    TrainingConfig,
    build_dataloader,
    build_model,
    encode_board,
    encode_example,
    evaluate,
    main,
    parse_args,
    prepare_dataset,
    prepare_eval_dataset,
    resolve_device,
    run_training,
    sudoku_collate_fn,
    training_step,
)


def make_board(pattern: str) -> str:
    repeat = (SUDOKU_SEQ_LEN + len(pattern) - 1) // len(pattern)
    board = (pattern * repeat)[:SUDOKU_SEQ_LEN]
    return board


@pytest.fixture()
def sudoku_rows() -> list[dict[str, str]]:
    return [
        {"puzzle": make_board("."), "solution": make_board("1")},
        {"puzzle": make_board(".2"), "solution": make_board("2")},
        {"puzzle": make_board(".3"), "solution": make_board("3")},
    ]


def test_encode_board_validates_length() -> None:
    board = make_board(".1")
    tokens = encode_board(board)
    assert isinstance(tokens, list)
    assert len(tokens) == SUDOKU_SEQ_LEN
    assert tokens[0] == BOARD_VOCAB.index(".")
    assert tokens[1] == BOARD_VOCAB.index("1")


def test_encode_board_rejects_incorrect_length() -> None:
    with pytest.raises(ValueError):
        encode_board("." * (SUDOKU_SEQ_LEN - 1))


def test_encode_board_rejects_unknown_characters() -> None:
    with pytest.raises(ValueError):
        encode_board("X" * SUDOKU_SEQ_LEN)


class DummyHFDataset:
    def __init__(self, rows: list[dict[str, str | list[int]]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, str | list[int]]:
        return self._rows[index]

    def select(self, indices: range) -> "DummyHFDataset":
        selected = [self._rows[i] for i in indices]
        return DummyHFDataset(selected)

    def shuffle(self, seed: int) -> "DummyHFDataset":
        # Deterministic shuffle for test purposes.
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(self._rows), generator=rng).tolist()
        return DummyHFDataset([self._rows[i] for i in perm])

    def map(self, func, remove_columns=None) -> "DummyHFDataset":
        mapped = [func(row) for row in self._rows]
        return DummyHFDataset(mapped)


def test_encode_example_returns_dict() -> None:
    puzzle = make_board(".")
    solution = make_board("1")
    example = {"puzzle": puzzle, "solution": solution}
    encoded = encode_example(example)
    assert isinstance(encoded, dict)
    assert "puzzle" in encoded
    assert "solution" in encoded
    assert len(encoded["puzzle"]) == SUDOKU_SEQ_LEN
    assert len(encoded["solution"]) == SUDOKU_SEQ_LEN
    assert all(t == 0 for t in encoded["puzzle"])
    assert all(t == BOARD_VOCAB.index("1") for t in encoded["solution"])


def test_encode_example_accepts_question_answer_fields() -> None:
    puzzle = make_board(".")
    solution = make_board("1")
    example = {"question": puzzle, "answer": solution}
    encoded = encode_example(example)
    assert len(encoded["puzzle"]) == SUDOKU_SEQ_LEN
    assert len(encoded["solution"]) == SUDOKU_SEQ_LEN


class DummyStreamingDataset:
    def __init__(self, rows: list[dict[str, str | list[int]]]) -> None:
        self._rows = rows

    def __iter__(self):
        yield from itertools.cycle(self._rows)

    def map(self, func, remove_columns=None) -> "DummyStreamingMappedDataset":
        # For streaming datasets, map is lazy but we'll simulate it here
        return DummyStreamingMappedDataset(self._rows, func)


class DummyStreamingMappedDataset:
    def __init__(self, rows: list[dict[str, str | list[int]]], func) -> None:
        self._rows = rows
        self._func = func

    def __iter__(self):
        for row in itertools.cycle(self._rows):
            yield self._func(row)


def test_sudoku_collate_fn_batches_examples() -> None:
    puzzle = encode_board(make_board("."))
    solution = encode_board(make_board("1"))
    batch = sudoku_collate_fn([{"puzzle": puzzle, "solution": solution}, {"puzzle": puzzle, "solution": solution}])
    puzzles, solutions = batch
    assert puzzles.shape == (2, SUDOKU_SEQ_LEN)
    assert solutions.shape == (2, SUDOKU_SEQ_LEN)


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
    assert model.embedding.num_embeddings == len(BOARD_VOCAB)
    assert model.embedding.embedding_dim == config.hidden_dim


def test_prepare_dataset_map_style(monkeypatch: pytest.MonkeyPatch, sudoku_rows: list[dict[str, str]]) -> None:
    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        assert name == "sapientinc/sudoku-extreme"
        assert split == "train"
        assert not streaming
        return DummyHFDataset(sudoku_rows)

    monkeypatch.setattr("trm.train.load_dataset", fake_load_dataset)
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

    monkeypatch.setattr("trm.train.load_dataset", fake_load_dataset)
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

    monkeypatch.setattr("trm.train.load_dataset", fake_load_dataset)
    config = TrainingConfig(eval_split="validation", max_eval_samples=1)
    dataset = prepare_eval_dataset(config)
    assert len(dataset) == 1


def test_prepare_eval_dataset_streaming(monkeypatch: pytest.MonkeyPatch, sudoku_rows: list[dict[str, str]]) -> None:
    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        return DummyStreamingDataset(sudoku_rows)

    monkeypatch.setattr("trm.train.load_dataset", fake_load_dataset)
    config = TrainingConfig(streaming=True, eval_split="validation", max_eval_samples=2)
    dataset = prepare_eval_dataset(config)
    items = list(itertools.islice(dataset, 3))
    assert len(items) == 3  # Streaming cycles indefinitely


def test_prepare_eval_dataset_requires_split() -> None:
    config = TrainingConfig(eval_split=None)
    with pytest.raises(ValueError):
        prepare_eval_dataset(config)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    puzzles = torch.randint(0, len(BOARD_VOCAB), (2, SUDOKU_SEQ_LEN))
    solutions = torch.randint(0, len(BOARD_VOCAB), (2, SUDOKU_SEQ_LEN))
    loss, halting_logits = training_step(model, optimizer, puzzles, solutions, config=config)
    assert loss >= 0.0
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
    sample = {"puzzle": encode_board(make_board(".")), "solution": encode_board(make_board("1"))}
    dataloader = DataLoader([sample], batch_size=1, collate_fn=sudoku_collate_fn)
    metrics = evaluate(model, dataloader, device=torch.device("cpu"), config=config, max_batches=1)
    assert "loss" in metrics and "cell_accuracy" in metrics


def test_build_dataloader_batches_items() -> None:
    config = TrainingConfig(batch_size=2)
    dataset = [
        {"puzzle": encode_board(make_board(".")), "solution": encode_board(make_board("1"))},
        {"puzzle": encode_board(make_board(".2")), "solution": encode_board(make_board("2"))},
    ]
    dataloader = build_dataloader(dataset, config=config, is_streaming=False)
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

    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        if streaming:
            return FiniteStreamingDataset(sudoku_rows)
        return DummyHFDataset(sudoku_rows)

    monkeypatch.setattr("trm.train.load_dataset", fake_load_dataset)
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

    def fake_load_dataset(name: str, split: str, streaming: bool = False):
        if streaming:
            return FiniteStreamingDataset(sudoku_rows)
        return DummyHFDataset(sudoku_rows)

    monkeypatch.setattr("trm.train.load_dataset", fake_load_dataset)
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
        log_every=10,
        eval_split="train",
        max_eval_samples=2,
        eval_batches=1,
        device="cpu",
    )
    run_training(config)


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
