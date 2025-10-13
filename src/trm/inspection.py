"""Utilities for sampling and inspecting sudoku data."""

import argparse
from dataclasses import dataclass
from typing import Any, Sequence

from datasets import Dataset, load_dataset

from .config import TrainingConfig, apply_overrides
from .data import (
    ENCODED_PUZZLE_COLUMN,
    ENCODED_SOLUTION_COLUMN,
    PUZZLE_TEXT_COLUMN,
    SOLUTION_TEXT_COLUMN,
    decode_board,
    encode_board,
    format_board,
)


@dataclass(slots=True)
class SudokuSample:
    """Container for a single sudoku example."""

    source: str
    rating: str
    question: str
    answer: str
    encoded_puzzle: list[int]
    encoded_solution: list[int]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SudokuSample":
        question = _require_string(row, PUZZLE_TEXT_COLUMN)
        answer = _require_string(row, SOLUTION_TEXT_COLUMN)
        source = str(row.get("source", "unknown"))
        rating_value = row.get("rating", "unknown")
        rating = str(rating_value)
        try:
            encoded_puzzle = encode_board(question)
        except ValueError as exc:
            msg = f"Failed to encode {PUZZLE_TEXT_COLUMN!r} for round-trip validation: {exc}"
            raise ValueError(msg) from exc
        try:
            encoded_solution = encode_board(answer)
        except ValueError as exc:
            msg = f"Failed to encode {SOLUTION_TEXT_COLUMN!r} for round-trip validation: {exc}"
            raise ValueError(msg) from exc
        return cls(
            source=source,
            rating=rating,
            question=question,
            answer=answer,
            encoded_puzzle=encoded_puzzle,
            encoded_solution=encoded_solution,
        )

    def as_dict(self) -> dict[str, Any]:
        """Represent the sample as a JSON-serialisable dictionary."""
        return {
            "source": self.source,
            "rating": self.rating,
            PUZZLE_TEXT_COLUMN: self.question,
            SOLUTION_TEXT_COLUMN: self.answer,
            ENCODED_PUZZLE_COLUMN: self.encoded_puzzle,
            ENCODED_SOLUTION_COLUMN: self.encoded_solution,
        }

    def pretty(self) -> str:
        """Format the sample for printing."""
        return (
            f"source={self.source} rating={self.rating}\n"
            f"Puzzle:\n{format_board(self.question)}\n"
            f"Solution:\n{format_board(self.answer)}"
        )

    def verify_roundtrip(self) -> None:
        """Ensure encode/decode roundtrips cleanly."""
        decoded_puzzle = decode_board(self.encoded_puzzle)
        decoded_solution = decode_board(self.encoded_solution)
        if decoded_puzzle != self.question:
            raise ValueError("Encoded puzzle does not round-trip to original question.")
        if decoded_solution != self.answer:
            raise ValueError("Encoded solution does not round-trip to original answer.")


def _require_string(row: dict[str, Any], column: str) -> str:
    if column not in row:
        available = ", ".join(sorted(row))
        raise KeyError(f"Column {column!r} missing from dataset row. Available keys: {available}.")
    value = row[column]
    if not isinstance(value, str):
        raise TypeError(f"Column {column!r} must be a string, received {type(value).__name__}.")
    return value


def pull_sudoku_samples(
    *,
    dataset_name: str,
    split: str,
    count: int,
    seed: int | None = None,
) -> list[SudokuSample]:
    """Load a small set of sudoku samples for inspection."""
    if count <= 0:
        return []

    dataset: Dataset = load_dataset(dataset_name, split=split)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    upper = min(len(dataset), count)
    selected = dataset.select(range(upper))
    samples = [SudokuSample.from_row(row) for row in selected]
    for sample in samples:
        sample.verify_roundtrip()
    return samples


def preview_samples(config: TrainingConfig, *, count: int) -> list[SudokuSample]:
    """Convenience wrapper that respects the project's TrainingConfig."""
    return pull_sudoku_samples(
        dataset_name=config.dataset_name,
        split=config.split,
        count=count,
        seed=config.seed,
    )


def _build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    overrides: dict[str, object] = {}
    if args.dataset_name is not None:
        overrides["dataset_name"] = args.dataset_name
    if args.split is not None:
        overrides["train_split"] = args.split
    if args.seed is not None:
        overrides["seed"] = args.seed
    return apply_overrides(TrainingConfig(), overrides)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preview sudoku-extreme dataset samples.")
    parser.add_argument("--dataset-name", type=str, default=TrainingConfig().dataset_name)
    parser.add_argument("--split", type=str, default=TrainingConfig().split)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=TrainingConfig().seed)
    args = parser.parse_args(argv)

    config = _build_config_from_args(args)
    samples = preview_samples(config, count=args.count)

    if not samples:
        print("No samples requested.")
        return
    for index, sample in enumerate(samples, start=1):
        header = f"=== Sample {index} ==="
        print(header)
        print(sample.pretty())
        print()


if __name__ == "__main__":  # pragma: no cover
    main()
