import pytest

from trm.inspection import SudokuSample, pull_sudoku_samples


class DummyDataset:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]

    def __iter__(self):
        return iter(self._rows)

    def shuffle(self, seed=None):
        return self

    def select(self, indices):
        selected_rows = [self._rows[i] for i in indices]
        return DummyDataset(selected_rows)


@pytest.fixture
def sudoku_rows():
    board = "123456789" * 9
    solved = "987654321" * 9
    return [
        {
            "source": "synthetic",
            "rating": "extreme",
            "question": board,
            "answer": solved,
        }
    ]


def test_pull_sudoku_samples(monkeypatch, sudoku_rows):
    monkeypatch.setattr("trm.inspection.load_dataset", lambda *args, **kwargs: DummyDataset(sudoku_rows))
    samples = pull_sudoku_samples(dataset_name="dummy", split="train", count=1, seed=0)
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, SudokuSample)
    assert sample.source == "synthetic"
    assert sample.rating == "extreme"
    assert sample.question == sudoku_rows[0]["question"]
    assert sample.encoded_puzzle[0] == 1


def test_pull_sudoku_samples_zero_count(monkeypatch):
    def fail_load(*args, **kwargs):
        raise AssertionError("load_dataset should not be called when count is zero.")

    monkeypatch.setattr("trm.inspection.load_dataset", fail_load)
    samples = pull_sudoku_samples(dataset_name="dummy", split="train", count=0, seed=0)
    assert samples == []


def test_pull_sudoku_samples_roundtrip_validation(monkeypatch, sudoku_rows):
    bad_rows = [dict(sudoku_rows[0], answer="." * 80 + "X")]
    monkeypatch.setattr("trm.inspection.load_dataset", lambda *args, **kwargs: DummyDataset(bad_rows))
    with pytest.raises(ValueError, match="round-trip"):
        pull_sudoku_samples(dataset_name="dummy", split="train", count=1, seed=0)
