from dataclasses import dataclass

import torch

from trm.config import TrainingConfig
from trm.data import BOARD_VOCAB, SUDOKU_SEQ_LEN, encode_board
from trm.inspection import SudokuSample
from trm.viewer import _build_config, _predict_solution


@dataclass
class Args:
    config: str | None = None
    dataset_name: str | None = None
    split: str | None = None
    seed: int | None = None
    hidden_dim: int | None = None
    n_layers: int | None = None
    num_heads: int | None = None
    latent_steps: int | None = None
    recurse: int | None = None
    dropout: float | None = None
    device: str | None = None


class DummyModel(torch.nn.Module):
    def __init__(self, target_tokens):
        super().__init__()
        vocab = len(BOARD_VOCAB)
        self.embedding = torch.nn.Embedding(vocab, 4)
        self._target_tokens = torch.tensor(target_tokens)

    def deep_recursion(self, x_hidden, zeros_y, zeros_z, *, n, recurse):
        batch, seq_len, _ = x_hidden.shape
        vocab = len(BOARD_VOCAB)
        logits = torch.full((batch, seq_len, vocab), fill_value=-1e3)
        for idx, token in enumerate(self._target_tokens.tolist()):
            logits[:, idx, token] = 1e3
        halting = torch.zeros(batch, seq_len, 1)
        return zeros_y, zeros_z, logits, halting


def test_build_config_overrides_defaults():
    args = Args(dataset_name="custom", split="validation", hidden_dim=128, n_layers=2, device="cpu")
    config = _build_config(args)
    assert isinstance(config, TrainingConfig)
    assert config.dataset_name == "custom"
    assert config.split == "validation"
    assert config.hidden_dim == 128
    assert config.n_layers == 2
    assert config.device == "cpu"


def test_predict_solution_roundtrip():
    board = ".12345678" * 9
    tokens = encode_board(board)
    sample = SudokuSample(
        source="synthetic",
        rating="test",
        question=board,
        answer=board,
        encoded_puzzle=tokens,
        encoded_solution=tokens,
    )
    model = DummyModel(tokens)
    config = TrainingConfig(latent_steps=1, recurse=1)
    device = torch.device("cpu")
    predicted, accuracy, mismatches = _predict_solution(model, sample, device=device, config=config)
    assert predicted == board
    assert accuracy == 1.0
    assert mismatches == set()
