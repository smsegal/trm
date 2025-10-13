"""Configuration helpers for Tiny Reasoning Model training."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

DEFAULT_DATASET = "sapientinc/sudoku-extreme"
DEFAULT_SPLIT = "train"


@dataclass
class TrainingConfig:
    dataset_name: str = DEFAULT_DATASET
    split: str = DEFAULT_SPLIT
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hidden_dim: int = 256
    n_layers: int = 4
    num_heads: int | None = None
    dropout: float = 0.1
    latent_steps: int = 4
    recurse: int = 6
    log_every: int = 50
    max_train_samples: int | None = None
    max_steps_per_epoch: int | None = None
    eval_split: str | None = "test"
    max_eval_samples: int | None = 4096
    eval_batches: int | None = None
    streaming: bool = False
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


def macbook_quickstart_config(**overrides: Any) -> TrainingConfig:
    """Preset tuned for quick CPU/MPS experimentation on a MacBook."""
    base = TrainingConfig(
        batch_size=16,
        epochs=4,
        learning_rate=5e-4,
        weight_decay=0.0,
        hidden_dim=128,
        n_layers=2,
        num_heads=4,
        dropout=0.0,
        latent_steps=4,
        recurse=2,
        log_every=10,
        max_train_samples=512,
        max_steps_per_epoch=1000,
        max_eval_samples=256,
        eval_batches=4,
        num_workers=0,
        device="auto",
    )
    if overrides:
        return replace(base, **overrides)
    return base


def gpu_workstation_config(**overrides: Any) -> TrainingConfig:
    """Preset targeting a single modern GPU for fuller fidelity debugging."""
    base = TrainingConfig(
        batch_size=128,
        epochs=3,
        learning_rate=1e-3,
        weight_decay=0.01,
        hidden_dim=256,
        n_layers=6,
        num_heads=None,
        dropout=0.1,
        latent_steps=4,
        recurse=6,
        log_every=25,
        max_train_samples=4096,
        max_steps_per_epoch=None,
        max_eval_samples=2048,
        eval_batches=16,
        num_workers=4,
        device="cuda",
    )
    if overrides:
        return replace(base, **overrides)
    return base


CONFIG_PRESETS: dict[str, Callable[[], TrainingConfig]] = {
    "default": TrainingConfig,
    "macbook": macbook_quickstart_config,
    "gpu": gpu_workstation_config,
}
