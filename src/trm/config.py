"""Configuration helpers for Tiny Reasoning Model training."""

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Any


@dataclass
class DataConfig:
    """Dataset configuration."""

    dataset_name: str = field(default="sapientinc/sudoku-extreme", metadata={"help": "Name of the dataset to use"})
    train_split: str = field(default="train", metadata={"help": "Training split name"})
    eval_split: str = field(default="test", metadata={"help": "Evaluation split name"})
    columns: dict[str, str] = field(
        default_factory=lambda: {"x": "question", "y": "answer"}, metadata={"help": "Column mapping for dataset"}
    )
    streaming: bool = field(default=False, metadata={"help": "Use streaming for dataset loading"})


@dataclass
class ModelConfig:
    """Model configuration."""

    hidden_dim: int = field(default=128, metadata={"help": "Hidden dimension size"})
    n_layers: int = field(default=2, metadata={"help": "Number of transformer layers"})
    num_heads: int = field(default=4, metadata={"help": "Number of attention heads"})
    dropout: float = field(default=0.0, metadata={"help": "Dropout probability"})
    latent_steps: int = field(default=4, metadata={"help": "Number of latent steps"})
    recurse: int = field(default=2, metadata={"help": "Recursion depth"})
    max_position_embeddings: int = field(default=2048, metadata={"help": "Maximum position embeddings"})
    rope_base: float = field(default=10000.0, metadata={"help": "RoPE base value"})


DefaultModelConfig = ModelConfig(
    hidden_dim=128,
    n_layers=2,
    num_heads=4,
    dropout=0.0,
    latent_steps=4,
    recurse=2,
    max_position_embeddings=2048,
    rope_base=10000.0,
)


SudokuDatasetConfig = DataConfig(
    dataset_name="sapientinc/sudoku-extreme",
    train_split="train",
    eval_split="test",
    columns={"x": "question", "y": "answer"},
)


@dataclass
class TrainingConfig:
    """Configuration for TRM training."""

    # Config preset to use
    config: str = field(default="macbook", metadata={"help": "Choose a preset configuration (macbook, gpu)"})
    data: DataConfig = field(default_factory=lambda: SudokuDatasetConfig, metadata={"help": "Dataset configuration"})
    model: ModelConfig = field(default_factory=lambda: DefaultModelConfig, metadata={"help": "Model configuration"})
    batch_size: int = field(default=16, metadata={"help": "Batch size for training"})
    num_workers: int = field(default=0, metadata={"help": "Number of workers for data loading"})
    epochs: int = field(default=4, metadata={"help": "Number of training epochs"})
    learning_rate: float = field(default=5e-4, metadata={"help": "Learning rate for the optimizer"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for regularization"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm for clipping"})
    log_every: int = field(default=10, metadata={"help": "Log every N steps"})
    max_train_samples: int | None = field(default=512, metadata={"help": "Maximum number of training samples"})
    max_steps_per_epoch: int | None = field(default=1000, metadata={"help": "Maximum steps per epoch"})
    max_eval_samples: int | None = field(default=256, metadata={"help": "Maximum number of evaluation samples"})
    eval_batches: int | None = field(default=4, metadata={"help": "Number of evaluation batches"})
    streaming: bool = field(default=False, metadata={"help": "Use streaming for dataset loading"})
    seed: int = field(default=19, metadata={"help": "Random seed for reproducibility"})
    device: str = field(default="auto", metadata={"help": "Device to run training on (auto, cpu, cuda)"})
    checkpoint_dir: str | None = field(
        default="checkpoints/macbook", metadata={"help": "Directory to save checkpoints"}
    )
    keep_last_n_checkpoints: int = field(default=2, metadata={"help": "Keep last N checkpoints"})
    checkpoint_metric: str = field(default="accuracy", metadata={"help": "Metric to use for checkpointing"})
    checkpoint_metric_higher_is_better: bool = field(
        default=True, metadata={"help": "Whether higher metric values are better"}
    )
    eval_preview_examples: int = field(default=3, metadata={"help": "Number of examples to preview during evaluation"})
    wandb_project: str = field(default="trm-training", metadata={"help": "Wandb project name"})
    wandb_name: str | None = field(default=None, metadata={"help": "Wandb run name (auto-generated if None)"})
    wandb_log_model: bool = field(default=False, metadata={"help": "Log model artifacts to wandb"})


DefaultTrainingConfig = TrainingConfig(
    data=SudokuDatasetConfig,
    model=DefaultModelConfig,
    batch_size=16,
    num_workers=0,
    epochs=4,
    learning_rate=5e-4,
    weight_decay=0.0,
    max_grad_norm=1.0,
    log_every=10,
    max_train_samples=512,
    max_steps_per_epoch=1000,
    max_eval_samples=256,
    eval_batches=4,
    streaming=False,
    seed=19,
    device="auto",
    checkpoint_dir="checkpoints/macbook",
    keep_last_n_checkpoints=2,
    checkpoint_metric="accuracy",
    checkpoint_metric_higher_is_better=True,
    eval_preview_examples=3,
    wandb_project="trm-training",
    wandb_name=None,
    wandb_log_model=False,
)


def apply_overrides(base: TrainingConfig, overrides: dict[str, Any]) -> TrainingConfig:
    if not overrides:
        return base

    training_field_names = {field.name for field in fields(TrainingConfig) if field.name not in {"data", "model"}}
    model_field_names = {field.name for field in fields(ModelConfig)}
    data_field_names = {field.name for field in fields(DataConfig)}

    remaining = dict(overrides)

    if "model" in remaining:
        base = replace(base, model=remaining.pop("model"))

    model_overrides = {key: remaining.pop(key) for key in list(remaining) if key in model_field_names}
    if model_overrides:
        model = replace(base.model, **model_overrides)
        base = replace(base, model=model)

    if "data" in remaining:
        base = replace(base, data=remaining.pop("data"))

    data_overrides = {key: remaining.pop(key) for key in list(remaining) if key in data_field_names}
    if data_overrides:
        data = replace(base.data, **data_overrides)
        base = replace(base, data=data)

    training_overrides = {key: remaining.pop(key) for key in list(remaining) if key in training_field_names}
    if training_overrides:
        base = replace(base, **training_overrides)

    if remaining:
        unknown = ", ".join(sorted(remaining))
        raise ValueError(f"Unknown TrainingConfig override(s): {unknown}.")
    return base


def macbook_quickstart_config(**overrides: Any) -> TrainingConfig:
    """Preset tuned for quick CPU/MPS experimentation on a MacBook."""
    base = TrainingConfig(
        data=SudokuDatasetConfig,
        model=DefaultModelConfig,
        batch_size=16,
        num_workers=10,
        epochs=4,
        learning_rate=5e-4,
        weight_decay=0.0,
        max_grad_norm=1.0,
        log_every=10,
        max_train_samples=512,
        max_steps_per_epoch=1000,
        max_eval_samples=256,
        eval_batches=4,
        streaming=False,
        seed=19,
        device="auto",
        checkpoint_dir="checkpoints/macbook",
        keep_last_n_checkpoints=2,
        checkpoint_metric="accuracy",
        checkpoint_metric_higher_is_better=True,
        eval_preview_examples=3,
    )
    return apply_overrides(base, overrides)


def gpu_workstation_config(**overrides: Any) -> TrainingConfig:
    """Preset targeting a single modern GPU for fuller fidelity debugging."""
    base = TrainingConfig(
        data=SudokuDatasetConfig,
        model=ModelConfig(
            hidden_dim=256,
            n_layers=6,
            num_heads=8,
            dropout=0.1,
            latent_steps=4,
            recurse=6,
            max_position_embeddings=2048,
            rope_base=10000.0,
        ),
        batch_size=128,
        num_workers=4,
        epochs=3,
        learning_rate=1e-3,
        weight_decay=0.01,
        max_grad_norm=1.0,
        log_every=25,
        max_train_samples=4096,
        max_steps_per_epoch=None,
        max_eval_samples=2048,
        eval_batches=16,
        streaming=False,
        seed=42,
        device="cuda",
        checkpoint_dir="checkpoints/gpu",
        keep_last_n_checkpoints=4,
        checkpoint_metric="accuracy",
        checkpoint_metric_higher_is_better=True,
        eval_preview_examples=3,
    )
    return apply_overrides(base, overrides)


CONFIG_PRESETS: dict[str, Callable[[], TrainingConfig]] = {
    "macbook": macbook_quickstart_config,
    "gpu": gpu_workstation_config,
}
