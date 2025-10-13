"""Configuration helpers for Tiny Reasoning Model training."""

from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping


@dataclass
class DataConfig:
    dataset_name: str
    train_split: str
    eval_split: str
    columns: dict[str, str]
    streaming: bool = field(default=False)


@dataclass
class ModelConfig:
    hidden_dim: int
    n_layers: int
    num_heads: int
    dropout: float
    latent_steps: int
    recurse: int
    max_position_embeddings: int
    rope_base: float


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
    data: DataConfig
    model: ModelConfig
    batch_size: int
    num_workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    log_every: int
    max_train_samples: int | None
    max_steps_per_epoch: int | None
    max_eval_samples: int | None
    eval_batches: int | None
    streaming: bool
    seed: int
    device: str
    checkpoint_dir: str | None
    keep_last_n_checkpoints: int
    checkpoint_metric: str
    checkpoint_metric_higher_is_better: bool
    eval_preview_examples: int


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


@dataclass(frozen=True)
class ConfigPreset:
    """Metadata describing a reusable training configuration preset."""

    description: str
    builder: Callable[[], TrainingConfig]

    def build(self) -> TrainingConfig:
        """Construct a fresh TrainingConfig instance for this preset."""
        return self.builder()


def _apply_overrides(base: TrainingConfig, overrides: Mapping[str, Any] | None) -> TrainingConfig:
    """Return a new TrainingConfig with keyword overrides applied."""
    if overrides:
        return replace(base, **overrides)
    return base


def macbook_quickstart_config(**overrides: Any) -> TrainingConfig:
    """Preset tuned for quick CPU/MPS experimentation on a MacBook."""
    from trm.configs import macbook as macbook_preset

    base = macbook_preset.build()
    return _apply_overrides(base, overrides)


def gpu_workstation_config(**overrides: Any) -> TrainingConfig:
    """Preset targeting a single modern GPU for fuller fidelity debugging."""
    from trm.configs import gpu as gpu_preset

    base = gpu_preset.build()
    return _apply_overrides(base, overrides)


from trm.configs import gpu as _gpu_preset  # noqa: E402
from trm.configs import macbook as _macbook_preset  # noqa: E402

CONFIG_PRESETS: dict[str, ConfigPreset] = {
    "default": ConfigPreset(
        description="Baseline configuration targeting general-purpose training.",
        builder=TrainingConfig,
    ),
    "macbook": ConfigPreset(
        description=_macbook_preset.DESCRIPTION,
        builder=_macbook_preset.build,
    ),
    "gpu": ConfigPreset(
        description=_gpu_preset.DESCRIPTION,
        builder=_gpu_preset.build,
    ),
}


def get_config_preset(name: str) -> TrainingConfig:
    """Return a TrainingConfig instantiated from the named preset."""
    preset = CONFIG_PRESETS[name]
    return preset.build()


def iter_cli_config_options() -> dict[str, tuple[str, TrainingConfig]]:
    """Return CLI-ready mapping of preset names to (description, config instance)."""
    return {name: (preset.description, preset.build()) for name, preset in CONFIG_PRESETS.items()}
