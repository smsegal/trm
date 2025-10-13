"""Single-GPU training configuration preset."""

from trm.config import TrainingConfig

DESCRIPTION = "Preset targeting a single modern GPU for fuller fidelity debugging."


def build() -> TrainingConfig:
    """Return a TrainingConfig tuned for single-GPU debugging."""
    return TrainingConfig(
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
        checkpoint_dir="checkpoints/gpu",
        keep_last_n_checkpoints=4,
    )
