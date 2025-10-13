"""MacBook-friendly training configuration preset."""

from trm.config import TrainingConfig

DESCRIPTION = "Preset tuned for quick CPU/MPS experimentation on a MacBook."


def build() -> TrainingConfig:
    """Return a TrainingConfig tuned for MacBook experimentation."""
    return TrainingConfig(
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
        checkpoint_dir="checkpoints/macbook",
        keep_last_n_checkpoints=2,
    )
