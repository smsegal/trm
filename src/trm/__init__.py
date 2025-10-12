"""A modern PyTorch project template."""

from .model import TRM
from .train import (
    TrainingConfig,
    encode_board,
    encode_example,
    run_training,
)

__version__ = "0.1.0"

__all__ = [
    "TRM",
    "TrainingConfig",
    "__version__",
    "encode_board",
    "encode_example",
    "run_training",
]
