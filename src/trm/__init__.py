"""A modern PyTorch project template."""

from .config import ModelConfig, TrainingConfig, gpu_workstation_config, macbook_quickstart_config
from .data import decode_board, format_board
from .model import TRM
from .training import run_training

__version__ = "0.1.0"

__all__ = [
    "TRM",
    "ModelConfig",
    "TrainingConfig",
    "__version__",
    "decode_board",
    "format_board",
    "gpu_workstation_config",
    "macbook_quickstart_config",
    "run_training",
]
