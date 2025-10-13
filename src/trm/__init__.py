"""A modern PyTorch project template."""

from .config import TrainingConfig, gpu_workstation_config, macbook_quickstart_config
from .data import decode_board, encode_board, encode_example, format_board
from .inspection import preview_samples
from .model import TRM
from .training import run_training

__version__ = "0.1.0"

__all__ = [
    "TRM",
    "TrainingConfig",
    "gpu_workstation_config",
    "__version__",
    "decode_board",
    "encode_board",
    "encode_example",
    "format_board",
    "macbook_quickstart_config",
    "preview_samples",
    "run_training",
]
