"""A modern PyTorch project template."""

from .config import TrainingConfig, gpu_workstation_config, macbook_quickstart_config
from .model import TRM
from .training import encode_board, encode_example, run_training

__version__ = "0.1.0"

__all__ = [
    "TRM",
    "TrainingConfig",
    "gpu_workstation_config",
    "__version__",
    "encode_board",
    "encode_example",
    "macbook_quickstart_config",
    "run_training",
]
