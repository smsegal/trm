from trm import gpu_workstation_config as package_gpu_config
from trm import macbook_quickstart_config as package_mac_config
from trm.config import gpu_workstation_config, macbook_quickstart_config


def test_macbook_quickstart_config_defaults() -> None:
    config = macbook_quickstart_config()
    assert config.batch_size == 16
    assert config.hidden_dim == 128
    assert config.n_layers == 2
    assert config.max_train_samples == 512
    assert config.eval_batches == 4
    assert config.device == "auto"


def test_macbook_quickstart_config_overrides() -> None:
    config = macbook_quickstart_config(batch_size=8, device="cpu")
    assert config.batch_size == 8
    assert config.device == "cpu"
    assert config.hidden_dim == 128


def test_gpu_workstation_config_defaults() -> None:
    config = gpu_workstation_config()
    assert config.device == "cuda"
    assert config.batch_size == 128
    assert config.epochs == 3
    assert config.log_every == 25


def test_configs_reexported() -> None:
    assert package_mac_config is macbook_quickstart_config
    assert package_gpu_config is gpu_workstation_config
