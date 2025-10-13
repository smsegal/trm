# Continuation Checklist

- Stay inside this `jj` workspace (`config-cli-alignment`) for all edits. The default workspace already has user changes—do not touch it.
- Confirm the preset refactor: keep `TrainingConfig` in `src/trm/config.py`, move preset builders to `src/trm/configs/`, and expose them via `ConfigPreset` metadata.
- Replace the argparse CLI in `src/trm/train.py` with Tyro. Use `tyro.extras.overridable_config_cli(iter_cli_config_options())` so both preset selection and nested overrides work. Keep a thin shim for legacy `--config` usage that forwards to the Tyro subcommand form.
- Update other CLI entry points (e.g., `src/trm/viewer.py`) to call `CONFIG_PRESETS[name].build()` instead of the old callable map.
- Ensure tests reflect the new flow: add coverage in `tests/test_train.py` for Tyro parsing (subcommand and override cases) and adjust fixtures if signatures change.
- Add Tyro to `pyproject.toml` dependencies and confirm no other tooling needs tweaks.
- Once refactoring stabilizes, run `uv run pytest` and `uv run basedpyright`; capture command outputs for the eventual PR summary.
- Follow repo style rules (type hints, jaxtyping compatibility, concise comments) and keep instructions aligned with the Tiny Reasoning Model architecture references in `src/trm/model.py`.
