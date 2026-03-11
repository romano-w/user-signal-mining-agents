from __future__ import annotations

from pathlib import Path

from user_signal_mining_agents import config
from user_signal_mining_agents.config import Settings


def test_ensure_scaffold_directories_creates_paths(tmp_path: Path) -> None:
    settings = Settings(
        prompts_dir=tmp_path / "prompts",
        index_dir=tmp_path / "artifacts" / "index",
        run_artifacts_dir=tmp_path / "artifacts" / "runs",
        yelp_dataset_dir=tmp_path / "data" / "raw" / "Yelp-JSON",
        founder_prompts_path=tmp_path / "founder_prompts" / "restaurants.json",
    )

    dirs = config.ensure_scaffold_directories(settings)

    assert dirs
    for directory in dirs:
        assert directory.exists()


def test_get_settings_is_cached() -> None:
    first = config.get_settings()
    second = config.get_settings()
    assert first is second
