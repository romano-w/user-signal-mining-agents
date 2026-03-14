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


def test_settings_include_retrieval_stack_v2_defaults() -> None:
    settings = Settings()

    assert settings.retrieval_mode == "hybrid"
    assert settings.retrieval_dense_weight == 1.0
    assert settings.retrieval_lexical_weight == 1.0
    assert settings.retrieval_fusion_k == 60
    assert settings.retrieval_reranker == "token_overlap"
    assert settings.retrieval_reranker_weight == 0.75
