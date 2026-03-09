from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PROMPTS_DIR = ROOT_DIR / "prompts"
FOUNDER_PROMPTS_DIR = ROOT_DIR / "founder_prompts"


class Settings(BaseSettings):
    """Application settings shared across the experiment pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="USM_",
        extra="ignore",
    )

    environment: str = "development"
    llm_provider: str = "openai"
    llm_model: str = "gpt-5-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    founder_prompts_path: Path = Field(
        default=FOUNDER_PROMPTS_DIR / "restaurants.example.json"
    )
    yelp_businesses_path: Path = Field(
        default=DATA_DIR / "raw" / "yelp_academic_dataset_business.json"
    )
    yelp_reviews_path: Path = Field(
        default=DATA_DIR / "raw" / "yelp_academic_dataset_review.json"
    )
    working_subset_path: Path = Field(
        default=DATA_DIR / "processed" / "restaurant_reviews.jsonl"
    )
    index_dir: Path = Field(default=ARTIFACTS_DIR / "index")
    run_artifacts_dir: Path = Field(default=ARTIFACTS_DIR / "runs")
    prompts_dir: Path = Field(default=PROMPTS_DIR)

    retrieval_top_k: int = 50
    synthesis_evidence_k: int = 15
    min_focus_points: int = 3
    max_focus_points: int = 5


def ensure_scaffold_directories(settings: Settings) -> list[Path]:
    """Create the local working directories used by the project."""

    directories = [
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        ARTIFACTS_DIR,
        settings.index_dir,
        settings.run_artifacts_dir,
        settings.prompts_dir,
        FOUNDER_PROMPTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return directories


@lru_cache
def get_settings() -> Settings:
    return Settings()
