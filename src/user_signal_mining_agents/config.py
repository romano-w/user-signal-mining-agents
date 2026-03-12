from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PROMPTS_DIR = ROOT_DIR / "prompts"
FOUNDER_PROMPTS_DIR = ROOT_DIR / "founder_prompts"
DOMAIN_PACKS_PATH = FOUNDER_PROMPTS_DIR / "domain_packs.json"
YELP_DATASET_DIR = DATA_DIR / "raw" / "Yelp-JSON"


class Settings(BaseSettings):
    """Application settings shared across the experiment pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="USM_",
        extra="ignore",
    )

    environment: str = "development"
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "USM_OPENAI_API_KEY"),
    )
    gemini_api_key_1: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY_1", "USM_GEMINI_API_KEY_1", "GEMINI_API_KEY"),
    )
    gemini_api_key_2: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY_2", "USM_GEMINI_API_KEY_2"),
    )
    openrouter_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "USM_OPENROUTER_API_KEY"),
    )
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.1-flash-lite-preview"
    llm_base_url: str = ""
    llm_temperature: float = 0.3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    founder_prompts_path: Path = Field(default=FOUNDER_PROMPTS_DIR / "restaurants.json")
    domain_packs_path: Path = Field(default=DOMAIN_PACKS_PATH)
    active_domains: str = ""
    yelp_download_url: str = "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"
    yelp_dataset_dir: Path = Field(default=YELP_DATASET_DIR)
    yelp_download_zip_path: Path = Field(default=YELP_DATASET_DIR / "Yelp-JSON.zip")
    yelp_tar_path: Path = Field(default=YELP_DATASET_DIR / "yelp_dataset.tar")
    yelp_businesses_path: Path = Field(
        default=YELP_DATASET_DIR / "yelp_academic_dataset_business.json"
    )
    yelp_reviews_path: Path = Field(
        default=YELP_DATASET_DIR / "yelp_academic_dataset_review.json"
    )
    working_subset_path: Path = Field(
        default=DATA_DIR / "processed" / "restaurant_reviews.jsonl"
    )
    index_dir: Path = Field(default=ARTIFACTS_DIR / "index")
    run_artifacts_dir: Path = Field(default=ARTIFACTS_DIR / "runs")
    prompts_dir: Path = Field(default=PROMPTS_DIR)

    restaurant_review_limit: int | None = 75000
    max_reviews_per_business: int = 200
    min_review_characters: int = 60
    chunk_sentence_window: int = 2
    chunk_sentence_stride: int = 1
    max_chunks_per_review: int = 3

    retrieval_top_k: int = 50
    synthesis_evidence_k: int = 15
    embedding_batch_size: int = 128
    min_focus_points: int = 3
    max_focus_points: int = 5


def ensure_scaffold_directories(settings: Settings) -> list[Path]:
    """Create the local working directories used by the project."""

    directories = [
        DATA_DIR / "raw",
        settings.yelp_dataset_dir,
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
