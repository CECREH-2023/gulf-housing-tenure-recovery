"""Gulf_Hazard_Interview configuration using pydantic.

This module provides centralized configuration for the Gulf Hazard Interview
research project, including paths, embedding settings, and CENTAUR integration.

Usage:
    from config.settings import get_settings

    settings = get_settings()
    print(settings.data_dir)
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for Gulf Hazard Interview project.

    Settings can be overridden via environment variables prefixed with GULF_HAZARD_.
    For example: GULF_HAZARD_CENTAUR_PATH=/custom/path
    """

    # ==========================================================================
    # Project Paths
    # ==========================================================================

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Root directory of the Gulf_Hazard_Interview project",
    )

    @property
    def data_dir(self) -> Path:
        """Data directory for processed and raw data."""
        return self.project_root / "data"

    @property
    def indices_dir(self) -> Path:
        """Directory for FAISS vector indices."""
        return self.data_dir / "indices"

    @property
    def processed_dir(self) -> Path:
        """Directory for processed data files."""
        return self.data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        """Directory for raw interview data."""
        return self.data_dir / "raw"

    @property
    def models_dir(self) -> Path:
        """Directory for model outputs."""
        return self.data_dir / "models"

    # ==========================================================================
    # CENTAUR Connection
    # ==========================================================================

    centaur_path: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "centaur-platform",
        description="Path to CENTAUR centaur-platform installation (with multilanguage support)",
    )

    # ==========================================================================
    # Embedding Settings (aligned with CENTAUR defaults)
    # ==========================================================================

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )

    embedding_dimensions: int = Field(
        default=384,
        description="Embedding vector dimensions (384 for MiniLM)",
    )

    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )

    # ==========================================================================
    # Research Data Parameters
    # ==========================================================================

    n_respondents: int = Field(
        default=521,
        description="Total number of interview respondents",
    )

    n_renters: int = Field(
        default=258,
        description="Number of renter respondents",
    )

    n_homeowners: int = Field(
        default=263,
        description="Number of homeowner respondents",
    )

    themes: list[str] = Field(
        default=[
            "housing_status",
            "financial_hardship",
            "displacement",
            "prior_disaster",
            "property_damage",
            "health_or_disability",
            "aid_dependency",
        ],
        description="Research theme categories for BERT classification",
    )

    theme_threshold: float = Field(
        default=0.7,
        description="Default threshold for high-confidence theme classification",
    )

    # ==========================================================================
    # LLM Settings
    # ==========================================================================

    default_provider: str = Field(
        default="anthropic",
        description="Default LLM provider (anthropic, openai, ollama)",
    )

    temperature: float = Field(
        default=0.1,
        description="LLM temperature for analysis tasks (low for consistency)",
    )

    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM responses",
    )

    # ==========================================================================
    # RAG Settings
    # ==========================================================================

    retrieval_k: int = Field(
        default=10,
        description="Default number of passages to retrieve",
    )

    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity score for retrieval",
    )

    # ==========================================================================
    # File Paths (computed properties for key data files)
    # ==========================================================================

    @property
    def bert_scores_path(self) -> Path:
        """Path to BERT theme scores CSV."""
        return self.processed_dir / "bert_scores" / "zero_shot_theme_scores.csv"

    @property
    def regression_dataset_path(self) -> Path:
        """Path to regression-ready dataset."""
        return self.processed_dir / "regression" / "regression_dataset_with_bert_themes.csv"

    @property
    def insurance_dataset_path(self) -> Path:
        """Path to insurance model dataset."""
        return self.processed_dir / "insurance" / "insurance_model_dataset.csv"

    @property
    def displacement_geojson_path(self) -> Path:
        """Path to displacement GeoJSON."""
        return self.processed_dir / "displacement" / "neighborhood_polygons.geojson"

    @property
    def interviews_index_path(self) -> Path:
        """Path to FAISS interview index."""
        return self.indices_dir / "interviews"

    # ==========================================================================
    # Pydantic Configuration
    # ==========================================================================

    class Config:
        env_prefix = "GULF_HAZARD_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance with all configuration values.
    """
    return Settings()


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    settings = get_settings()

    directories = [
        settings.data_dir,
        settings.indices_dir,
        settings.processed_dir,
        settings.raw_dir,
        settings.models_dir,
        settings.processed_dir / "bert_scores",
        settings.processed_dir / "regression",
        settings.processed_dir / "insurance",
        settings.processed_dir / "displacement",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "Settings",
    "get_settings",
    "ensure_directories",
]
