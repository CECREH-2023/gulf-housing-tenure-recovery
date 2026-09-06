"""Interview metadata dataclasses for Gulf Hazard Interview analysis.

This module defines metadata structures for interview documents, respondents,
and search results specific to the disaster interview research.

Usage:
    from src.rag.interview_metadata import InterviewMetadata, RespondentProfile

    metadata = InterviewMetadata(
        respondent_id="R0001",
        tenure_type="renter",
        county="Harris",
        theme_scores={"financial_hardship": 0.85}
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class TenureType(str, Enum):
    """Housing tenure classification."""

    RENTER = "renter"
    HOMEOWNER = "homeowner"
    UNKNOWN = "unknown"


class StructureType(str, Enum):
    """Housing structure type for renters."""

    MULTIFAMILY = "multifamily"  # MFR - apartments, condos
    SINGLE_FAMILY = "single_family"  # SFR - houses, mobile homes
    UNKNOWN = "unknown"


class DisasterType(str, Enum):
    """Types of disasters experienced."""

    HURRICANE = "hurricane"
    FLOOD = "flood"
    TORNADO = "tornado"
    WINTER_STORM = "winter_storm"
    OTHER = "other"


@dataclass
class ThemeScoreSet:
    """Theme classification scores for an interview."""

    housing_status: float = 0.0
    financial_hardship: float = 0.0
    displacement: float = 0.0
    prior_disaster: float = 0.0
    property_damage: float = 0.0
    health_or_disability: float = 0.0
    aid_dependency: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "housing_status": self.housing_status,
            "financial_hardship": self.financial_hardship,
            "displacement": self.displacement,
            "prior_disaster": self.prior_disaster,
            "property_damage": self.property_damage,
            "health_or_disability": self.health_or_disability,
            "aid_dependency": self.aid_dependency,
        }

    def high_confidence_themes(self, threshold: float = 0.7) -> list[str]:
        """Get themes above confidence threshold."""
        return [
            name for name, score in self.to_dict().items()
            if score >= threshold
        ]

    @property
    def max_theme(self) -> tuple[str, float]:
        """Get the theme with highest score."""
        scores = self.to_dict()
        max_name = max(scores, key=scores.get)
        return max_name, scores[max_name]

    @property
    def theme_count(self) -> int:
        """Count themes above 0.5 threshold."""
        return sum(1 for score in self.to_dict().values() if score >= 0.5)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "ThemeScoreSet":
        """Create from dictionary."""
        return cls(
            housing_status=data.get("housing_status", 0.0),
            financial_hardship=data.get("financial_hardship", 0.0),
            displacement=data.get("displacement", 0.0),
            prior_disaster=data.get("prior_disaster", 0.0),
            property_damage=data.get("property_damage", 0.0),
            health_or_disability=data.get("health_or_disability", 0.0),
            aid_dependency=data.get("aid_dependency", 0.0),
        )


@dataclass
class RespondentProfile:
    """Demographic and disaster profile of a respondent."""

    respondent_id: str
    tenure_type: TenureType = TenureType.UNKNOWN
    structure_type: StructureType = StructureType.UNKNOWN

    # Demographics
    county: str = ""
    state: str = "TX"  # Default to Texas (Gulf Coast study)
    age_group: Optional[str] = None
    income_bracket: Optional[str] = None
    household_size: Optional[int] = None

    # Disaster exposure
    disaster_types: list[DisasterType] = field(default_factory=list)
    displaced: bool = False
    insured: bool = False

    # Financial
    financial_loss: Optional[float] = None
    received_aid: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "respondent_id": self.respondent_id,
            "tenure_type": self.tenure_type.value,
            "structure_type": self.structure_type.value,
            "county": self.county,
            "state": self.state,
            "age_group": self.age_group,
            "income_bracket": self.income_bracket,
            "household_size": self.household_size,
            "disaster_types": [d.value for d in self.disaster_types],
            "displaced": self.displaced,
            "insured": self.insured,
            "financial_loss": self.financial_loss,
            "received_aid": self.received_aid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RespondentProfile":
        """Create from dictionary."""
        return cls(
            respondent_id=data.get("respondent_id", ""),
            tenure_type=TenureType(data.get("tenure_type", "unknown")),
            structure_type=StructureType(data.get("structure_type", "unknown")),
            county=data.get("county", ""),
            state=data.get("state", "TX"),
            age_group=data.get("age_group"),
            income_bracket=data.get("income_bracket"),
            household_size=data.get("household_size"),
            disaster_types=[DisasterType(d) for d in data.get("disaster_types", [])],
            displaced=data.get("displaced", False),
            insured=data.get("insured", False),
            financial_loss=data.get("financial_loss"),
            received_aid=data.get("received_aid", False),
        )


@dataclass
class InterviewMetadata:
    """Complete metadata for an interview document.

    Combines respondent profile with theme scores and source information.
    """

    respondent_id: str
    tenure_type: str = "unknown"  # "renter" or "homeowner"
    county: str = ""
    state: str = "TX"

    # Theme classification
    theme_scores: dict[str, float] = field(default_factory=dict)

    # Respondent characteristics
    structure_type: str = "unknown"
    displaced: bool = False
    insured: bool = False
    received_aid: bool = False
    financial_loss: Optional[float] = None

    # Disaster exposure
    disaster_types: list[str] = field(default_factory=list)

    # Source tracking
    source: str = "gulf_hazard_interview"
    chunk_index: int = 0
    total_chunks: int = 1

    # Timestamps
    interview_date: Optional[datetime] = None
    indexed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for FAISS metadata storage."""
        return {
            "respondent_id": self.respondent_id,
            "tenure_type": self.tenure_type,
            "county": self.county,
            "state": self.state,
            "theme_scores": self.theme_scores,
            "structure_type": self.structure_type,
            "displaced": self.displaced,
            "insured": self.insured,
            "received_aid": self.received_aid,
            "financial_loss": self.financial_loss,
            "disaster_types": self.disaster_types,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "interview_date": self.interview_date.isoformat() if self.interview_date else None,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterviewMetadata":
        """Create from dictionary."""
        interview_date = None
        if data.get("interview_date"):
            interview_date = datetime.fromisoformat(data["interview_date"])

        indexed_at = None
        if data.get("indexed_at"):
            indexed_at = datetime.fromisoformat(data["indexed_at"])

        return cls(
            respondent_id=data.get("respondent_id", ""),
            tenure_type=data.get("tenure_type", "unknown"),
            county=data.get("county", ""),
            state=data.get("state", "TX"),
            theme_scores=data.get("theme_scores", {}),
            structure_type=data.get("structure_type", "unknown"),
            displaced=data.get("displaced", False),
            insured=data.get("insured", False),
            received_aid=data.get("received_aid", False),
            financial_loss=data.get("financial_loss"),
            disaster_types=data.get("disaster_types", []),
            source=data.get("source", "gulf_hazard_interview"),
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks", 1),
            interview_date=interview_date,
            indexed_at=indexed_at,
        )

    @property
    def is_renter(self) -> bool:
        """Check if respondent is a renter."""
        return self.tenure_type.lower() == "renter"

    @property
    def is_homeowner(self) -> bool:
        """Check if respondent is a homeowner."""
        return self.tenure_type.lower() == "homeowner"

    @property
    def high_confidence_themes(self) -> list[str]:
        """Get themes with score >= 0.7."""
        return [
            theme for theme, score in self.theme_scores.items()
            if score >= 0.7
        ]

    @property
    def theme_burden(self) -> int:
        """Count of high-confidence themes (multi-theme burden)."""
        return len(self.high_confidence_themes)


@dataclass
class InterviewSearchResult:
    """Search result with interview-specific context."""

    content: str
    score: float
    rank: int
    metadata: InterviewMetadata

    @property
    def respondent_id(self) -> str:
        """Get respondent ID."""
        return self.metadata.respondent_id

    @property
    def tenure_type(self) -> str:
        """Get tenure type."""
        return self.metadata.tenure_type

    @property
    def summary(self) -> str:
        """Get a brief summary of the result."""
        themes = ", ".join(self.metadata.high_confidence_themes[:3]) or "none"
        return (
            f"{self.respondent_id} ({self.tenure_type}): "
            f"score={self.score:.3f}, themes=[{themes}]"
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "TenureType",
    "StructureType",
    "DisasterType",

    # Dataclasses
    "ThemeScoreSet",
    "RespondentProfile",
    "InterviewMetadata",
    "InterviewSearchResult",
]
