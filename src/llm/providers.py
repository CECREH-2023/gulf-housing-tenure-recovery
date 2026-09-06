"""LLM provider interface for Gulf Hazard Interview analysis.

This module re-exports CENTAUR LLM components and provides interview-specific
utilities for theme classification and research analysis.

Usage:
    from src.llm.providers import get_provider, LLMMessage, embed_texts

    # Get LLM provider for analysis
    provider = get_provider()

    # Create messages
    messages = [
        LLMMessage.system("You are a disaster research analyst."),
        LLMMessage.user("Analyze this interview excerpt...")
    ]

    # Get completion
    response = await provider.complete(messages)
"""

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import json

# Import from centaur_link
from centaur_link import (
    get_llm_provider,
    get_embedding_provider,
    get_llm_message_class,
    get_llm_response_class,
    get_message_role_enum,
    create_system_message,
    create_user_message,
    create_assistant_message,
    ensure_centaur_available,
)

from config.settings import get_settings


# =============================================================================
# Re-export CENTAUR Classes (lazy loading)
# =============================================================================

def _get_classes():
    """Lazy load CENTAUR classes."""
    ensure_centaur_available()
    return {
        "LLMMessage": get_llm_message_class(),
        "LLMResponse": get_llm_response_class(),
        "MessageRole": get_message_role_enum(),
    }


# For type hints, we create wrapper references
LLMMessage = property(lambda self: _get_classes()["LLMMessage"])
LLMResponse = property(lambda self: _get_classes()["LLMResponse"])
MessageRole = property(lambda self: _get_classes()["MessageRole"])


# =============================================================================
# Provider Functions
# =============================================================================

def get_provider(provider_type: Optional[str] = None):
    """Get an LLM provider for interview analysis.

    Args:
        provider_type: Optional provider type. If None, uses settings default.

    Returns:
        An LLM provider instance.
    """
    settings = get_settings()
    return get_llm_provider(provider_type or settings.default_provider)


def get_embedder(model_name: Optional[str] = None):
    """Get an embedding provider for semantic search.

    Args:
        model_name: Optional model name. If None, uses settings default.

    Returns:
        An embedding provider instance.
    """
    settings = get_settings()
    return get_embedding_provider(model_name or settings.embedding_model)


# =============================================================================
# Interview Analysis Prompts
# =============================================================================

THEME_CLASSIFICATION_SYSTEM_PROMPT = """You are a disaster research analyst specializing in
Gulf Coast hurricane impacts on renters and homeowners.

Your task is to analyze interview excerpts and identify themes related to:
- housing_status: Housing conditions, damage, habitability, landlord issues
- financial_hardship: Economic losses, job loss, inability to afford necessities
- displacement: Evacuation, relocation, temporary housing, return challenges
- prior_disaster: Previous disaster experiences, repeated exposure
- property_damage: Physical damage to home and possessions
- health_or_disability: Physical and mental health impacts
- aid_dependency: Government assistance, FEMA, charities

For each theme, provide a confidence score from 0.0 to 1.0 based on how strongly
the text relates to that theme. Be conservative - only assign high scores (>0.7)
when there is clear, explicit evidence in the text."""

INTERPRETATION_SYSTEM_PROMPT = """You are a disaster research analyst interpreting
statistical findings from a study of 521 Gulf Coast disaster interviews.

The study compares 258 renters with 263 homeowners across 7 themes:
- Housing status (landlord issues, rental instability)
- Financial hardship (job loss, inability to afford necessities)
- Displacement (evacuation, temporary housing)
- Prior disaster (previous disaster experiences)
- Property damage (physical damage to home)
- Health/disability (physical and mental health impacts)
- Aid dependency (government assistance)

Key findings: Renters face concentrated financial stressors (+6.5pp financial hardship,
+2.7pp housing instability) compared to homeowners. The gap is about resilience,
NOT exposure - both groups have similar disaster exposure rates.

When interpreting results, focus on actionable insights for disaster policy."""


# =============================================================================
# Theme Classification
# =============================================================================

@dataclass
class ThemeScores:
    """Theme classification scores for an interview excerpt."""

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
        scores = self.to_dict()
        return [theme for theme, score in scores.items() if score >= threshold]

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "ThemeScores":
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


async def classify_themes(
    text: str,
    provider=None,
) -> ThemeScores:
    """Classify interview text against research themes using LLM.

    Args:
        text: Interview excerpt to classify.
        provider: Optional LLM provider. If None, uses default.

    Returns:
        ThemeScores with confidence scores for each theme.
    """
    if provider is None:
        provider = get_provider()

    settings = get_settings()
    LLMMessage = get_llm_message_class()

    prompt = f"""Analyze this disaster interview excerpt and score how strongly it relates to each theme.

Return ONLY a JSON object with scores from 0.0 to 1.0 for each theme:
{{
    "housing_status": <score>,
    "financial_hardship": <score>,
    "displacement": <score>,
    "prior_disaster": <score>,
    "property_damage": <score>,
    "health_or_disability": <score>,
    "aid_dependency": <score>
}}

Interview excerpt:
"{text}"

JSON scores:"""

    messages = [
        LLMMessage.system(THEME_CLASSIFICATION_SYSTEM_PROMPT),
        LLMMessage.user(prompt),
    ]

    response = await provider.complete(
        messages,
        temperature=settings.temperature,
        max_tokens=256,
    )

    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        scores_dict = json.loads(content)
        return ThemeScores.from_dict(scores_dict)
    except (json.JSONDecodeError, KeyError) as e:
        # Return empty scores on parse failure
        return ThemeScores()


# =============================================================================
# Embedding Functions
# =============================================================================

async def embed_text(text: str, model_name: Optional[str] = None) -> list[float]:
    """Embed a single text using CENTAUR embeddings.

    Args:
        text: Text to embed.
        model_name: Optional model name.

    Returns:
        Embedding vector as list of floats.
    """
    embedder = get_embedder(model_name)
    result = await embedder.embed([text])
    return result.embeddings[0]


async def embed_texts(
    texts: list[str],
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> list[list[float]]:
    """Embed multiple texts using CENTAUR embeddings.

    Args:
        texts: List of texts to embed.
        model_name: Optional model name.
        batch_size: Optional batch size.

    Returns:
        List of embedding vectors.
    """
    settings = get_settings()
    embedder = get_embedder(model_name)
    batch = batch_size or settings.embedding_batch_size

    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i + batch]
        result = await embedder.embed(batch_texts)
        all_embeddings.extend(result.embeddings)

    return all_embeddings


# =============================================================================
# Statistical Interpretation
# =============================================================================

async def interpret_regression_results(
    coefficients: dict[str, float],
    model_type: str = "logistic",
    context: Optional[str] = None,
    provider=None,
) -> str:
    """Generate LLM interpretation of regression results.

    Args:
        coefficients: Dictionary of variable names to coefficients.
        model_type: Type of regression ("logistic", "ols", "mixed").
        context: Optional additional context about the analysis.
        provider: Optional LLM provider.

    Returns:
        Natural language interpretation of the results.
    """
    if provider is None:
        provider = get_provider()

    settings = get_settings()
    LLMMessage = get_llm_message_class()

    # Format coefficients for prompt
    coef_str = "\n".join(f"  {var}: {coef:.4f}" for var, coef in coefficients.items())

    prompt = f"""Interpret these {model_type} regression results from the Gulf Hazard Interview study:

Coefficients:
{coef_str}

{f"Additional context: {context}" if context else ""}

Provide a concise interpretation focusing on:
1. Which variables have the strongest effects?
2. What do these findings mean for renters vs homeowners?
3. What policy implications emerge?

Keep the interpretation to 2-3 paragraphs."""

    messages = [
        LLMMessage.system(INTERPRETATION_SYSTEM_PROMPT),
        LLMMessage.user(prompt),
    ]

    response = await provider.complete(
        messages,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    return response.content


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Provider functions
    "get_provider",
    "get_embedder",

    # Message helpers
    "create_system_message",
    "create_user_message",
    "create_assistant_message",

    # Theme classification
    "ThemeScores",
    "classify_themes",

    # Embeddings
    "embed_text",
    "embed_texts",

    # Interpretation
    "interpret_regression_results",

    # Prompts
    "THEME_CLASSIFICATION_SYSTEM_PROMPT",
    "INTERPRETATION_SYSTEM_PROMPT",
]
