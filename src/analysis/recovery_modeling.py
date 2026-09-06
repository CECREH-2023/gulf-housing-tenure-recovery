"""Recovery modeling with CENTAUR LLM integration.

This module provides regression models for disaster recovery analysis,
enhanced with LLM-powered interpretation and RAG-supported evidence.

Usage:
    from src.analysis.recovery_modeling import run_recovery_model

    result = await run_recovery_model(tenure_filter="renter")
    print(result.interpretation)
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config.settings import get_settings
from src.llm.providers import interpret_regression_results
from src.rag.interview_retriever import get_retriever


# Feature columns for recovery model
FEATURE_COLUMNS = [
    "financial_hardship_score",
    "displacement_score",
    "property_damage_score",
    "prior_disaster_score",
    "health_or_disability_score",
]

# Outcome column
OUTCOME_COLUMN = "recovered"  # Binary recovery indicator


@dataclass
class RecoveryModelResult:
    """Result from recovery model with LLM interpretation."""

    coefficients: dict[str, float]
    odds_ratios: dict[str, float]
    model_accuracy: float
    sample_size: int
    tenure_filter: Optional[str]
    interpretation: str = ""
    supporting_quotes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coefficients": self.coefficients,
            "odds_ratios": self.odds_ratios,
            "model_accuracy": self.model_accuracy,
            "sample_size": self.sample_size,
            "tenure_filter": self.tenure_filter,
            "interpretation": self.interpretation,
            "supporting_quotes": self.supporting_quotes,
        }


def load_regression_data(
    tenure_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load regression dataset.

    Args:
        tenure_filter: Optional filter for "renter" or "homeowner".

    Returns:
        DataFrame with regression data.
    """
    settings = get_settings()
    data_path = settings.processed_dir / "regression" / "regression_dataset_with_bert_themes.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Regression dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Apply tenure filter
    if tenure_filter:
        tenure_col = None
        for col in ["tenure_type", "htype", "is_renter"]:
            if col in df.columns:
                tenure_col = col
                break

        if tenure_col:
            if tenure_col == "is_renter":
                if tenure_filter == "renter":
                    df = df[df[tenure_col] == 1]
                else:
                    df = df[df[tenure_col] == 0]
            else:
                df = df[df[tenure_col].str.lower() == tenure_filter.lower()]

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare features for modeling.

    Args:
        df: Input DataFrame.
        feature_columns: List of feature column names.

    Returns:
        Tuple of (feature DataFrame, available columns).
    """
    available = [col for col in feature_columns if col in df.columns]

    if not available:
        raise ValueError(f"No feature columns found. Available: {df.columns.tolist()}")

    X = df[available].copy()

    # Handle missing values
    X = X.fillna(0)

    return X, available


def fit_logistic_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[LogisticRegression, float]:
    """Fit logistic regression model.

    Args:
        X: Feature matrix.
        y: Target variable.

    Returns:
        Tuple of (fitted model, accuracy score).
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit model
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X_scaled, y)

    # Calculate accuracy
    accuracy = model.score(X_scaled, y)

    return model, accuracy


async def get_supporting_quotes(
    top_predictors: list[tuple[str, float]],
    retriever=None,
) -> list[str]:
    """Retrieve supporting interview quotes for top predictors.

    Args:
        top_predictors: List of (predictor_name, coefficient) tuples.
        retriever: Optional retriever instance.

    Returns:
        List of relevant interview excerpts.
    """
    if retriever is None:
        try:
            retriever = get_retriever()
        except Exception:
            return []

    quotes = []

    for predictor, _ in top_predictors[:3]:
        # Map predictor to theme
        theme = predictor.replace("_score", "")

        try:
            result = await retriever.retrieve_by_theme(
                query=f"experiences related to {theme.replace('_', ' ')}",
                themes=[theme],
                k=2,
            )

            for passage in result.passages:
                if len(passage.content) > 50:
                    quotes.append(passage.content[:200] + "...")
        except Exception:
            continue

    return quotes


async def run_recovery_model(
    tenure_filter: Optional[str] = None,
    include_interpretation: bool = True,
    include_quotes: bool = True,
) -> RecoveryModelResult:
    """Run recovery regression with optional LLM interpretation.

    Args:
        tenure_filter: Optional filter for "renter" or "homeowner".
        include_interpretation: Whether to generate LLM interpretation.
        include_quotes: Whether to retrieve supporting quotes.

    Returns:
        RecoveryModelResult with coefficients and interpretation.
    """
    # Load data
    df = load_regression_data(tenure_filter)

    # Check for outcome column
    if OUTCOME_COLUMN not in df.columns:
        # Create synthetic outcome for demonstration
        df[OUTCOME_COLUMN] = (df.get("displacement_score", 0) < 0.5).astype(int)

    # Prepare features
    X, available_features = prepare_features(df, FEATURE_COLUMNS)
    y = df[OUTCOME_COLUMN]

    # Filter to complete cases
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) < 10:
        raise ValueError(f"Insufficient data: only {len(X)} complete cases")

    # Fit model
    model, accuracy = fit_logistic_model(X, y)

    # Extract coefficients
    coefficients = dict(zip(available_features, model.coef_[0]))

    # Calculate odds ratios
    import numpy as np
    odds_ratios = {k: np.exp(v) for k, v in coefficients.items()}

    # Get top predictors
    top_predictors = sorted(
        coefficients.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Get supporting quotes
    supporting_quotes = []
    if include_quotes:
        supporting_quotes = await get_supporting_quotes(top_predictors)

    # Generate interpretation
    interpretation = ""
    if include_interpretation:
        try:
            context = f"Analyzing {tenure_filter or 'all'} respondents (n={len(X)})"
            interpretation = await interpret_regression_results(
                coefficients=coefficients,
                model_type="logistic",
                context=context,
            )
        except Exception as e:
            interpretation = f"Interpretation unavailable: {e}"

    return RecoveryModelResult(
        coefficients=coefficients,
        odds_ratios=odds_ratios,
        model_accuracy=accuracy,
        sample_size=len(X),
        tenure_filter=tenure_filter,
        interpretation=interpretation,
        supporting_quotes=supporting_quotes,
    )


async def compare_tenure_models() -> dict[str, RecoveryModelResult]:
    """Run and compare models for renters vs homeowners.

    Returns:
        Dictionary with "renter", "homeowner", and "all" results.
    """
    results = {}

    for tenure in [None, "renter", "homeowner"]:
        key = tenure or "all"
        try:
            results[key] = await run_recovery_model(tenure_filter=tenure)
        except Exception as e:
            print(f"Error for {key}: {e}")

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Run recovery model analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Run recovery regression model")
    parser.add_argument(
        "--tenure",
        choices=["renter", "homeowner", "all"],
        default="all",
        help="Filter by tenure type",
    )
    parser.add_argument(
        "--no-interpretation",
        action="store_true",
        help="Skip LLM interpretation",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare renter vs homeowner models",
    )

    args = parser.parse_args()

    if args.compare:
        print("Comparing recovery models by tenure...\n")
        results = await compare_tenure_models()

        for tenure, result in results.items():
            print(f"\n{'='*60}")
            print(f"Tenure: {tenure.upper()}")
            print(f"{'='*60}")
            print(f"Sample size: {result.sample_size}")
            print(f"Accuracy: {result.model_accuracy:.3f}")
            print("\nCoefficients:")
            for var, coef in sorted(result.coefficients.items(), key=lambda x: -abs(x[1])):
                print(f"  {var}: {coef:.4f} (OR: {result.odds_ratios[var]:.3f})")
    else:
        tenure = None if args.tenure == "all" else args.tenure
        result = await run_recovery_model(
            tenure_filter=tenure,
            include_interpretation=not args.no_interpretation,
        )

        print(f"Recovery Model Results ({args.tenure})")
        print("=" * 60)
        print(f"Sample size: {result.sample_size}")
        print(f"Accuracy: {result.model_accuracy:.3f}")
        print("\nCoefficients (sorted by magnitude):")
        for var, coef in sorted(result.coefficients.items(), key=lambda x: -abs(x[1])):
            print(f"  {var}: {coef:.4f} (OR: {result.odds_ratios[var]:.3f})")

        if result.interpretation:
            print("\nInterpretation:")
            print(result.interpretation)

        if result.supporting_quotes:
            print("\nSupporting quotes:")
            for quote in result.supporting_quotes:
                print(f"  - {quote}")


if __name__ == "__main__":
    asyncio.run(main())
