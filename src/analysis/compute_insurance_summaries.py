#!/usr/bin/env python3
"""Compute insurance coverage summaries used across Gulf Interview reporting.

This script rebuilds the tenure- and income-level aggregates that feed the
figure pipeline and manuscript tables. It corrects earlier miscoding that
treated ``insurance_yes`` as an indicator for *uninsured* households.
Outputs include both the share insured (primary) and share uninsured figures
to make downstream provenance explicit.

This module is integrated with CENTAUR infrastructure for path management.

Usage:
    python -m src.analysis.compute_insurance_summaries
    python -m src.analysis.compute_insurance_summaries --verbose
"""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd

from config.settings import get_settings

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
settings = get_settings()

REGRESSION_PATH = settings.processed_dir / "regression" / "regression_dataset_with_bert_themes.csv"
ALIGNED_PATH = settings.processed_dir / "insurance" / "insurance_model_dataset.csv"
OUTPUT_DIR = settings.processed_dir / "insurance"

TENURE_COL = "htype"
ALIGNED_COL = "coverage_aligned"
INCOME_COL = "1.9"
STRUCTURE_COL = "structure_category"

# Preserve the income ordering used throughout the project.
INCOME_ORDER: List[str] = [
    "Less than $10,000",
    "$10,000 to $19,999",
    "$20,000 to $29,999",
    "$30,000 to $39,999",
    "$40,000 to $49,999",
    "$50,000 to $59,999",
    "$60,000 to $69,999",
    "$70,000 to $79,999",
    "$80,000 to $89,999",
    "$90,000 to $99,999",
    "$100,000 to $149,999",
    "$150,000 or more",
]


def load_dataset() -> pd.DataFrame:
    """Load regression data and merge peril-aligned insurance coverage."""
    base = pd.read_csv(REGRESSION_PATH, low_memory=False)
    keep = [
        "ResponseId",
        TENURE_COL,
        INCOME_COL,
        STRUCTURE_COL,
    ]
    base_subset = base[keep].copy()

    aligned = pd.read_csv(ALIGNED_PATH, usecols=["response_id", "coverage_peril_aligned"])
    merged = base_subset.merge(
        aligned,
        left_on="ResponseId",
        right_on="response_id",
        how="left",
    )
    merged[ALIGNED_COL] = pd.to_numeric(merged["coverage_peril_aligned"], errors="coerce")
    merged = merged.drop(columns=["response_id", "coverage_peril_aligned"])
    merged = merged.dropna(subset=[TENURE_COL, ALIGNED_COL])
    return merged


def summarise_by_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate insurance coverage by tenure."""
    grouped = df.groupby(TENURE_COL)[ALIGNED_COL]
    summary = grouped.agg(["mean", "count"]).reset_index()
    summary = summary.rename(
        columns={
            TENURE_COL: "htype",
            "mean": "share_insured",
            "count": "count",
        }
    )
    summary["mean"] = summary["share_insured"]
    summary["share_uninsured"] = 1.0 - summary["share_insured"]
    return summary


def summarise_by_tenure_income(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate insurance coverage by tenure and income bracket."""
    subset = df.dropna(subset=[INCOME_COL])
    # Ensure consistent income ordering (keep only observed categories).
    categories = [label for label in INCOME_ORDER if label in subset[INCOME_COL].unique()]
    subset[INCOME_COL] = pd.Categorical(subset[INCOME_COL], categories=categories, ordered=True)
    grouped = subset.groupby([TENURE_COL, INCOME_COL])[ALIGNED_COL]
    summary = grouped.agg(["mean", "count"]).reset_index()
    summary = summary.rename(
        columns={
            TENURE_COL: "htype",
            INCOME_COL: "income_bracket",
            "mean": "share_insured",
            "count": "count",
        }
    )
    summary["mean"] = summary["share_insured"]
    summary["share_uninsured"] = 1.0 - summary["share_insured"]
    # Sort for readability.
    summary = summary.sort_values(["income_bracket", "htype"]).reset_index(drop=True)
    return summary


def summarise_renter_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Summaries for renter sub-groups used in manuscript discussion."""
    renters = df[df[TENURE_COL] == "Renter"].copy()

    def classify_structure(raw: str | float) -> str:
        value = (raw or "") if isinstance(raw, str) else ""
        lower = value.lower()
        if "multifamily" in lower or "apartment" in lower:
            return "Multifamily"
        if any(keyword in lower for keyword in ("single family", "mobile", "manufactured")):
            return "Single-family/Mobile"
        return "Other / Missing"

    renters["structure_group"] = renters[STRUCTURE_COL].apply(classify_structure)
    grouped = renters.groupby("structure_group")[ALIGNED_COL]
    summary = grouped.agg(["mean", "count"]).reset_index()
    summary = summary.rename(
        columns={
            "mean": "share_insured",
            "count": "count",
        }
    )
    summary["mean"] = summary["share_insured"]
    summary["share_uninsured"] = 1.0 - summary["share_insured"]
    summary = summary.sort_values("count", ascending=False).reset_index(drop=True)
    return summary


def write_csv(df: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / name, index=False)


def main(args: argparse.Namespace) -> None:
    df = load_dataset()

    tenure_table = summarise_by_tenure(df)
    write_csv(tenure_table, "insurance_by_tenure.csv")

    tenure_income_table = summarise_by_tenure_income(df)
    write_csv(tenure_income_table, "insurance_by_tenure_income.csv")

    renter_structure = summarise_renter_structure(df)
    write_csv(renter_structure, "insurance_by_renter_structure.csv")

    if args.verbose:
        pd.set_option("display.float_format", "{:.3f}".format)
        print("Insurance coverage by tenure:")
        print(tenure_table)
        print("\nInsurance coverage by tenure and income:")
        print(tenure_income_table.head(12))
        print("\nRenter coverage by structure group:")
        print(renter_structure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print summaries to stdout.")
    main(parser.parse_args())
