#!/usr/bin/env python3
"""Recompute insurance logistic regression tables with consistent renter coding.

This module is integrated with CENTAUR infrastructure for path management.

Usage:
    python -m src.analysis.run_insurance_models
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config.settings import get_settings

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
settings = get_settings()

LOGIT_DIR = settings.processed_dir / "logit"
FACTOR_PATH = (
    settings.processed_dir
    / "factor_analysis"
    / "polychoric_fa"
    / "promax"
    / "factor_scores_with_ids.csv"
)
REGRESSION_PATH = settings.processed_dir / "regression" / "regression_dataset_with_bert_themes.csv"

SURVEY_YEAR = 2021


def _ordered_map(values: Sequence[str]) -> Mapping[str, int]:
    return {value: idx + 1 for idx, value in enumerate(values)}


def prepare_dataset() -> pd.DataFrame:
    base = pd.read_csv(REGRESSION_PATH, low_memory=False)
    factors = pd.read_csv(
        FACTOR_PATH,
        usecols=[
            "ResponseId",
            "attachment_belonging",
            "move_intent_disamenity",
            "trust_safety",
            "social_cohesion",
            "quality_of_life",
        ],
    )

    df = (
        base.merge(factors, on="ResponseId", how="left", suffixes=("", "_factor"))
        .rename(columns={"ResponseId": "response_id"})
        .dropna(subset=["insurance_yes"])
    )

    income_map = _ordered_map(
        [
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
    )
    education_map = _ordered_map(
        [
            "Less than high school degree",
            "High school graduate (high school diploma or equivalent including GED)",
            "Some college but no degree",
            "Associate degree in college (2-year)",
            "Bachelor's degree in college (4-year)",
            "Master's degree",
            "Professional degree (JD, MD)",
            "Doctoral degree",
        ]
    )
    damage_map = {
        "No damage": 0,
        "Not very extensive": 1,
        "Somewhat extensive": 2,
        "Very extensive": 3,
    }

    df["renter_flag"] = (df["htype"] == "Renter").astype(float)
    df["mfr_flag"] = df["structure_category"].str.contains("Multifamily", case=False, na=False).astype(float)
    df["income_ord"] = df["1.9"].map(income_map)
    df["employed"] = (df["1.8"] == "Employed").astype(float)
    df["age_years"] = SURVEY_YEAR - df["age"]
    df["household_size"] = df["hsize"]
    df["damage_ord"] = df["2.8_1"].map(damage_map)
    df["aid_flag"] = (df["2.12"] == "Yes").astype(float)
    df["displacement_flag"] = pd.to_numeric(df["displaced_event_flag"], errors="coerce")
    df["female"] = (df["gender"] == "Woman").astype(float)
    df["nonwhite"] = (~df["race"].fillna("").str.contains("White", case=False)).astype(float)
    df["hispanic"] = df["race"].fillna("").str.contains("Hispanic", case=False).astype(float)
    df["education_ord"] = df["education"].map(education_map)
    df["married"] = (df["marital"] == "Married").astype(float)

    factor_cols = [
        "attachment_belonging",
        "move_intent_disamenity",
        "trust_safety",
        "social_cohesion",
        "quality_of_life",
    ]
    df[factor_cols] = df[factor_cols].apply(pd.to_numeric, errors="coerce")
    return df


def dropna_for(features: Iterable[str], df: pd.DataFrame) -> pd.DataFrame:
    return df[["insurance_yes", *features]].dropna()


def fit_logit(features: Sequence[str], data: pd.DataFrame) -> sm.Logit:
    cleaned = dropna_for(features, data)
    X = sm.add_constant(cleaned[features])
    y = cleaned["insurance_yes"]
    model = sm.Logit(y, X, missing="raise")
    return model.fit(disp=False)


def coef_table(result: sm.Logit) -> pd.DataFrame:
    params = result.params
    bse = result.bse
    z_scores = params / bse
    conf_int = result.conf_int()
    table = pd.DataFrame(
        {
            "Coef.": params,
            "Std.Err.": bse,
            "z": z_scores,
            "P>|z|": result.pvalues,
            "[0.025": conf_int[0],
            "0.975]": conf_int[1],
        }
    )
    table.index.name = ""
    return table


def odds_ratio_table(result: sm.Logit, scenario: str, label_map: Mapping[str, str]) -> pd.DataFrame:
    params = result.params
    conf_int = result.conf_int()
    frame = pd.DataFrame(
        {
            "scenario": scenario,
            "variable": params.index,
            "odds_ratio": np.exp(params),
            "ci_lower": np.exp(conf_int[0]),
            "ci_upper": np.exp(conf_int[1]),
            "p_value": result.pvalues,
        }
    )
    frame["label"] = frame["variable"].map(label_map).fillna(frame["variable"])
    return frame


def write_coef(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)


def run() -> None:
    data = prepare_dataset()

    base_features = [
        "renter_flag",
        "income_ord",
        "damage_ord",
        "aid_flag",
        "displacement_flag",
        "age_years",
        "household_size",
        "employed",
        "attachment_belonging",
        "move_intent_disamenity",
        "trust_safety",
        "social_cohesion",
        "quality_of_life",
    ]
    structure_features = [
        "renter_flag",
        "mfr_flag",
        "income_ord",
        "employed",
        "age_years",
        "household_size",
        "damage_ord",
        "attachment_belonging",
    ]
    demographics_features = structure_features + [
        "aid_flag",
        "displacement_flag",
        "female",
        "nonwhite",
        "hispanic",
        "education_ord",
        "married",
    ]

    labels = {
        "const": "const",
        "renter_flag": "Renter (1) vs homeowner (0)",
        "income_ord": "Household income (ordinal)",
        "damage_ord": "Self-rated damage severity",
        "aid_flag": "Received aid (1=yes)",
        "displacement_flag": "Displaced by event (1=yes)",
        "age_years": "Age (years)",
        "household_size": "Household size",
        "employed": "Currently employed (1=yes)",
        "attachment_belonging": "Attachment/Belonging",
        "move_intent_disamenity": "Move Intent/Disamenity",
        "trust_safety": "Trust/Safety",
        "social_cohesion": "Social Cohesion",
        "quality_of_life": "Quality of Life",
        "mfr_flag": "Multifamily structure (1=yes)",
        "female": "Female (1=yes)",
        "nonwhite": "Race != White",
        "hispanic": "Hispanic (1=yes)",
        "education_ord": "Education (ordinal)",
        "married": "Married (1=yes)",
    }

    full_result = fit_logit(base_features, data)
    write_coef(coef_table(full_result), LOGIT_DIR / "logit_insurance_clustered.csv")
    odds = odds_ratio_table(full_result, "insurance", labels)
    odds.to_csv(LOGIT_DIR / "logit_insurance_odds_ratios.csv", index=False)
    odds.to_csv(LOGIT_DIR / "logit_insurance_odds_ratios_labeled.csv", index=False)

    structure_result = fit_logit(structure_features, data)
    write_coef(coef_table(structure_result), LOGIT_DIR / "logit_with_structure_insurance_yes.csv")

    demo_result = fit_logit(demographics_features, data)
    demo_table = coef_table(demo_result)
    write_coef(demo_table, LOGIT_DIR / "logit_with_demographics_insurance_yes.csv")
    write_coef(demo_table, LOGIT_DIR / "logit_insurance_mfr_demographics.csv")
    write_coef(demo_table, LOGIT_DIR / "logit_renter_insurance_with_demographics.csv")


if __name__ == "__main__":
    run()
