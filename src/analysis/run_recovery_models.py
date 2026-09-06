#!/usr/bin/env python3
"""Re-estimate Gulf Interview logistic models with clustered SEs and scenario outputs.

This module is integrated with CENTAUR infrastructure for path management.

Usage:
    python -m src.analysis.run_recovery_models
    python -m src.analysis.run_recovery_models --include-state-clusters
"""

from __future__ import annotations

import argparse
import warnings
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import default_rng
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_cluster

from config.settings import get_settings
from src.analysis.event_timing import add_event_timing

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
settings = get_settings()

# Data paths from CENTAUR settings
REGRESSION_PATH = settings.processed_dir / "regression" / "regression_dataset_with_bert_themes.csv"
FACTOR_PATH = (
    settings.processed_dir
    / "factor_analysis"
    / "polychoric_fa"
    / "promax"
    / "factor_scores_with_ids.csv"
)
INSURANCE_PATH = settings.processed_dir / "insurance" / "insurance_model_dataset.csv"

# Results directory for model outputs
RESULTS_DIR = settings.project_root / "results" / "clustered_models"

SURVEY_YEAR = 2021

RNG = default_rng(2024)


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

def load_base_dataset() -> pd.DataFrame:
    """Load the regression dataset and merge factor scores."""
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
    insurance = pd.read_csv(
        INSURANCE_PATH, usecols=["response_id", "coverage_peril_aligned"]
    )
    merged = base.merge(factors, on="ResponseId", how="left")
    merged = merged.merge(
        insurance, left_on="ResponseId", right_on="response_id", how="left"
    )
    return merged.drop(columns=["response_id"])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create common_feature columns shared across models."""
    income_map = {label: idx + 1 for idx, label in enumerate(INCOME_ORDER)}
    education_map = {
        "Less than high school degree": 1,
        "High school graduate (high school diploma or equivalent including GED)": 2,
        "Some college but no degree": 3,
        "Associate degree in college (2-year)": 4,
        "Bachelor's degree in college (4-year)": 5,
        "Master's degree": 6,
        "Professional degree (JD, MD)": 7,
        "Doctoral degree": 8,
    }
    damage_map = {
        "No damage": 0,
        "Not very extensive": 1,
        "Somewhat extensive": 2,
        "Very extensive": 3,
    }

    df = add_event_timing(df.copy())
    df["insurance_yes"] = pd.to_numeric(df["insurance_yes"], errors="coerce")
    df["coverage_peril_aligned"] = pd.to_numeric(
        df["coverage_peril_aligned"], errors="coerce"
    )
    df["aid_flag"] = (df["2.12"] == "Yes").astype(float)
    df["displacement_flag"] = pd.to_numeric(df["displaced_event_flag"], errors="coerce")

    df["income_ord"] = df["1.9"].map(income_map)
    df["attachment_factor"] = pd.to_numeric(df["attachment_belonging"], errors="coerce")
    df["employed"] = (df["1.8"] == "Employed").astype(float)
    df["age_years"] = SURVEY_YEAR - pd.to_numeric(df["age"], errors="coerce")
    df["household_size_raw"] = pd.to_numeric(df["hsize"], errors="coerce")
    hh_cap = df["household_size_raw"].quantile(0.99)
    df["household_size"] = df["household_size_raw"].clip(upper=hh_cap)
    df["damage_ord"] = df["2.8_1"].map(damage_map)
    df["renter_flag"] = (df["htype"] == "Renter").astype(float)
    df["mfr_flag"] = df["structure_category"].str.contains("Multifamily", case=False, na=False).astype(float)
    df["female"] = (df["gender"] == "Woman").astype(float)
    df["nonwhite"] = (~df["race"].fillna("").str.contains("White", case=False)).astype(float)
    df["hispanic"] = df["race"].fillna("").str.contains("Hispanic", case=False).astype(float)
    df["education_ord"] = df["education"].map(education_map)
    df["married"] = (df["marital"] == "Married").astype(float)

    df["hazard_cluster"] = df["hazard_category"].fillna("Unknown")
    state = df["state_abbr"].replace("Unknown", pd.NA)
    df["state_cluster"] = state
    df["state_or_hazard_cluster"] = state.fillna(df["hazard_cluster"])

    # Centered variables for non-linearity checks
    df["age_centered"] = df["age_years"] - df["age_years"].mean()
    df["age_centered_sq"] = df["age_centered"] ** 2
    df["hhsize_centered"] = df["household_size"] - df["household_size"].mean()
    df["hhsize_centered_sq"] = df["hhsize_centered"] ** 2
    days_source = df.get("days_since_event_optionB", df.get("days_since_event_optionA"))
    df["days_since_event"] = pd.to_numeric(days_source, errors="coerce")

    return df


def dropna(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return dataframe with required columns non-missing."""
    return df.dropna(subset=list(columns))


class ClusteredResult:
    """Lightweight wrapper storing cluster-robust statistics."""

    def __init__(
        self,
        base_result: sm.discrete.discrete_model.BinaryResultsWrapper,
        cov: np.ndarray,
        exog_names: List[str],
        model_data: pd.DataFrame,
        cluster_col: str,
    ) -> None:
        self.base_result = base_result
        self.params = base_result.params
        self.cov_matrix = pd.DataFrame(cov, index=exog_names, columns=exog_names)
        self.bse = pd.Series(np.sqrt(np.diag(self.cov_matrix)), index=exog_names)
        self.zvalues = self.params / self.bse
        self.pvalues = 2 * (1 - norm.cdf(np.abs(self.zvalues)))
        self.model = base_result.model
        self.model_data = model_data
        self.cluster_col = cluster_col
        self.feature_names = exog_names

    def cov_params(self) -> pd.DataFrame:
        return self.cov_matrix

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        q = norm.ppf(1 - alpha / 2)
        lower = self.params - q * self.bse
        upper = self.params + q * self.bse
        return pd.DataFrame({0: lower, 1: upper})


def fit_logit(
    df: pd.DataFrame,
    outcome: str,
    features: List[str],
    cluster_col: str,
) -> ClusteredResult:
    """Fit logistic regression with clustered covariance."""
    required_cols = [outcome, *features, cluster_col]
    model_data = dropna(df, required_cols).copy()
    if model_data[cluster_col].nunique() < 2:
        raise ValueError(f"Cluster column {cluster_col} has <2 unique groups after dropping NAs.")

    X = sm.add_constant(model_data[features], has_constant="add").astype(float)
    y = model_data[outcome].astype(float)

    model = sm.Logit(y, X)
    result = model.fit(disp=False, maxiter=200)
    cov = cov_cluster(result, model_data[cluster_col])
    exog_names = list(result.model.exog_names)
    return ClusteredResult(result, cov, exog_names, model_data, cluster_col)


def coef_table(result: ClusteredResult) -> pd.DataFrame:
    params = result.params
    bse = result.bse
    z_scores = params / bse
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "coef": params,
            "std_err": bse,
            "z": z_scores,
            "p": result.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
    )
    table.index.name = "variable"
    return table


def odds_ratio_table(result: ClusteredResult) -> pd.DataFrame:
    params = result.params
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "odds_ratio": np.exp(params),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "p": result.pvalues,
        }
    )
    table.index.name = "variable"
    return table


def save_table(df: pd.DataFrame, filename: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / filename)


def model_fit_summary(result: ClusteredResult, model_name: str) -> pd.DataFrame:
    base = result.base_result
    n_obs = result.model_data.shape[0]
    n_clusters = result.model_data[result.cluster_col].nunique()
    llnull = getattr(base, "llnull", np.nan)
    if np.isfinite(llnull) and not np.isclose(llnull, 0.0):
        pseudo_r2 = 1 - (base.llf / llnull)
    else:
        pseudo_r2 = np.nan
    return pd.DataFrame(
        {
            "model": [model_name],
            "n_obs": [n_obs],
            "n_clusters": [n_clusters],
            "cluster_var": [result.cluster_col],
            "log_likelihood": [base.llf],
            "ll_null": [llnull],
            "pseudo_r2": [pseudo_r2],
            "aic": [base.aic],
            "bic": [base.bic],
        }
    )


def simulate_predictions(
    result: ClusteredResult,
    scenarios: pd.DataFrame,
    draws: int = 2000,
) -> pd.DataFrame:
    """Predict probabilities for supplied scenarios (point estimates only)."""
    exog_names = result.feature_names
    cols = [name for name in exog_names if name != "const"]
    X = sm.add_constant(scenarios[cols], has_constant="add")
    X = X[exog_names]
    linear = X.to_numpy(dtype=float) @ result.params.loc[exog_names].to_numpy()
    probs = 1 / (1 + np.exp(-linear))

    summary = []
    for idx, row in scenarios.iterrows():
        summary.append({**row.to_dict(), "predicted": float(probs[idx])})
    return pd.DataFrame(summary)


def build_insurance_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare tenure × income quartile × attachment scenarios."""
    income_values = df["income_ord"].dropna()
    attachment = df["attachment_factor"].dropna()
    income_quartiles = np.quantile(income_values, [0.25, 0.5, 0.75])
    income_lookup = {idx + 1: label for idx, label in enumerate(INCOME_ORDER)}
    attachment_levels = [
        ("-1 SD", attachment.mean() - attachment.std()),
        ("Mean", attachment.mean()),
        ("+1 SD", attachment.mean() + attachment.std()),
    ]

    baseline = {
        "income_ord": income_values.median(),
        "employed": df["employed"].mean(),
        "age_years": df["age_years"].median(),
        "household_size": df["household_size"].median(),
        "damage_ord": df["damage_ord"].median(),
        "mfr_flag": df["mfr_flag"].mean(),
    }

    rows: List[Mapping[str, float]] = []
    for tenure, renter_flag in (("Homeowner", 0.0), ("Renter", 1.0)):
        for income_level in income_quartiles:
            income_idx = int(round(income_level))
            income_idx = min(max(income_idx, 1), len(INCOME_ORDER))
            income_label = income_lookup.get(income_idx, str(income_idx))
            for attachment_label, attachment_val in attachment_levels:
                scenario = {
                    "tenure": tenure,
                    "income_bracket": income_label,
                    "attachment_level": attachment_label,
                    "renter_flag": renter_flag,
                    "income_ord": float(income_idx),
                    "attachment_factor": float(attachment_val),
                    **baseline,
                }
                rows.append(scenario)
    return pd.DataFrame(rows)


def build_displacement_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare tenure × damage severity scenarios for displacement models."""
    baseline = {
        "income_ord": df["income_ord"].median(),
        "employed": df["employed"].mean(),
        "age_years": df["age_years"].median(),
        "household_size": df["household_size"].median(),
        "attachment_factor": df["attachment_factor"].mean(),
        "mfr_flag": df["mfr_flag"].mean(),
        "coverage_peril_aligned": df["coverage_peril_aligned"].mean(),
        "aid_flag": df["aid_flag"].mean(),
        "days_since_event": df["days_since_event"].median(),
    }
    scenarios: List[Dict[str, float]] = []
    for tenure, renter_flag in (("Homeowner", 0.0), ("Renter", 1.0)):
        for damage in sorted(df["damage_ord"].dropna().unique()):
            scenario = {
                "tenure": tenure,
                "renter_flag": renter_flag,
                "damage_ord": float(damage),
                **baseline,
            }
            scenarios.append(scenario)
    return pd.DataFrame(scenarios)


def main(include_state_clusters: bool = False) -> None:
    df = prepare_features(load_base_dataset())

    insurance_features = [
        "renter_flag",
        "mfr_flag",
        "income_ord",
        "attachment_factor",
        "employed",
        "age_years",
        "household_size",
        "damage_ord",
    ]

    aid_features = [
        "renter_flag",
        "mfr_flag",
        "income_ord",
        "employed",
        "age_years",
        "household_size",
        "damage_ord",
        "attachment_factor",
        "days_since_event",
    ]

    displacement_features = [
        "renter_flag",
        "mfr_flag",
        "income_ord",
        "employed",
        "age_years",
        "household_size",
        "damage_ord",
        "attachment_factor",
        "days_since_event",
        "coverage_peril_aligned",
        "aid_flag",
    ]

    displacement_without_controls = [
        feature
        for feature in displacement_features
        if feature not in {"coverage_peril_aligned", "aid_flag"}
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fit_rows = []

    # Insurance model (hazard-clustered)
    ins_model = fit_logit(
        df, "coverage_peril_aligned", insurance_features, "hazard_cluster"
    )
    save_table(coef_table(ins_model), "insurance_logit_hazard_cluster.csv")
    save_table(odds_ratio_table(ins_model), "insurance_logit_hazard_cluster_odds.csv")
    fit_rows.append(model_fit_summary(ins_model, "insurance"))

    # Predicted probabilities
    insurance_scenarios = build_insurance_scenarios(df)
    insurance_preds = simulate_predictions(ins_model, insurance_scenarios)
    save_table(insurance_preds, "insurance_predicted_probabilities.csv")

    # Aid model
    aid_model = fit_logit(df, "aid_flag", aid_features, "hazard_cluster")
    save_table(coef_table(aid_model), "aid_logit_hazard_cluster.csv")
    save_table(odds_ratio_table(aid_model), "aid_logit_hazard_cluster_odds.csv")
    fit_rows.append(model_fit_summary(aid_model, "aid"))

    # Displacement with insurance/aid controls
    displacement_model = fit_logit(
        df, "displacement_flag", displacement_features, "hazard_cluster"
    )
    save_table(
        coef_table(displacement_model),
        "displacement_logit_with_controls_hazard_cluster.csv",
    )
    save_table(
        odds_ratio_table(displacement_model),
        "displacement_logit_with_controls_hazard_cluster_odds.csv",
    )
    fit_rows.append(model_fit_summary(displacement_model, "displacement_with_controls"))

    # Displacement without insurance/aid controls (total effect)
    displacement_total = fit_logit(
        df,
        "displacement_flag",
        displacement_without_controls,
        "hazard_cluster",
    )
    save_table(
        coef_table(displacement_total),
        "displacement_logit_without_controls_hazard_cluster.csv",
    )
    save_table(
        odds_ratio_table(displacement_total),
        "displacement_logit_without_controls_hazard_cluster_odds.csv",
    )
    fit_rows.append(model_fit_summary(displacement_total, "displacement_without_controls"))

    displacement_scenarios = build_displacement_scenarios(df)
    displacements_with = simulate_predictions(displacement_model, displacement_scenarios)
    displacements_with["model"] = "with_controls"
    displacements_without = simulate_predictions(displacement_total, displacement_scenarios)
    displacements_without["model"] = "without_controls"
    disp_predictions = pd.concat([displacements_with, displacements_without], ignore_index=True)
    save_table(disp_predictions, "displacement_predicted_probabilities.csv")

    model_fit = pd.concat(fit_rows, ignore_index=True)
    save_table(model_fit, "model_fit_summary.csv")

        # Hazard-type controls
    hazard_counts = df["hazard_category"].value_counts(dropna=False)
    small_hazards = hazard_counts[hazard_counts < 5].index
    df["hazard_category_collapsed"] = df["hazard_category"].fillna("Unknown")
    df["hazard_category_collapsed"] = df["hazard_category_collapsed"].where(
        ~df["hazard_category_collapsed"].isin(small_hazards),
        "Other (<=4 cases)",
    )
    hazard_dummies = pd.get_dummies(
        df["hazard_category_collapsed"], prefix="hazard", drop_first=True
    )
    df_hazard = pd.concat([df, hazard_dummies], axis=1)
    hazard_cols = hazard_dummies.columns.tolist()

    insurance_hazard_features = insurance_features + hazard_cols
    try:
        ins_hazard = fit_logit(
            df_hazard, "coverage_peril_aligned", insurance_hazard_features, "hazard_cluster"
        )
        save_table(coef_table(ins_hazard), 'insurance_logit_hazard_controls.csv')
    except (np.linalg.LinAlgError, ValueError) as exc:
        warnings.warn(f'Insurance hazard-control model skipped: {exc}')

    aid_hazard_features = aid_features + hazard_cols
    try:
        aid_hazard = fit_logit(df_hazard, 'aid_flag', aid_hazard_features, 'hazard_cluster')
        save_table(coef_table(aid_hazard), 'aid_logit_hazard_controls.csv')
    except (np.linalg.LinAlgError, ValueError) as exc:
        warnings.warn(f'Aid hazard-control model skipped: {exc}')

    displacement_hazard_features = displacement_features + hazard_cols
    try:
        disp_hazard = fit_logit(df_hazard, 'displacement_flag', displacement_hazard_features, 'hazard_cluster')
        save_table(coef_table(disp_hazard), 'displacement_logit_hazard_controls.csv')
    except (np.linalg.LinAlgError, ValueError) as exc:
        warnings.warn(f'Displacement hazard-control model skipped: {exc}')

# Interaction checks
    df["renter_x_attachment"] = df["renter_flag"] * df["attachment_factor"]
    insurance_interaction_features = insurance_features + ["renter_x_attachment"]
    ins_interaction = fit_logit(
        df, "coverage_peril_aligned", insurance_interaction_features, "hazard_cluster"
    )
    save_table(coef_table(ins_interaction), "insurance_logit_interaction_hazard_cluster.csv")

    df["renter_x_damage"] = df["renter_flag"] * df["damage_ord"]
    displacement_interaction_features = displacement_without_controls + ["renter_x_damage"]
    disp_interaction = fit_logit(
        df,
        "displacement_flag",
        displacement_interaction_features,
        "hazard_cluster",
    )
    save_table(coef_table(disp_interaction), "displacement_logit_interaction_hazard_cluster.csv")

    # Nonlinearity checks (quadratic terms)
    insurance_quadratic_features = insurance_features + ["age_centered_sq", "hhsize_centered_sq"]
    ins_quadratic = fit_logit(
        df, "coverage_peril_aligned", insurance_quadratic_features, "hazard_cluster"
    )
    save_table(coef_table(ins_quadratic), "insurance_logit_quadratic_hazard_cluster.csv")

    displacement_quadratic_features = displacement_without_controls + [
        "age_centered_sq",
        "hhsize_centered_sq",
    ]
    disp_quadratic = fit_logit(
        df,
        "displacement_flag",
        displacement_quadratic_features,
        "hazard_cluster",
    )
    save_table(coef_table(disp_quadratic), "displacement_logit_quadratic_hazard_cluster.csv")

    # Damage dummy specification for displacement
    damage_dummies = pd.get_dummies(df["damage_ord"], prefix="damage", drop_first=True)
    df_dummies = pd.concat([df, damage_dummies], axis=1)
    dummy_features = [
        "renter_flag",
        "mfr_flag",
        "income_ord",
        "employed",
        "age_years",
        "household_size",
        "attachment_factor",
        *damage_dummies.columns.tolist(),
    ]
    disp_dummy = fit_logit(df_dummies, "displacement_flag", dummy_features, "hazard_cluster")
    save_table(coef_table(disp_dummy), "displacement_logit_damage_dummies_hazard_cluster.csv")

    # Income nonlinearity checks (categorical)
    income_dummies = pd.get_dummies(df["income_ord"], prefix="income", drop_first=True)
    df_income = pd.concat([df, income_dummies], axis=1)
    income_cols = income_dummies.columns.tolist()
    insurance_income_features = [
        feature for feature in insurance_features if feature != "income_ord"
    ] + income_cols
    ins_income = fit_logit(
        df_income, "coverage_peril_aligned", insurance_income_features, "hazard_cluster"
    )
    save_table(coef_table(ins_income), "insurance_logit_income_dummies_hazard_cluster.csv")

    displacement_income_features = [
        feature for feature in displacement_features if feature != "income_ord"
    ] + income_cols
    disp_income = fit_logit(
        df_income, "displacement_flag", displacement_income_features, "hazard_cluster"
    )
    save_table(coef_table(disp_income), "displacement_logit_income_dummies_hazard_cluster.csv")

    # Demographic robustness checks
    demographic_controls = ["female", "nonwhite", "hispanic", "education_ord", "married"]
    ins_demo = fit_logit(
        df,
        "coverage_peril_aligned",
        insurance_features + demographic_controls,
        "hazard_cluster",
    )
    save_table(coef_table(ins_demo), "insurance_logit_demographics_hazard_cluster.csv")

    disp_demo = fit_logit(
        df,
        "displacement_flag",
        displacement_features + demographic_controls,
        "hazard_cluster",
    )
    save_table(coef_table(disp_demo), "displacement_logit_demographics_hazard_cluster.csv")

    if include_state_clusters:
        # Restrict to rows with state info for supplemental models
        state_df = df.dropna(subset=["state_cluster"]).copy()
        if not state_df.empty and state_df["state_cluster"].nunique() >= 2:
            ins_state = fit_logit(
                state_df, "coverage_peril_aligned", insurance_features, "state_cluster"
            )
            save_table(coef_table(ins_state), "insurance_logit_state_cluster.csv")
            disp_state = fit_logit(
                state_df,
                "displacement_flag",
                displacement_features,
                "state_cluster",
            )
            save_table(coef_table(disp_state), "displacement_logit_with_mediators_state_cluster.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-state-clusters",
        action="store_true",
        help="Also fit models clustered by state_abbr when available.",
    )
    args = parser.parse_args()
    main(include_state_clusters=args.include_state_clusters)
