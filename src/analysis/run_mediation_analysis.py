#!/usr/bin/env python3
"""Estimate natural direct and indirect effects for the Gulf Interview Project.

This module is integrated with CENTAUR infrastructure for path management.

Usage:
    python -m src.analysis.run_mediation_analysis
    python -m src.analysis.run_mediation_analysis --bootstrap-samples 1000
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError

from config.settings import get_settings
from src.analysis.event_timing import add_event_timing

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
settings = get_settings()

REGRESSION_PATH = settings.processed_dir / "regression" / "regression_dataset_with_bert_themes.csv"
FACTOR_PATH = (
    settings.processed_dir
    / "factor_analysis"
    / "polychoric_fa"
    / "promax"
    / "factor_scores_with_ids.csv"
)
RESULTS_DIR = settings.project_root / "results" / "mediation"

SURVEY_YEAR = 2021
TIMING_MODE = os.getenv("TIMING_MODE", "optionB")
TIMING_COLUMN = (
    "days_since_event_optionA"
    if TIMING_MODE == "optionA"
    else "days_since_event_optionB"
)


INCOME_MAP: Mapping[str, int] = {
    "Less than $10,000": 1,
    "$10,000 to $19,999": 2,
    "$20,000 to $29,999": 3,
    "$30,000 to $39,999": 4,
    "$40,000 to $49,999": 5,
    "$50,000 to $59,999": 6,
    "$60,000 to $69,999": 7,
    "$70,000 to $79,999": 8,
    "$80,000 to $89,999": 9,
    "$90,000 to $99,999": 10,
    "$100,000 to $149,999": 11,
    "$150,000 or more": 12,
}

EDUCATION_MAP: Mapping[str, int] = {
    "Less than high school degree": 1,
    "High school graduate (high school diploma or equivalent including GED)": 2,
    "Some college but no degree": 3,
    "Associate degree in college (2-year)": 4,
    "Bachelor's degree in college (4-year)": 5,
    "Master's degree": 6,
    "Professional degree (JD, MD)": 7,
    "Doctoral degree": 8,
}

DAMAGE_MAP: Mapping[str, int] = {
    "No damage": 0,
    "Not very extensive": 1,
    "Somewhat extensive": 2,
    "Very extensive": 3,
}


MEDIATOR_COVARIATES_BASE: List[str] = [
    "attachment_factor",
    "income_ord",
    "age_years",
    "household_size",
    "damage_ord",
    "employed",
    "female",
    "nonwhite",
    "hispanic",
    "education_ord",
    "married",
    "mfr_flag",
]


@dataclass
class MediationEffects:
    """Container for mediation effects on the probability and odds-ratio scales."""

    ey_11: float
    ey_00: float
    ey_10: float
    ey_01: float
    te_or: float
    nde_or: float
    nie_or: float
    te_rd: float
    nde_rd: float
    nie_rd: float
    prop_mediated_or: float

    def as_summary_frame(self) -> pd.DataFrame:
        """Return tidy summary with odds ratios and risk differences."""
        return pd.DataFrame(
            [
                {
                    "effect": "Total Effect",
                    "odds_ratio": self.te_or,
                    "risk_difference": self.te_rd,
                },
                {
                    "effect": "Natural Direct Effect",
                    "odds_ratio": self.nde_or,
                    "risk_difference": self.nde_rd,
                },
                {
                    "effect": "Natural Indirect Effect",
                    "odds_ratio": self.nie_or,
                    "risk_difference": self.nie_rd,
                },
            ]
        )


def _ordered_map(values: Iterable[str]) -> Dict[str, int]:
    return {value: idx + 1 for idx, value in enumerate(values)}


def load_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """Load survey and factor data with harmonized covariates."""
    base = add_event_timing(pd.read_csv(REGRESSION_PATH, low_memory=False))
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
        base.merge(factors, on="ResponseId", how="left")
        .rename(columns={"ResponseId": "response_id"})
        .assign(
            renter_flag=lambda d: (d["htype"] == "Renter").astype(float),
            aid_flag=lambda d: (d["2.12"] == "Yes").astype(float),
            insurance_yes=lambda d: pd.to_numeric(d["insurance_yes"], errors="coerce"),
            displacement_flag=lambda d: pd.to_numeric(
                d["displaced_event_flag"], errors="coerce"
            ),
            income_ord=lambda d: d["1.9"].map(INCOME_MAP),
            employed=lambda d: (d["1.8"] == "Employed").astype(float),
            age_years=lambda d: SURVEY_YEAR - pd.to_numeric(d["age"], errors="coerce"),
            household_size=lambda d: pd.to_numeric(d["hsize"], errors="coerce"),
            damage_ord=lambda d: d["2.8_1"].map(DAMAGE_MAP),
            female=lambda d: (d["gender"] == "Woman").astype(float),
            nonwhite=lambda d: (
                ~d["race"].fillna("").str.contains("White", case=False)
            ).astype(float),
            hispanic=lambda d: d["race"]
            .fillna("")
            .str.contains("Hispanic", case=False)
            .astype(float),
            education_ord=lambda d: d["education"].map(EDUCATION_MAP),
            married=lambda d: (d["marital"] == "Married").astype(float),
            mfr_flag=lambda d: d["structure_category"]
            .str.contains("Multifamily", case=False, na=False)
            .astype(float),
            attachment_factor=lambda d: pd.to_numeric(
                d["attachment_belonging"], errors="coerce"
            ),
            days_since_event_optionA=lambda d: pd.to_numeric(
                d.get("days_since_event_optionA"), errors="coerce"
            ),
            days_since_event_optionB=lambda d: pd.to_numeric(
                d.get("days_since_event_optionB"), errors="coerce"
            ),
            time_since_event_bin=lambda d: d.get("time_since_event_bin"),
        )
    )

    df, timing_features = add_timing_features(df)

    mediator_covariates = MEDIATOR_COVARIATES_BASE + timing_features
    keep_cols = [
        "response_id",
        "renter_flag",
        "aid_flag",
        "insurance_yes",
        "displacement_flag",
        *mediator_covariates,
    ]
    clean = df.dropna(subset=keep_cols).reset_index(drop=True)
    return clean, mediator_covariates


def add_timing_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if TIMING_MODE == "optionC":
        categories = [
            "0-3 months",
            "3-6 months",
            "6-9 months",
            "9-12 months",
            ">12 months",
            "Unknown",
        ]
        cat = pd.Categorical(df["time_since_event_bin"], categories=categories)
        dummies = pd.get_dummies(cat, prefix="time_bin", drop_first=True)
        dummies = dummies.loc[:, (dummies != 0).any(axis=0)]
        if dummies.empty:
            return df, []
        clean_cols = [
            col.replace(" ", "_").replace("-", "_").replace(">", "gt").replace("<", "lt")
            for col in dummies.columns
        ]
        dummies.columns = clean_cols
        df = pd.concat([df, dummies], axis=1)
        timing_features = clean_cols
        for col in timing_features:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        return df, timing_features

    df[TIMING_COLUMN] = pd.to_numeric(df[TIMING_COLUMN], errors="coerce")
    return df, [TIMING_COLUMN]


def fit_logit(dep: str, features: List[str], data: pd.DataFrame) -> sm.Logit:
    """Fit a logistic regression with statsmodels."""
    exog = sm.add_constant(data[features], has_constant="add")
    model = sm.Logit(data[dep], exog)
    result = model.fit(disp=False, method="newton", maxiter=200)
    if not result.mle_retvals.get("converged", True):
        raise RuntimeError(f"{dep} model failed to converge.")
    return result


def _predict(
    result: sm.Logit,
    base: pd.DataFrame,
    features: List[str],
    overrides: Mapping[str, float],
) -> np.ndarray:
    """Predict probabilities for a scenario."""
    exog = base[features].copy()
    for name, value in overrides.items():
        if name not in exog.columns:
            raise KeyError(f"{name} not in feature set {features}")
        exog[name] = value
    exog = sm.add_constant(exog, has_constant="add")
    exog = exog[result.model.exog_names]
    return result.model.predict(result.params, exog)


def compute_effects(
    sample: pd.DataFrame,
    ins_model: sm.Logit,
    aid_model: sm.Logit,
    disp_model: sm.Logit,
    covariates: List[str],
) -> MediationEffects:
    """G-computation for natural effects with two binary mediators."""
    ins_features = ["renter_flag", *covariates]
    aid_features = ["renter_flag", "insurance_yes", *covariates]
    outcome_features = ["renter_flag", "insurance_yes", "aid_flag", *covariates]

    base_ins = sample[ins_features].copy()
    base_aid = sample[aid_features].copy()
    base_out = sample[outcome_features].copy()

    p_m1: Dict[int, np.ndarray] = {
        t: _predict(ins_model, base_ins, ins_features, {"renter_flag": float(t)})
        for t in (0, 1)
    }

    p_m2: Dict[Tuple[int, int], np.ndarray] = {}
    for t in (0, 1):
        for m1 in (0, 1):
            p_m2[(t, m1)] = _predict(
                aid_model,
                base_aid,
                aid_features,
                {"renter_flag": float(t), "insurance_yes": float(m1)},
            )

    p_y: Dict[Tuple[int, int, int], np.ndarray] = {}
    for t in (0, 1):
        for m1 in (0, 1):
            for m2 in (0, 1):
                p_y[(t, m1, m2)] = _predict(
                    disp_model,
                    base_out,
                    outcome_features,
                    {
                        "renter_flag": float(t),
                        "insurance_yes": float(m1),
                        "aid_flag": float(m2),
                    },
                )

    n = len(sample)

    def expected_prob(t_out: int, t_med: int) -> np.ndarray:
        total = np.zeros(n)
        p_m1_1 = p_m1[t_med]
        p_m1_0 = 1.0 - p_m1_1
        for m1 in (0, 1):
            p_m1_prob = p_m1_1 if m1 == 1 else p_m1_0
            p_m2_1 = p_m2[(t_med, m1)]
            p_m2_0 = 1.0 - p_m2_1
            for m2 in (0, 1):
                p_m2_prob = p_m2_1 if m2 == 1 else p_m2_0
                total += p_m1_prob * p_m2_prob * p_y[(t_out, m1, m2)]
        return total

    ey_11 = expected_prob(1, 1).mean()
    ey_00 = expected_prob(0, 0).mean()
    ey_10 = expected_prob(1, 0).mean()
    ey_01 = expected_prob(0, 1).mean()

    odds = lambda p: np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9)
    te_or = float(odds(ey_11) / odds(ey_00))
    nde_or = float(odds(ey_10) / odds(ey_00))
    nie_or = float(odds(ey_01) / odds(ey_00))

    te_rd = float(ey_11 - ey_00)
    nde_rd = float(ey_10 - ey_00)
    nie_rd = float(ey_01 - ey_00)

    with np.errstate(divide="ignore", invalid="ignore"):
        prop_mediated = float(np.log(nie_or) / np.log(te_or)) if te_or != 1 else np.nan

    return MediationEffects(
        ey_11=ey_11,
        ey_00=ey_00,
        ey_10=ey_10,
        ey_01=ey_01,
        te_or=te_or,
        nde_or=nde_or,
        nie_or=nie_or,
        te_rd=te_rd,
        nde_rd=nde_rd,
        nie_rd=nie_rd,
        prop_mediated_or=prop_mediated,
    )


def nie_sensitivity_analysis(
    sample: pd.DataFrame,
    ins_model: sm.Logit,
    aid_model: sm.Logit,
    disp_model: sm.Logit,
    covariates: List[str],
    scales: Iterable[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
) -> pd.DataFrame:
    """Explore NIE robustness by scaling the aid→displacement coefficient."""
    base_coef = float(disp_model.params["aid_flag"])
    records: List[Dict[str, float]] = []
    for scale in scales:
        adjusted = copy.deepcopy(disp_model)
        adjusted.params = adjusted.params.copy()
        adjusted.params["aid_flag"] = base_coef * scale
        effects = compute_effects(sample, ins_model, aid_model, adjusted, covariates)
        records.append(
            {
                "aid_scale": float(scale),
                "aid_coef": float(base_coef * scale),
                "nie_or": effects.nie_or,
                "nie_rd": effects.nie_rd,
                "nde_or": effects.nde_or,
                "te_or": effects.te_or,
            }
        )
    return pd.DataFrame.from_records(records)


def compute_evalue(or_value: float) -> float:
    """Compute VanderWeele-style E-value for an odds (risk) ratio."""
    if or_value <= 1:
        or_value = 1 / max(or_value, 1e-12)
    return float(or_value + np.sqrt(or_value * (or_value - 1)))


def run_models(data: pd.DataFrame, covariates: List[str]) -> Tuple[sm.Logit, sm.Logit, sm.Logit, sm.Logit]:
    """Fit mediator, outcome, and total-effect models."""
    ins_model = fit_logit("insurance_yes", ["renter_flag", *covariates], data)
    aid_model = fit_logit("aid_flag", ["renter_flag", "insurance_yes", *covariates], data)
    disp_model = fit_logit(
        "displacement_flag", ["renter_flag", "insurance_yes", "aid_flag", *covariates], data
    )
    total_model = fit_logit(
        "displacement_flag", ["renter_flag", *covariates], data
    )
    return ins_model, aid_model, disp_model, total_model


def bootstrap_effects(
    data: pd.DataFrame,
    base_models: Tuple[sm.Logit, sm.Logit, sm.Logit],
    covariates: List[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap natural effect estimates."""
    rng = np.random.default_rng(seed)
    ins_model, aid_model, disp_model = base_models
    records: List[Dict[str, float]] = []

    for idx in range(n_boot):
        sample_idx = rng.integers(0, len(data), len(data))
        sample = data.iloc[sample_idx].reset_index(drop=True)
        try:
            ins_fit = fit_logit("insurance_yes", ["renter_flag", *covariates], sample)
            aid_fit = fit_logit(
                "aid_flag", ["renter_flag", "insurance_yes", *covariates], sample
            )
            disp_fit = fit_logit(
                "displacement_flag",
                ["renter_flag", "insurance_yes", "aid_flag", *covariates],
                sample,
            )
        except (PerfectSeparationError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            continue
        effects = compute_effects(sample, ins_fit, aid_fit, disp_fit, covariates)
        records.append(
            {
                "te_or": effects.te_or,
                "nde_or": effects.nde_or,
                "nie_or": effects.nie_or,
                "te_rd": effects.te_rd,
                "nde_rd": effects.nde_rd,
                "nie_rd": effects.nie_rd,
                "prop_mediated_or": effects.prop_mediated_or,
            }
        )

    if not records:
        raise RuntimeError("No successful bootstrap replications; results unavailable.")

    return pd.DataFrame.from_records(records)


def summarize_with_ci(point: MediationEffects, boot_df: pd.DataFrame | None) -> pd.DataFrame:
    """Combine point estimates with bootstrapped intervals (if available)."""
    summary = point.as_summary_frame()
    if boot_df is None:
        return summary

    ci_table = []
    for effect_key, column in [
        ("Total Effect", "te_or"),
        ("Natural Direct Effect", "nde_or"),
        ("Natural Indirect Effect", "nie_or"),
    ]:
        ci_or = np.percentile(boot_df[column], [2.5, 97.5])
        ci_rd = np.percentile(
            boot_df[column.replace("_or", "_rd")], [2.5, 97.5]
        )
        ci_table.append(
            {
                "effect": effect_key,
                "or_ci_lower": float(ci_or[0]),
                "or_ci_upper": float(ci_or[1]),
                "rd_ci_lower": float(ci_rd[0]),
                "rd_ci_upper": float(ci_rd[1]),
            }
        )

    ci_frame = pd.DataFrame(ci_table)
    return summary.merge(ci_frame, on="effect", how="left")


def write_outputs(
    total_summary: pd.DataFrame,
    appendix_summary: pd.DataFrame,
    point: MediationEffects,
    total_model: sm.Logit,
    boot_df: pd.DataFrame | None,
    sensitivity: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Persist results to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    main_path = RESULTS_DIR / "mediation_total_effect.csv"
    total_summary.to_csv(main_path, index=False)
    appendix_path = RESULTS_DIR / "mediation_appendix_effects.csv"
    appendix_summary.to_csv(appendix_path, index=False)
    if not sensitivity.empty:
        sensitivity_path = RESULTS_DIR / "mediation_nie_sensitivity.csv"
        sensitivity.to_csv(sensitivity_path, index=False)

    metadata = {
        "n_observations": int(total_model.nobs),
        "te_logit_coef": float(total_model.params["renter_flag"]),
        "te_logit_or": float(np.exp(total_model.params["renter_flag"])),
        "proportion_mediated_or": float(point.prop_mediated_or),
        "bootstrap_samples": int(args.bootstrap_samples or 0),
        "bootstrap_completed": int(0 if boot_df is None else len(boot_df)),
        "nie_sensitivity_scales": sensitivity["aid_scale"].tolist() if not sensitivity.empty else [],
    }

    metadata_path = RESULTS_DIR / "mediation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Number of bootstrap replications for uncertainty estimates.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2024,
        help="Seed for bootstrap replicates.",
    )
    args = parser.parse_args()

    data, mediator_covariates = load_dataset()
    ins_model, aid_model, disp_model, total_model = run_models(data, mediator_covariates)

    point_effects = compute_effects(
        data, ins_model, aid_model, disp_model, mediator_covariates
    )

    boot_results = None
    if args.bootstrap_samples:
        boot_results = bootstrap_effects(
            data,
            (ins_model, aid_model, disp_model),
            mediator_covariates,
            n_boot=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )

    summary = summarize_with_ci(point_effects, boot_results)
    total_summary = summary.loc[summary["effect"] == "Total Effect"].reset_index(drop=True)
    appendix_summary = summary.loc[summary["effect"] != "Total Effect"].reset_index(drop=True)
    sensitivity_df = nie_sensitivity_analysis(
        data,
        ins_model,
        aid_model,
        disp_model,
        mediator_covariates,
    )

    write_outputs(total_summary, appendix_summary, point_effects, total_model, boot_results, sensitivity_df, args)

    print(total_summary.to_string(index=False))
    if not appendix_summary.empty:
        print("\nExploratory mediation components (appendix only):")
        print(appendix_summary.to_string(index=False))
    print("\nNIE sensitivity to aid→displacement effect scaling:")
    print(sensitivity_df.to_string(index=False))
    print(f"\nProportion mediated (log-odds scale): {point_effects.prop_mediated_or:.3f}")
    print(f"Total effect (logit OR) from model without mediators: {np.exp(total_model.params['renter_flag']):.3f}")

    # Sensitivity (E-values) for quick reference.
    evalue_rows = []
    for effect_name, or_value in [
        ("Total Effect", point_effects.te_or),
        ("Natural Direct Effect", point_effects.nde_or),
    ]:
        row = {
            "effect": effect_name,
            "odds_ratio": or_value,
            "e_value_point": compute_evalue(or_value),
        }
        if boot_results is not None:
            ci_col = {
                "Total Effect": "te_or",
                "Natural Direct Effect": "nde_or",
            }[effect_name]
            lower = float(np.percentile(boot_results[ci_col], 2.5))
            row["e_value_ci_lower"] = compute_evalue(lower)
        evalue_rows.append(row)

    evalue_path = RESULTS_DIR / "mediation_sensitivity.csv"
    pd.DataFrame(evalue_rows).to_csv(evalue_path, index=False)


if __name__ == "__main__":
    # Silence convergence warnings that are retried via bootstrap.
    with np.errstate(over="ignore", invalid="ignore"):
        import warnings

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        main()
