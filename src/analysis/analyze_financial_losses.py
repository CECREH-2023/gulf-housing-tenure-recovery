#!/usr/bin/env python3
"""Generate financial loss distribution diagnostics with trimming/winsorization scenarios.

This module is integrated with CENTAUR infrastructure for path management.

Usage:
    python -m src.analysis.analyze_financial_losses
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config.settings import get_settings

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
settings = get_settings()

DATA_PATH = settings.processed_dir / "regression" / "recovery_model_dataset.csv"
RESULTS_DIR = settings.project_root / "results" / "loss"

LOSS_COL = "2.9_1"  # damages to home
TENURE_COL = "htype"
ID_COL = "ResponseId"

QUANTILES: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
TIMING_MODE = os.getenv("TIMING_MODE", "optionB")
TIMING_COLUMN = (
    "days_since_event_optionA"
    if TIMING_MODE == "optionA"
    else "days_since_event_optionB"
)


def load_losses() -> pd.DataFrame:
    usecols = [
        ID_COL,
        TENURE_COL,
        LOSS_COL,
        "days_since_event_optionA",
        "days_since_event_optionB",
        "time_since_event_bin",
    ]
    df = pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)
    df[LOSS_COL] = pd.to_numeric(df[LOSS_COL], errors="coerce")
    df = df.dropna(subset=[TENURE_COL, LOSS_COL])
    df = df[df[LOSS_COL] >= 0]
    df = df.rename(columns={LOSS_COL: "loss_home"}).reset_index(drop=True)
    return df


def prepare_timing_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    if TIMING_MODE == "optionC":
        midpoint_map = {
            "0-3 months": 1.5,
            "3-6 months": 4.5,
            "6-9 months": 7.5,
            "9-12 months": 10.5,
            ">12 months": 13.5,
            "Unknown": np.nan,
        }
        df["time_bin_midpoint"] = df["time_since_event_bin"].map(midpoint_map)
        median_midpoint = df["time_bin_midpoint"].median()
        df["time_bin_midpoint"] = df["time_bin_midpoint"].fillna(median_midpoint)
        df["time_bin_unknown"] = (df["time_since_event_bin"] == "Unknown").astype(float)
        timing_features = ["time_bin_midpoint", "time_bin_unknown"]
    else:
        df[TIMING_COLUMN] = pd.to_numeric(df[TIMING_COLUMN], errors="coerce")
        timing_features = [TIMING_COLUMN]
    timing_reference = {feat: df[feat].mean() for feat in timing_features if feat in df}
    return df, timing_features, timing_reference


def apply_trim(df: pd.DataFrame, upper_quantile: float) -> pd.DataFrame:
    threshold = df["loss_home"].quantile(upper_quantile)
    return df[df["loss_home"] <= threshold].copy()


def apply_winsor(df: pd.DataFrame, upper_quantile: float) -> pd.DataFrame:
    threshold = df["loss_home"].quantile(upper_quantile)
    capped = df.copy()
    capped.loc[capped["loss_home"] > threshold, "loss_home"] = threshold
    return capped


def summarize_by_tenure(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for tenure, subset in df.groupby(TENURE_COL):
        stats = {
            "scenario": scenario,
            "htype": tenure,
            "n": len(subset),
            "mean": subset["loss_home"].mean(),
            "median": subset["loss_home"].median(),
            "p90": subset["loss_home"].quantile(0.90),
            "p95": subset["loss_home"].quantile(0.95),
            "p99": subset["loss_home"].quantile(0.99),
            "max": subset["loss_home"].max(),
        }
        for q in QUANTILES:
            stats[f"p{int(q*100)}"] = subset["loss_home"].quantile(q)
        records.append(stats)
    return pd.DataFrame.from_records(records)


def summarize_overall(df: pd.DataFrame, scenario: str) -> Dict[str, float]:
    stats = {
        "scenario": scenario,
        "n": len(df),
        "mean": df["loss_home"].mean(),
        "median": df["loss_home"].median(),
        "p90": df["loss_home"].quantile(0.90),
        "p95": df["loss_home"].quantile(0.95),
        "p99": df["loss_home"].quantile(0.99),
        "max": df["loss_home"].max(),
        "total_loss": df["loss_home"].sum(),
    }
    for q in QUANTILES:
        stats[f"p{int(q*100)}"] = df["loss_home"].quantile(q)
    return stats


def pareto_tail(df: pd.DataFrame, cutoff: float = 0.9) -> pd.DataFrame:
    threshold = df["loss_home"].quantile(cutoff)
    tail = df[df["loss_home"] >= threshold].copy()
    tail = tail.sort_values("loss_home", ascending=False).reset_index(drop=True)
    tail["rank"] = tail.index + 1
    tail["ccdf"] = (len(tail) - tail["rank"] + 1) / len(df)
    tail["cdf"] = 1 - tail["ccdf"]
    return tail[[ID_COL, TENURE_COL, "loss_home", "rank", "cdf", "ccdf"]]


def two_part_model(
    df: pd.DataFrame,
    timing_features: List[str],
    timing_reference: Dict[str, float],
    boot_iters: int = 1000,
) -> pd.DataFrame:
    working = (
        df.assign(
            renter=lambda d: np.where(d[TENURE_COL] == "Renter", 1, 0),
            positive=lambda d: np.where(d["loss_home"] > 0, 1, 0)
        )
        .dropna(subset=timing_features)
    )
    if timing_features:
        timing_terms = " + ".join(timing_features)
        logit_formula = f"positive ~ renter + {timing_terms}"
        ols_formula = f"log_loss ~ renter + {timing_terms}"
    else:
        logit_formula = "positive ~ renter"
        ols_formula = "log_loss ~ renter"

    logit = smf.logit(logit_formula, data=working).fit(disp=0)
    ols_data = working[working["loss_home"] > 0].copy()
    ols_data["log_loss"] = np.log(ols_data["loss_home"] + 1e-9)
    ols = smf.ols(ols_formula, data=ols_data).fit()
    sigma2 = ols.mse_resid

    def expected_loss(renter_flag: int) -> float:
        base = {"renter": [renter_flag]}
        for feat in timing_features:
            base[feat] = [timing_reference.get(feat, 0.0)]
        base_df = pd.DataFrame(base)
        p = logit.predict(base_df)[0]
        mean_log = ols.predict(base_df)[0]
        return float(p * np.exp(mean_log + 0.5 * sigma2))

    renter_expected = expected_loss(1)
    owner_expected = expected_loss(0)

    diffs = []
    renter_boot = []
    owner_boot = []
    rng = np.random.default_rng(20251026)
    for _ in range(boot_iters):
        sample = working.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        try:
            logit_b = smf.logit(logit_formula, data=sample).fit(disp=0)
        except Exception:
            continue
        ols_sample = sample[sample["loss_home"] > 0].copy()
        ols_sample["log_loss"] = np.log(ols_sample["loss_home"] + 1e-9)
        try:
            ols_b = smf.ols(ols_formula, data=ols_sample).fit()
        except Exception:
            continue
        sigma2_b = ols_b.mse_resid

        def exp_loss_boot(r_flag: int) -> float:
            base = {"renter": [r_flag]}
            for feat in timing_features:
                base[feat] = [timing_reference.get(feat, 0.0)]
            base_df = pd.DataFrame(base)
            p = logit_b.predict(base_df)[0]
            mean_log = ols_b.predict(base_df)[0]
            return float(p * np.exp(mean_log + 0.5 * sigma2_b))

        renter_val = exp_loss_boot(1)
        owner_val = exp_loss_boot(0)
        renter_boot.append(renter_val)
        owner_boot.append(owner_val)
        diffs.append(renter_val - owner_val)

    diff_array = np.array(diffs)
    renter_array = np.array(renter_boot)
    owner_array = np.array(owner_boot)
    return pd.DataFrame(
        {
            "renter_expected_loss": [renter_expected],
            "owner_expected_loss": [owner_expected],
            "renter_ci_low": [
                np.quantile(renter_array, 0.025) if len(renter_array) > 0 else np.nan
            ],
            "renter_ci_high": [
                np.quantile(renter_array, 0.975) if len(renter_array) > 0 else np.nan
            ],
            "owner_ci_low": [
                np.quantile(owner_array, 0.025) if len(owner_array) > 0 else np.nan
            ],
            "owner_ci_high": [
                np.quantile(owner_array, 0.975) if len(owner_array) > 0 else np.nan
            ],
            "difference": [renter_expected - owner_expected],
            "diff_ci_low": [np.quantile(diff_array, 0.025) if len(diff_array) > 0 else np.nan],
            "diff_ci_high": [np.quantile(diff_array, 0.975) if len(diff_array) > 0 else np.nan],
            "boot_iterations": [len(diff_array)],
        }
    )


def quantile_regression(df: pd.DataFrame, quantiles: Iterable[float]) -> pd.DataFrame:
    working = df.assign(renter=lambda d: np.where(d[TENURE_COL] == "Renter", 1, 0))
    records: List[Dict[str, float]] = []
    for q in quantiles:
        model = smf.quantreg("loss_home ~ renter", data=working).fit(q=q)
        coef = model.params["renter"]
        conf_int = model.conf_int().loc["renter"]
        records.append(
            {
                "quantile": q,
                "coef": coef,
                "ci_low": conf_int[0],
                "ci_high": conf_int[1],
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = load_losses()
    base, timing_features, timing_reference = prepare_timing_features(base)

    scenarios = {
        "baseline": base,
        "trim_99_5": apply_trim(base, 0.995),
        "winsor_99_5": apply_winsor(base, 0.995),
    }

    tenure_tables = []
    overall_records = []
    for name, frame in scenarios.items():
        tenure_tables.append(summarize_by_tenure(frame, name))
        overall_records.append(summarize_overall(frame, name))

    tenure_summary = pd.concat(tenure_tables, ignore_index=True)
    tenure_summary.to_csv(RESULTS_DIR / "loss_quantiles_by_tenure.csv", index=False)

    pd.DataFrame(overall_records).to_csv(RESULTS_DIR / "loss_overall_summary.csv", index=False)

    pareto = pareto_tail(base, cutoff=0.9)
    pareto.to_csv(RESULTS_DIR / "loss_pareto_tail.csv", index=False)

    two_part = two_part_model(base, timing_features, timing_reference)
    two_part.to_csv(RESULTS_DIR / "loss_two_part_model.csv", index=False)

    quantiles = quantile_regression(base, quantiles=(0.5, 0.75))
    quantiles.to_csv(RESULTS_DIR / "loss_quantile_regression.csv", index=False)

    metadata = {
        "source_column": LOSS_COL,
        "tenure_column": TENURE_COL,
        "quantiles": QUANTILES,
        "scenarios": {
            "baseline": "Raw self-reported home damage losses",
            "trim_99_5": "Observations with loss_home above the 99.5th percentile removed",
            "winsor_99_5": "Observations above the 99.5th percentile capped at that value",
        },
    }
    (RESULTS_DIR / "loss_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
