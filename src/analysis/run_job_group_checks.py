#!/usr/bin/env python3
"""Generate job-group comparison summaries with multiple-comparison control.

This module is integrated with CENTAUR infrastructure for path management.

Usage:
    python -m src.analysis.run_job_group_checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import get_settings

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
settings = get_settings()

PROCESSED_DIR = settings.processed_dir
REPORT_DIR = settings.project_root / "reports" / "tables"

FACTOR_PRE = PROCESSED_DIR / "factor_scores_bcfa.csv"
FACTOR_POST = PROCESSED_DIR / "factor_scores_bcfa_post.csv"
IDS_PRE = PROCESSED_DIR / "short_term_pre_with_ids.csv"
IDS_POST = PROCESSED_DIR / "short_term_post_with_ids.csv"
REGRESSION_DATA = PROCESSED_DIR / "regression_dataset.csv"

SHORT_TERM_COLUMNS: Sequence[str] = (
    "Communication",
    "Food",
    "Water",
    "Shelter",
    "Lifelines",
    "Job",
    "Health",
    "Social_capital",
    "Financial_support",
)

GROUP_LABELS: Dict[int, str] = {1: "Researcher", 2: "Practitioner", 3: "Other"}


@dataclass
class WelchResult:
    group1: str
    group2: str
    column: str
    mean_group1: float
    mean_group2: float
    difference: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    cohen_d: float
    n_group1: int
    n_group2: int
    label: str

    def to_dict(self) -> Dict[str, float | int | str]:
        return {
            "group1": self.group1,
            "group2": self.group2,
            "column": self.column,
            "label": self.label,
            "mean_group1": self.mean_group1,
            "mean_group2": self.mean_group2,
            "difference": self.difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "cohen_d": self.cohen_d,
            "n_group1": self.n_group1,
            "n_group2": self.n_group2,
        }


def benjamini_hochberg(p_values: Iterable[float]) -> List[float]:
    """Return Benjamini–Hochberg FDR-adjusted *p* values."""
    p_values = np.asarray(list(p_values), dtype=float)
    n = p_values.size
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted = np.empty_like(sorted_p)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = sorted_p[i] * n / rank
        prev = min(prev, adj)
        adjusted[i] = prev
    full = np.empty_like(p_values)
    full[order] = adjusted
    return full.clip(max=1.0).tolist()


def load_group_assignments() -> pd.Series:
    """Return respondent IDs mapped to the combined job labels."""
    data = pd.read_csv(REGRESSION_DATA, usecols=["response_id", "job_type"])
    series = data.set_index("response_id")["job_type"].map(GROUP_LABELS)
    return series.rename("job_group")


def build_factor_panel(groups: pd.Series) -> pd.DataFrame:
    """Return BCFA factor scores (pre/post/diff) with job groups."""
    pre_scores = pd.read_csv(FACTOR_PRE)
    post_scores = pd.read_csv(FACTOR_POST)
    pre_ids = pd.read_csv(IDS_PRE, usecols=["response_id"])
    post_ids = pd.read_csv(IDS_POST, usecols=["response_id"])

    pre = pd.concat([pre_ids, pre_scores.add_suffix("_pre")], axis=1)
    post = pd.concat([post_ids, post_scores.add_suffix("_post")], axis=1)

    panel = pre.merge(post, on="response_id")
    for factor in ("basic", "community", "financial"):
        panel[f"{factor}_diff"] = panel[f"{factor}_post"] - panel[f"{factor}_pre"]
    panel["job_group"] = panel["response_id"].map(groups)
    return panel


def build_item_panel(groups: pd.Series) -> pd.DataFrame:
    """Return 0–5 category ratings in pre/post/diff form."""
    pre = pd.read_csv(IDS_PRE, usecols=["response_id", *SHORT_TERM_COLUMNS]).rename(
        columns={col: f"{col}_pre" for col in SHORT_TERM_COLUMNS}
    )
    post = pd.read_csv(IDS_POST, usecols=["response_id", *SHORT_TERM_COLUMNS]).add_suffix("_post")
    panel = pre.merge(post, left_on="response_id", right_on="response_id_post", how="inner")
    panel = panel.drop(columns="response_id_post")
    for col in SHORT_TERM_COLUMNS:
        panel[f"{col}_diff"] = panel[f"{col}_post"] - panel[f"{col}_pre"]
    panel["job_group"] = panel["response_id"].map(groups)
    return panel


def welch_test(frame: pd.DataFrame, column: str, group1: str, group2: str) -> WelchResult | None:
    """Run a Welch t-test between two job groups."""
    subset = frame[["job_group", column]].dropna()
    grp1 = subset[subset["job_group"] == group1][column].astype(float)
    grp2 = subset[subset["job_group"] == group2][column].astype(float)
    if grp1.empty or grp2.empty:
        return None

    mean1, mean2 = grp1.mean(), grp2.mean()
    diff = mean1 - mean2
    t_stat, p_value = stats.ttest_ind(grp1, grp2, equal_var=False)

    var1, var2 = grp1.var(ddof=1), grp2.var(ddof=1)
    n1, n2 = grp1.shape[0], grp2.shape[0]
    pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2) if (n1 + n2) > 2 else np.nan
    d = diff / np.sqrt(pooled) if pooled > 0 else float("nan")

    se = np.sqrt(var1 / n1 + var2 / n2)
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
    df = df_num / df_den if df_den > 0 else n1 + n2 - 2
    margin = stats.t.ppf(0.975, df) * se

    return WelchResult(
        group1=group1,
        group2=group2,
        column=column,
        label="",
        mean_group1=float(mean1),
        mean_group2=float(mean2),
        difference=float(diff),
        ci_low=float(diff - margin),
        ci_high=float(diff + margin),
        t_stat=float(t_stat),
        p_value=float(p_value),
        cohen_d=float(d),
        n_group1=int(n1),
        n_group2=int(n2),
    )


def add_labels(results: List[WelchResult], mapping: Dict[str, str]) -> None:
    for result in results:
        result.label = mapping[result.column]


def attach_fdr(results: List[WelchResult]) -> List[Dict[str, object]]:
    """Attach BH-adjusted p-values grouped by (group1, group2)."""
    grouped: Dict[tuple[str, str], List[int]] = {}
    for idx, res in enumerate(results):
        grouped.setdefault((res.group1, res.group2), []).append(idx)

    rows = [res.to_dict() for res in results]
    for pair, indices in grouped.items():
        p_vals = [results[idx].p_value for idx in indices]
        adj = benjamini_hochberg(p_vals)
        for idx, p_adj in zip(indices, adj):
            rows[idx]["p_fdr"] = float(p_adj)
    return rows


def summarise_factors(panel: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
    """Return mean/SD summary for each job group."""
    outputs: Dict[str, List[Dict[str, object]]] = {}
    factors = ("basic", "community", "financial")
    stages = {
        "Pre (BCFA)": (lambda f: f"{f}_pre"),
        "Post (BCFA)": (lambda f: f"{f}_post"),
        "Change (Post-Pre)": (lambda f: f"{f}_diff"),
    }
    for label, picker in stages.items():
        rows: List[Dict[str, object]] = []
        for group, subset in panel.groupby("job_group"):
            row = {"job_group": group, "n": int(subset.shape[0])}
            for factor in factors:
                col = picker(factor)
                row[f"{factor}_mean"] = float(subset[col].mean())
                row[f"{factor}_std"] = float(subset[col].std(ddof=1))
            rows.append(row)
        outputs[label] = rows
    return outputs


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    job_groups = load_group_assignments()

    factor_panel = build_factor_panel(job_groups)
    item_panel = build_item_panel(job_groups)

    # Pairwise factor comparisons (all combinations).
    factor_tests: List[WelchResult] = []
    factor_labels = {
        "basic_pre": "basic (pre)",
        "community_pre": "community (pre)",
        "financial_pre": "financial (pre)",
        "basic_post": "basic (post)",
        "community_post": "community (post)",
        "financial_post": "financial (post)",
        "basic_diff": "basic (change)",
        "community_diff": "community (change)",
        "financial_diff": "financial (change)",
    }
    for g1, g2 in (("Practitioner", "Researcher"), ("Other", "Researcher"), ("Other", "Practitioner")):
        for column in factor_labels:
            res = welch_test(factor_panel, column, g1, g2)
            if res:
                res.label = factor_labels[column]
                factor_tests.append(res)

    # Raw short-term category comparisons (pre/post/change) for practitioners vs researchers and Other vs each.
    item_tests: List[WelchResult] = []
    stage_suffix = {"pre": "pre", "post": "post", "diff": "change"}
    for g1, g2 in (("Practitioner", "Researcher"), ("Other", "Researcher"), ("Other", "Practitioner")):
        for stage in ("pre", "post", "diff"):
            for col in SHORT_TERM_COLUMNS:
                column = f"{col}_{stage}" if stage != "diff" else f"{col}_diff"
                res = welch_test(item_panel, column, g1, g2)
                if res:
                    res.label = f"{col} ({stage_suffix[stage]})"
                    item_tests.append(res)

    # Write practitioner vs researcher subset (factor + raw).
    practitioner_mask = [
        idx
        for idx, res in enumerate(factor_tests)
        if res.group1 == "Practitioner" and res.group2 == "Researcher"
    ]
    practitioner_factor = [factor_tests[idx] for idx in practitioner_mask]
    practitioner_items = [
        res for res in item_tests if res.group1 == "Practitioner" and res.group2 == "Researcher"
    ]
    practitioner_data = {
        "factor_scores": attach_fdr(practitioner_factor),
        "raw_items": attach_fdr(practitioner_items),
    }
    (REPORT_DIR / "job_practitioner_combined_checks.json").write_text(
        json.dumps(practitioner_data, indent=2), encoding="utf-8"
    )

    # Write Other-group comparisons.
    other_factors = [
        res for res in factor_tests if res.group1 == "Other" and res.group2 in {"Researcher", "Practitioner"}
    ]
    other_items = [
        res for res in item_tests if res.group1 == "Other" and res.group2 in {"Researcher", "Practitioner"}
    ]
    other_data = {
        "factor_scores": attach_fdr(other_factors),
        "raw_items": attach_fdr(other_items),
    }
    (REPORT_DIR / "job_other_group_checks.json").write_text(
        json.dumps(other_data, indent=2), encoding="utf-8"
    )

    # Summary & pairwise stats across all categories.
    summary = {
        "group_stats": summarise_factors(factor_panel),
        "pairwise_comparisons": attach_fdr(factor_tests),
    }
    (REPORT_DIR / "job_group_collapsed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Job-group comparison tables regenerated.")


if __name__ == "__main__":
    main()
