#!/usr/bin/env python3
"""Replicate and extend the short-term recovery factor analysis.

This module is integrated with CENTAUR infrastructure for path management.

Usage:
    python -m src.analysis.run_factor_models
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pandas as pd
from factor_analyzer import (
    FactorAnalyzer,
    calculate_bartlett_sphericity,
    calculate_kmo,
)

from config.settings import get_settings

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
settings = get_settings()

DATA_PATH = settings.data_dir / "interim" / "grouped_all_pre.csv"
MATRIX_PATH = settings.data_dir / "interim" / "short_term_matrix.csv"
OUTPUT_DIR = settings.processed_dir / "factor_analysis"

COLUMN_GROUPS = {
    "Communication": ["Communication-ST-pre", "comct-st-1"],
    "Food": ["Food-ST-pre", "food-st-1"],
    "Water": ["Water-ST-pre", "water-st-1"],
    "Shelter": ["Shelter-ST-pre", "shelter-st-1"],
    "Lifelines": ["Infrastructure-ST-pre", "Power-ST-pre", "Transportation-ST-pre", "lifelines-st-1"],
    "Job": ["Employment-ST-pre", "job-st-1"],
    "Health": ["Healthcare-ST-pre", "health-st-2", "phealth-st-1", "mhealth-st-1"],
    "Social capital": ["SocialCapital -ST-pre", "SCP-ST-pre", "socn-st-1", "cmnty-st-1"],
    "Financial support": ["FinancialResources-ST-pre", "financial-st-2", "financial-st-1"],
}


def load_matrix() -> pd.DataFrame:
    """Build (or reuse) the 84×9 importance matrix."""
    if MATRIX_PATH.exists():
        return pd.read_csv(MATRIX_PATH)

    pre = pd.read_csv(DATA_PATH)

    def combine(columns: list[str]) -> pd.Series:
        result = pre[columns[0]].astype(float)
        for name in columns[1:]:
            result = result.fillna(pre[name].astype(float))
        return result.fillna(0)

    data = pd.DataFrame({name: combine(cols) for name, cols in COLUMN_GROUPS.items()})
    data.to_csv(MATRIX_PATH, index=False)
    return data


def nearest_pd(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest positive-definite matrix (Higham, 1988)."""
    sym = (matrix + matrix.T) / 2
    u, s, _ = np.linalg.svd(sym)
    h = u @ np.diag(s) @ u.T
    sym = (sym + h) / 2
    sym = (sym + sym.T) / 2
    spacing = np.spacing(np.linalg.norm(sym))
    identity = np.eye(sym.shape[0])
    k = 1
    while True:
        try:
            np.linalg.cholesky(sym)
            break
        except np.linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(sym)
            sym += identity * (-eigvals.min() * k**2 + spacing)
            k += 1
    return sym


def factor_report(fa: FactorAnalyzer, index: list[str]) -> Dict[str, pd.DataFrame]:
    loadings = pd.DataFrame(
        fa.loadings_,
        index=index,
        columns=[f"Factor{i+1}" for i in range(fa.n_factors)],
    )
    communalities = pd.Series(fa.get_communalities(), index=index, name="Communality")
    ss, pct, cum = fa.get_factor_variance()
    variance = pd.DataFrame(
        {"SS Loadings": ss, "% Var": pct, "Cumulative %": cum},
        index=[f"Factor{i+1}" for i in range(fa.n_factors)],
    )
    return {
        "loadings": loadings,
        "communalities": communalities.to_frame(),
        "variance": variance,
    }


def write_outputs(tag: str, report: Dict[str, pd.DataFrame]) -> None:
    tag_dir = OUTPUT_DIR / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in report.items():
        frame.to_csv(tag_dir / f"{name}.csv")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_matrix()

    bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(data)
    kmo_vars, kmo_overall = calculate_kmo(data)

    baseline_fa = FactorAnalyzer(n_factors=3, rotation="varimax", method="principal")
    baseline_fa.fit(data)
    baseline_report = factor_report(baseline_fa, list(data.columns))
    write_outputs("efa_principal_varimax", baseline_report)

    ord_data = data.replace(0, np.nan)
    spearman_corr = ord_data.corr(method="spearman", min_periods=5).fillna(0)
    corr_pd = nearest_pd(spearman_corr.values)
    alt_fa = FactorAnalyzer(
        n_factors=3,
        rotation="promax",
        method="minres",
        is_corr_matrix=True,
    )
    alt_fa.fit(corr_pd)
    alt_report = factor_report(alt_fa, list(data.columns))
    write_outputs("efa_spearman_minres_promax", alt_report)

    summary = {
        "diagnostics": {
            "bartlett_chi2": bartlett_chi2,
            "bartlett_p": bartlett_p,
            "kmo_overall": kmo_overall,
            "kmo_variables": dict(zip(data.columns, kmo_vars)),
        }
    }
    (OUTPUT_DIR / "factor_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
