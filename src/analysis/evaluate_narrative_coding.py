#!/usr/bin/env python3
"""Evaluate narrative theme classifiers and generate threshold sensitivity tables.

This module is integrated with CENTAUR infrastructure for path management.

Usage:
    python -m src.analysis.evaluate_narrative_coding
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from config.settings import get_settings

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
settings = get_settings()

ANALYSIS_DIR = settings.processed_dir / "bert_scores"
RESULTS_DIR = settings.project_root / "results" / "narrative"

MANUAL_EVAL_PATH = ANALYSIS_DIR / "enhanced_manual_eval_predictions.csv"
SCORES_PATH = ANALYSIS_DIR / "enhanced_zero_shot_scores.csv"

THEMES: Sequence[str] = (
    "financial_hardship",
    "housing_status",
    "displacement",
    "prior_disaster",
    "health_or_disability",
    "aid_dependency",
    "property_damage",
)

THEME_PROMPTS: Sequence[str] = (
    "financial hardship, job loss, unemployment, or inability to afford necessities",
    "housing instability, eviction, landlord problems, homelessness, or shelter issues",
    "displacement, evacuation, temporary housing, or staying with friends/family",
    "prior disaster experience, repeated natural hazards, or past storms and floods",
    "health problems, illness, disability, or long-term medical impacts (including COVID-19)",
    "dependence on FEMA, aid organizations, government assistance, or relief programs",
    "property damage, structural loss, flooding, or destruction of belongings",
)

ENSEMBLE_MODELS: Sequence[str] = (
    "facebook/bart-large-mnli",
    "MoritzLaurer/deberta-v3-large-zeroshot-v1",
)

THRESHOLDS: Sequence[float] = (0.6, 0.7, 0.8)


def load_manual_labels() -> pd.DataFrame:
    required = [
        "ResponseId",
        *THEMES,
        *[f"{theme}_ensemble_prob" for theme in THEMES],
    ]
    df = pd.read_csv(MANUAL_EVAL_PATH, usecols=lambda c: c in required or c in {"htype", "clean_text"})
    return df


def load_full_scores() -> pd.DataFrame:
    keep = ["ResponseId", *[f"{theme}_ensemble_prob" for theme in THEMES]]
    return pd.read_csv(SCORES_PATH, usecols=keep)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = y_score >= threshold
    tp = float(np.sum((preds == 1) & (y_true == 1)))
    fp = float(np.sum((preds == 1) & (y_true == 0)))
    fn = float(np.sum((preds == 0) & (y_true == 1)))
    tn = float(np.sum((preds == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if precision + recall == 0 or np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)
    support = float(np.sum(y_true == 1))
    prevalence = float(np.mean(preds))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "prevalence": prevalence,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_thresholds(manual: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for threshold in THRESHOLDS:
        for theme in THEMES:
            truth = manual[theme].astype(int).to_numpy()
            scores = manual[f"{theme}_ensemble_prob"].astype(float).to_numpy()
            metrics = binary_metrics(truth, scores, threshold)
            metrics.update({"threshold": threshold, "theme": theme})
            records.append(metrics)
    return pd.DataFrame.from_records(records)


def prevalence_tables(full_scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for threshold in THRESHOLDS:
        preds = (full_scores[[f"{theme}_ensemble_prob" for theme in THEMES]] >= threshold).astype(int)
        for idx, theme in enumerate(THEMES):
            rows.append(
                {
                    "threshold": threshold,
                    "theme": theme,
                    "prevalence": preds.iloc[:, idx].mean(),
                }
            )
    return pd.DataFrame(rows)


def compounding_burden(full_scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    probs = full_scores[[f"{theme}_ensemble_prob" for theme in THEMES]]
    for threshold in THRESHOLDS:
        flags = (probs >= threshold).astype(int)
        counts = flags.sum(axis=1)
        rows.append(
            {
                "threshold": threshold,
                "mean_themes": counts.mean(),
                "median_themes": counts.median(),
                "pct_any": (counts > 0).mean(),
                "pct_two_or_more": (counts >= 2).mean(),
                "pct_three_or_more": (counts >= 3).mean(),
                "max_themes": counts.max(),
            }
        )
    return pd.DataFrame(rows)


def write_json_metadata(manual: pd.DataFrame) -> None:
    metadata = {
        "ensemble_models": list(ENSEMBLE_MODELS),
        "prompt_definitions": dict(zip(THEMES, THEME_PROMPTS)),
        "thresholds_evaluated": list(THRESHOLDS),
        "manual_label_sample_size": int(len(manual)),
        "manual_label_positive_counts": {
            theme: int(manual[theme].sum()) for theme in THEMES
        },
    }
    (RESULTS_DIR / "narrative_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manual = load_manual_labels()
    scores = load_full_scores()

    metrics = evaluate_thresholds(manual)
    prevalence = prevalence_tables(scores)
    burden = compounding_burden(scores)

    metrics.to_csv(RESULTS_DIR / "threshold_metrics.csv", index=False)
    prevalence.to_csv(RESULTS_DIR / "threshold_prevalence.csv", index=False)
    burden.to_csv(RESULTS_DIR / "compounding_burden.csv", index=False)
    metrics.query("threshold == 0.7").to_csv(RESULTS_DIR / "metrics_threshold_0_7.csv", index=False)
    write_json_metadata(manual)


if __name__ == "__main__":
    main()
