"""Analysis module for Gulf Hazard Interview research.

This module provides refactored analysis scripts integrated with CENTAUR
infrastructure for LLM-powered interpretation and RAG-enhanced analysis.

Modules:
    recovery_modeling: Recovery regression models with LLM interpretation
    run_recovery_models: Logistic models with clustered SEs
    run_insurance_models: Insurance logistic regression
    analyze_financial_losses: Financial loss diagnostics
    build_insurance_dataset: Hazard-aligned insurance dataset
    compute_insurance_summaries: Insurance coverage summaries
    evaluate_narrative_coding: Theme classifier evaluation
    run_factor_models: Factor analysis replication
    run_mediation_analysis: Mediation effect estimation
    run_job_group_checks: Job-group comparisons
    r_runner: R script integration via CENTAUR
"""

from pathlib import Path

from config.settings import get_settings

# Get settings for path access
settings = get_settings()

# Analysis data paths (CENTAUR-integrated)
DATA_DIR = settings.processed_dir

# Import all analysis modules
from src.analysis import (
    run_recovery_models,
    run_insurance_models,
    analyze_financial_losses,
    build_insurance_dataset,
    compute_insurance_summaries,
    evaluate_narrative_coding,
    run_factor_models,
    run_mediation_analysis,
    run_job_group_checks,
    r_runner,
)

__all__ = [
    # Paths
    "DATA_DIR",
    # Python analysis modules
    "run_recovery_models",
    "run_insurance_models",
    "analyze_financial_losses",
    "build_insurance_dataset",
    "compute_insurance_summaries",
    "evaluate_narrative_coding",
    "run_factor_models",
    "run_mediation_analysis",
    "run_job_group_checks",
    # R integration
    "r_runner",
]
