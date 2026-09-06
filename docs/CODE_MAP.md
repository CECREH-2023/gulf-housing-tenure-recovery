# Source guide

Start with [REPRODUCING.md](REPRODUCING.md) for the supported entry point and its inputs. The modules below also include upstream preparation and supplementary analyses. Their presence does not establish that every stage is runnable from bundled data.

| Module | Purpose |
|---|---|
| [config/settings.py](../config/settings.py) | Gulf_Hazard_Interview configuration using pydantic. |
| [src/analysis/analyze_financial_losses.py](../src/analysis/analyze_financial_losses.py) | Generate financial loss distribution diagnostics with trimming/winsorization scenarios. |
| [src/analysis/build_insurance_dataset.py](../src/analysis/build_insurance_dataset.py) | Construct hazard-aligned insurance analysis dataset with exposure controls. |
| [src/analysis/compute_insurance_summaries.py](../src/analysis/compute_insurance_summaries.py) | Compute insurance coverage summaries used across Gulf Interview reporting. |
| [src/analysis/evaluate_narrative_coding.py](../src/analysis/evaluate_narrative_coding.py) | Evaluate narrative theme classifiers and generate threshold sensitivity tables. |
| [src/analysis/event_timing.py](../src/analysis/event_timing.py) | Utilities for deriving event timing variables. |
| [src/analysis/export_recovery_model_dataset.py](../src/analysis/export_recovery_model_dataset.py) | Export derived recovery-model dataset for downstream R analyses. |
| [src/analysis/r_runner.py](../src/analysis/r_runner.py) | R script runner using CENTAUR multilanguage support. |
| [src/analysis/recovery_modeling.py](../src/analysis/recovery_modeling.py) | Recovery modeling with CENTAUR LLM integration. |
| [src/analysis/run_factor_models.py](../src/analysis/run_factor_models.py) | Replicate and extend the short-term recovery factor analysis. |
| [src/analysis/run_insurance_models.py](../src/analysis/run_insurance_models.py) | Recompute insurance logistic regression tables with consistent renter coding. |
| [src/analysis/run_job_group_checks.py](../src/analysis/run_job_group_checks.py) | Generate job-group comparison summaries with multiple-comparison control. |
| [src/analysis/run_mediation_analysis.py](../src/analysis/run_mediation_analysis.py) | Exploratory mediation models; causal interpretation requires assumptions beyond the observational data. |
| [src/analysis/run_recovery_models.py](../src/analysis/run_recovery_models.py) | Re-estimate Gulf Interview logistic models with clustered SEs and scenario outputs. |
| [src/centaur_link.py](../src/centaur_link.py) | CENTAUR infrastructure linkage module. |
| [src/data_prep/extract_workbook.py](../src/data_prep/extract_workbook.py) | Decrypt and extract workbook sheets into analysis-friendly files. |
| [src/llm/providers.py](../src/llm/providers.py) | LLM provider interface for Gulf Hazard Interview analysis. |
| [src/rag/interview_metadata.py](../src/rag/interview_metadata.py) | Interview metadata dataclasses for Gulf Hazard Interview analysis. |
| [src/rag/interview_retriever.py](../src/rag/interview_retriever.py) | Interview-aware retriever extending CENTAUR's RAG capabilities. |
| [r_analysis/analyze_insurance_models.R](../r_analysis/analyze_insurance_models.R) | Analyze insurance models. |
| [r_analysis/run_cluster_inference.R](../r_analysis/run_cluster_inference.R) | Run cluster inference. |
| [r_analysis/run_hurricane_coverage_sensitivity.R](../r_analysis/run_hurricane_coverage_sensitivity.R) | Run hurricane coverage sensitivity. |
| [r_analysis/run_measurement_invariance.R](../r_analysis/run_measurement_invariance.R) | Run measurement invariance. |
| [r_analysis/run_missingness_and_mi.R](../r_analysis/run_missingness_and_mi.R) | Run missingness and mi. |
