# Data sources and availability

Gulf Coast disaster-recovery survey and associated coded measures.

Code, aggregate figures, and two reference insurance tables are released. Responses, transcripts, demographic personas, and retrieval indexes are excluded.

## External inputs for the entry point

Paths are relative to the repository root. They describe files or directories to supply; they are not bundled download links. Upstream acquisition and optional analyses may require additional inputs.

| Path | Availability |
|---|---|
| `data/processed/regression/regression_dataset_with_bert_themes.csv` | external; not bundled |
| `data/processed/factor_analysis/polychoric_fa/promax/factor_scores_with_ids.csv` | external; not bundled |
| `data/processed/insurance/insurance_model_dataset.csv` | external; not bundled |

## File definitions and provenance

- [Table inventory](TABLES.csv): distributed CSV columns and row counts.
- [Input inventory](INPUTS.json): paths checked by the input preflight.
- [Source fingerprints](../SOURCE_FILES.csv): hashes of retained source files and their public versions.
