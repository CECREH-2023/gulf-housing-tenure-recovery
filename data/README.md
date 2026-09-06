# Data sources and availability

Gulf Coast disaster-recovery survey and associated coded measures.

Only code and selected aggregate figures are released. Responses, transcripts, demographic personas, and retrieval indexes are excluded.

The retained model scripts read regression, factor-score, and insurance tables under data/processed. Run with PYTHONPATH=. from the package root. R scripts provide additional inference checks. Retrieval indexes, respondent maps, survey records, and human-derived personas are excluded.

## File-level records

- [Required inputs](INPUTS.json) describes separately acquired files.
- [Released table inventory](TABLES.csv) lists distributed table columns and row counts.
- [Source fingerprints](../SOURCE_FILES.csv) links retained source content to its hashes without publishing private workspace paths.

Raw records, access credentials, personal notes, correspondence, and publisher full-text collections are not distributed.
