# Housing Tenure and Gulf Coast Disaster Recovery

How are housing tenure and neighborhood attachment associated with insurance, displacement, and recovery burdens?

Survey-based comparisons and adjusted models. Respondent-level data and qualitative material require controlled access; causal effects are not established by the associations.

## Results and interpretation

The study concerns **521 surveyed households** affected by Gulf Coast hazards. Among respondents with nonmissing policy items, **81.9% of homeowners (n = 249)** and **57.3% of renters (n = 248)** report mixed-peril insurance coverage.

In the hazard-clustered adjusted model, renters have **0.411 times the odds of coverage (95% CI 0.268–0.629)** relative to homeowners. Post-event attachment and income are positively associated with reported coverage. Mixed-peril coverage means a potentially relevant policy was in force; it does not establish comprehensive coverage for every hazard. These observational relationships do not establish causal effects, and coverage-table denominators differ from the full survey sample. See the [aggregate insurance results](results/reference/).

## Explore this repository

- [Figures](figures/)
- [Reference results](results/reference/)
- [Methods](docs/METHODS.md)
- [Reproduction and dependencies](docs/REPRODUCING.md)
- [Analysis source guide](docs/CODE_MAP.md)
- [Data sources and availability](data/README.md)

## Reproduce the work

**Available reproduction:** Code and selected aggregate figures; controlled-access survey inputs required.

Start with `python scripts/check_package.py` to check the file manifest, then follow the [reproduction guide](docs/REPRODUCING.md). A file-integrity check does not rerun the research analysis. Only code and selected aggregate figures are released. Responses, transcripts, demographic personas, and retrieval indexes are excluded.

## Attribution and use

The associated manuscript is “Tenure and Attachment in Post-Disaster Recovery: Evidence from 521 Households across the Gulf Corridor.”

A research resource from [CECREH at Texas Tech University](https://www.depts.ttu.edu/cecreh/). Snapshot: September 6, 2026. For code citation, use the repository URL and the commit identifier for the version you used; see [citation guidance](CITATION.md).

No additional reuse license is granted by this snapshot. Contact the authors through CECREH about permissions; source-data terms apply separately.
