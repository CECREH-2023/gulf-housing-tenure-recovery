# Housing Tenure and Gulf Coast Disaster Recovery

This study examines how housing tenure and neighborhood attachment are associated with insurance, displacement, and recovery burdens among Gulf Coast households. It combines survey summaries with adjusted statistical models.

## Results and interpretation

The study concerns **521 surveyed households** affected by Gulf Coast hazards. Among respondents with nonmissing policy items, **81.9% of homeowners (n = 249)** and **57.3% of renters (n = 248)** report mixed-peril insurance coverage.

In the hazard-clustered adjusted model, renters have **0.411 times the odds of coverage (95% CI 0.268–0.629)** relative to homeowners. Post-event attachment and income are positively associated with reported coverage. Mixed-peril coverage means a potentially relevant policy was in force; it does not establish comprehensive coverage for every hazard. These observational relationships do not establish causal effects, and coverage-table denominators differ from the full survey sample. See the [aggregate insurance results](results/reference/).

## Explore this repository

- [Figures](figures/)
- [Reference results](results/reference/)
- [Methods](docs/METHODS.md)
- [Reproduction and dependencies](docs/REPRODUCING.md)
- [Source guide](docs/CODE_MAP.md)
- [Data sources and availability](data/README.md)

## Reproduce the work

Start with the [reproduction guide](docs/REPRODUCING.md) for the entry point, inputs, software, and validation limits.

**Scope:** Code and selected aggregate figures; controlled-access survey inputs required.

## Attribution and use

The associated manuscript is “Tenure and Attachment in Post-Disaster Recovery: Evidence from 521 Households across the Gulf Corridor.”

Maintained by [CECREH at Texas Tech University](https://www.depts.ttu.edu/cecreh/). Documentation reviewed September 6, 2026. Cite the repository version you used; see [citation guidance](CITATION.md).

The package metadata declares MIT terms; see [pyproject.toml](pyproject.toml). Source-data and third-party terms apply separately.
