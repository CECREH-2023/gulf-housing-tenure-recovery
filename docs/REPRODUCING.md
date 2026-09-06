# Reproduction guide

Code and selected aggregate figures; controlled-access survey inputs required.

## Software

Run from the repository root in a separate Python environment. The documented installation profile is:

```bash
python -m pip install numpy pandas scipy statsmodels pydantic pydantic-settings
```

See [software requirements](SOFTWARE.md) for optional stages and environment limits.

## Run the workflow

The retained model scripts read regression, factor-score, and insurance tables under data/processed. Run with PYTHONPATH=. from the package root. R scripts provide additional inference checks. Retrieval indexes, respondent maps, survey records, and human-derived personas are excluded.

```bash
PYTHONPATH=. python src/analysis/run_recovery_models.py
PYTHONPATH=. python src/analysis/run_insurance_models.py
```

Research scripts may overwrite their project-relative output files. Run analyses in a working copy and retain the checked-in reference results for comparison.

## Inputs

The exact external paths are listed in [data/README.md](../data/README.md) and [INPUTS.json](../data/INPUTS.json). `python scripts/check_inputs.py` checks their presence and exits with code 2 if any listed path is missing. It does not check the schema, verify access rights, or acquire upstream data.

## Verification scope

`python scripts/check_package.py` checks the distributed file hashes. It uses only the Python standard library and does not fit a model. Run it before generating outputs; new files outside the designated generated-results directory may be reported as extras.

[VALIDATION.json](../VALIDATION.json) records the checks performed for this version and their limits. Inclusion of an analysis module is not evidence that it has been executed. The [source guide](CODE_MAP.md) identifies the distributed modules.
