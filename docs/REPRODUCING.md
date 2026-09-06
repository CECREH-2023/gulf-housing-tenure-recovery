# Reproduction guide

Code and selected aggregate figures; controlled-access survey inputs required.

## Verify the distribution

From the package root:

```bash
python scripts/check_package.py
python scripts/check_inputs.py
```

The first command verifies the shipped files and hashes. The second checks whether separately acquired inputs are present and exits with code 2 when they are missing. Neither command estimates a statistical model.

## Run the selected workflow

The retained model scripts read regression, factor-score, and insurance tables under data/processed. Run with PYTHONPATH=. from the package root. R scripts provide additional inference checks. Retrieval indexes, respondent maps, survey records, and human-derived personas are excluded.

Use a disposable working copy when running the original analysis: several original scripts overwrite their project-relative output locations. Keep the distributed reference snapshot for comparison.

Environment: `python -m pip install -e .`

```bash
PYTHONPATH=. python src/analysis/run_recovery_models.py
PYTHONPATH=. python src/analysis/run_insurance_models.py
```

## Input contract

Required paths are listed in [data/INPUTS.json](../data/INPUTS.json). Only code and selected aggregate figures are released. Responses, transcripts, demographic personas, and retrieval indexes are excluded.

[Source guide](CODE_MAP.md) identifies additional acquisition, sensitivity, and rendering modules. Original modeling and uncertainty procedures are retained. Use the documented input definitions; undocumented data substitutions can change the analysis.

## Evidence

[VALIDATION.json](../VALIDATION.json) records the checks performed on this snapshot. A partial model run or a fictional demo is identified by its limited scope. Full reproduction is claimed only where that record explicitly supports it.
