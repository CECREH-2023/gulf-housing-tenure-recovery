#!/usr/bin/env python3
"""Export derived recovery-model dataset for downstream R analyses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.run_recovery_models import load_base_dataset, prepare_features  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "regression" / "recovery_model_dataset.csv"


def main() -> None:
    df = prepare_features(load_base_dataset())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Exported {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
