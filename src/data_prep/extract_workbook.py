#!/usr/bin/env python3
"""Decrypt and extract workbook sheets into analysis-friendly files."""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

try:
    import msoffcrypto
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "msoffcrypto is required. Install it with `pip install msoffcrypto-tool`."
    ) from exc

try:
    from striprtf.striprtf import rtf_to_text
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "striprtf is required. Install it with `pip install striprtf`."
    ) from exc


DEFAULT_WORKBOOK = "data/raw/emergency_needs_survey.xlsx"
DEFAULT_PASSWORD_FILE = "XLSX Credentials.rtf"
DEFAULT_OUTPUT_DIR = "data/interim"
DEFAULT_METADATA_PATH = "data/metadata/workbook_schema.json"
DEFAULT_LOG_PATH = "logs/extract_workbook.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all sheets from the project workbook into tabular files."
    )
    parser.add_argument(
        "--workbook",
        type=str,
        default=DEFAULT_WORKBOOK,
        help="Path to the encrypted Excel workbook.",
    )
    parser.add_argument(
        "--password-file",
        type=str,
        default=DEFAULT_PASSWORD_FILE,
        help="Path to the RTF/plaintext file containing the workbook password.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted sheets will be written.",
    )
    parser.add_argument(
        "--metadata-out",
        type=str,
        default=DEFAULT_METADATA_PATH,
        help="Path to write workbook schema metadata (JSON).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=DEFAULT_LOG_PATH,
        help="Path for the extraction run log.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["csv"],
        choices=["csv", "parquet"],
        help="File formats to emit for each sheet.",
    )
    return parser.parse_args()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def resolve_workbook_path(candidate: Path) -> Path:
    if candidate.exists():
        return candidate
    alt_path = Path(candidate.name)
    if alt_path.exists():
        logging.info("Workbook not found at %s; using %s instead.", candidate, alt_path)
        return alt_path
    raise FileNotFoundError(f"Workbook not found at {candidate} or {alt_path}.")


def read_password(path: Path) -> str:
    raw_text = path.read_text(encoding="utf-8")
    try:
        password = rtf_to_text(raw_text)
    except Exception:  # pragma: no cover - fallback for plain text
        password = raw_text
    password = password.strip()
    if not password:
        raise ValueError(f"Password file {path} did not contain any characters.")
    return password


def decrypt_workbook_bytes(workbook_path: Path, password: str) -> io.BytesIO:
    logging.info("Decrypting workbook %s", workbook_path)
    with workbook_path.open("rb") as encrypted_file:
        office_file = msoffcrypto.OfficeFile(encrypted_file)
        office_file.load_key(password=password)
        buffer = io.BytesIO()
        office_file.decrypt(buffer)
        buffer.seek(0)
    return buffer


def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[\\s\\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "sheet"


def export_sheet(
    df: pd.DataFrame,
    sheet_name: str,
    slug: str,
    out_dir: Path,
    formats: Iterable[str],
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    if "csv" in formats:
        csv_path = out_dir / f"{slug}.csv"
        df.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)
    if "parquet" in formats:
        parquet_path = out_dir / f"{slug}.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
            outputs["parquet"] = str(parquet_path)
        except ModuleNotFoundError:
            logging.warning(
                "pyarrow/fastparquet not installed; skipping parquet for %s.", sheet_name
            )
    return outputs


def dataframe_sample(df: pd.DataFrame, n: int = 5) -> list[dict]:
    sample_df = df.head(n).copy()
    sample_records = []
    for record in sample_df.to_dict(orient="records"):
        formatted_record = {}
        for key, value in record.items():
            if hasattr(value, "isoformat"):
                formatted_record[key] = value.isoformat()
            elif isinstance(value, float) and pd.isna(value):
                formatted_record[key] = None
            else:
                formatted_record[key] = value
        sample_records.append(formatted_record)
    return sample_records


def main() -> None:
    args = parse_args()

    log_path = Path(args.log_file)
    configure_logging(log_path)

    workbook_path = resolve_workbook_path(Path(args.workbook))
    password_file = Path(args.password_file)

    logging.info("Using password file %s", password_file)
    password = read_password(password_file)

    decrypted_bytes = decrypt_workbook_bytes(workbook_path, password)
    xls = pd.ExcelFile(decrypted_bytes)

    logging.info("Found %d sheets: %s", len(xls.sheet_names), ", ".join(xls.sheet_names))

    metadata: Dict[str, dict] = {}
    output_dir = Path(args.output_dir)

    for sheet in xls.sheet_names:
        logging.info("Processing sheet '%s'", sheet)
        df = pd.read_excel(xls, sheet_name=sheet)
        slug = slugify(sheet)
        outputs = export_sheet(df, sheet, slug, output_dir, args.formats)
        dtype_counts = {
            str(dtype): int(count) for dtype, count in df.dtypes.value_counts().items()
        }
        metadata[sheet] = {
            "slug": slug,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "column_names": df.columns.tolist(),
            "dtype_counts": dtype_counts,
            "output_files": outputs,
            "sample_rows": dataframe_sample(df),
        }

    metadata_path = Path(args.metadata_out)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logging.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
