#!/usr/bin/env python3
"""Utilities for deriving event timing variables."""

from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd

DATE_PATTERN = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")

# Canonical event dates used when respondents selected named hazards
# or when free-text dates are unavailable. Dates reflect primary landfall
# or impact windows for the 2020 Gulf storms and the 2021 winter storm.
HAZARD_DEFAULTS = {
    "Hurricane Laura": "2020-08-27",
    "Hurricane Delta": "2020-10-09",
    "Hurricane Hanna": "2020-07-25",
    "Hurricane Zeta": "2020-10-28",
    "Hurricane Sally": "2020-09-16",
    "Winter Storm 2021": "2021-02-15",
    "Other Flooding": "2021-05-15",
    "Other Hazard": "2021-04-01",
    "Fire": "2020-12-01",
    "Tornado": "2021-03-25",
}


def _parse_date_string(value: Optional[str]) -> pd.Timestamp:
    """Attempt to parse a date from a free-text field."""
    if not isinstance(value, str):
        return pd.NaT
    text = value.strip()
    if not text:
        return pd.NaT

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed

    match = DATE_PATTERN.search(text)
    if match:
        return pd.to_datetime(match.group(1), errors="coerce")

    return pd.NaT


def _combine_candidate_dates(columns: Iterable[pd.Series], index: pd.Index) -> pd.Series:
    """Prioritise the first non-missing parsed date across candidate columns."""
    candidate = None
    for series in columns:
        if series is None:
            continue
        current = pd.to_datetime(series)
        candidate = current if candidate is None else candidate.fillna(current)
    if candidate is None:
        candidate = pd.Series(pd.NaT, index=index)
    return candidate


def add_event_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with event-date metadata and multiple timing variants."""
    work = df.copy()
    recorded = pd.to_datetime(work.get("RecordedDate"), errors="coerce")

    text_cols = [
        work.get("2.2_7_TEXT"),
        work.get("2.2_8_TEXT"),
        work.get("2.3_3_TEXT"),
    ]
    parsed_columns = [
        col.map(_parse_date_string) if col is not None else None for col in text_cols
    ]
    event_date = _combine_candidate_dates(
        [col for col in parsed_columns if col is not None],
        index=work.index,
    )
    event_source = pd.Series("missing", index=work.index, dtype="object")
    parsed_mask = event_date.notna()
    event_source.loc[parsed_mask] = "parsed"

    hazard_series = work.get("hazard_category")
    if hazard_series is None:
        hazard_series = work.get("hazard_cluster")

    if hazard_series is not None:
        base_for_median = pd.DataFrame(
            {"hazard": hazard_series, "event": event_date}
        ).dropna()
        median_lookup = (
            base_for_median.groupby("hazard")["event"].median().to_dict()
            if not base_for_median.empty
            else {}
        )
        median_values = hazard_series.map(median_lookup)
        median_mask = event_date.isna() & median_values.notna()
        event_date.loc[median_mask] = median_values.loc[median_mask]
        event_source.loc[median_mask] = "hazard_median"

        hazard_defaults = hazard_series.map(HAZARD_DEFAULTS)
        hazard_defaults = pd.to_datetime(hazard_defaults, errors="coerce")
        default_mask = event_date.isna() & hazard_defaults.notna()
        event_date.loc[default_mask] = hazard_defaults.loc[default_mask]
        event_source.loc[default_mask] = "hazard_default"

    fallback_mask = event_date.isna()
    event_date.loc[fallback_mask] = recorded.loc[fallback_mask]
    event_source.loc[fallback_mask] = "fallback_recorded"

    work["recorded_datetime"] = recorded
    work["event_date"] = event_date
    work["event_date_source"] = event_source

    days_since = (recorded - event_date).dt.days
    days_since = days_since.clip(lower=0)
    work["days_since_event_numeric"] = days_since

    fallback_mask = event_source == "fallback_recorded"
    days_option_a = days_since.mask(fallback_mask)
    work["days_since_event_optionA"] = days_option_a

    median_days = days_since[~fallback_mask].median()
    if pd.isna(median_days):
        median_days = 0
    days_option_b = days_since.copy()
    days_option_b.loc[fallback_mask] = median_days
    work["days_since_event_optionB"] = days_option_b

    bins = [-1, 90, 180, 270, 360, 1_000_000]
    labels = [
        "0-3 months",
        "3-6 months",
        "6-9 months",
        "9-12 months",
        ">12 months",
    ]
    time_bins = pd.cut(days_since, bins=bins, labels=labels)
    time_bins = time_bins.astype("object")
    time_bins.loc[fallback_mask] = "Unknown"
    work["time_since_event_bin"] = time_bins

    return work
