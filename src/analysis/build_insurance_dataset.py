#!/usr/bin/env python3
"""Construct hazard-aligned insurance analysis dataset with exposure controls.

This module is integrated with CENTAUR infrastructure for path management.

Usage:
    python -m src.analysis.build_insurance_dataset
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod

from config.settings import get_settings

# =============================================================================
# CENTAUR-Integrated Path Configuration
# =============================================================================
settings = get_settings()

RESULTS_DIR = settings.project_root / "results" / "insurance"
PROCESSED_DIR = settings.processed_dir

REGRESSION_PATH = settings.processed_dir / "regression" / "regression_dataset_with_bert_themes.csv"
FACTOR_PATH = (
    settings.processed_dir
    / "factor_analysis"
    / "polychoric_fa"
    / "promax"
    / "factor_scores_with_ids.csv"
)
STATES_PATH = settings.data_dir / "spatial" / "ne_admin1" / "ne_10m_admin_1_states_provinces_lakes.shp"

SURVEY_YEAR = 2021

GULF_COORDS: Sequence[tuple[float, float]] = (
    (27.8, -97.05),  # Corpus Christi, TX
    (29.76, -95.37),  # Houston, TX
    (29.75, -94.97),  # Port Arthur, TX
    (29.77, -93.21),  # Lake Charles, LA
    (30.23, -91.88),  # Lafayette, LA
    (29.95, -90.07),  # New Orleans, LA
    (30.32, -89.45),  # Gulfport, MS
    (30.68, -88.04),  # Mobile, AL
    (30.25, -85.76),  # Panama City, FL
    (29.71, -83.96),  # Perry, FL
    (27.77, -82.64),  # Tampa, FL
    (26.14, -81.79),  # Naples, FL
)


def _ordered_map(values: Sequence[str]) -> Mapping[str, int]:
    return {value: idx + 1 for idx, value in enumerate(values)}


def yes_no(series: pd.Series) -> pd.Series:
    """Recodes Qualtrics Yes/No style responses to floats."""
    return series.map({"Yes": 1.0, "No": 0.0})


def assign_hazard_class(category: str | float) -> str:
    if isinstance(category, float) and math.isnan(category):
        return "unknown"
    if not category:
        return "unknown"
    label = str(category).lower()
    if "winter" in label:
        return "winter"
    if "flood" in label:
        return "flood"
    if "hurricane" in label:
        return "hurricane"
    if "tornado" in label:
        return "wind"
    if "fire" in label:
        return "fire"
    return "other"


def load_state_boundaries() -> gpd.GeoDataFrame:
    states = gpd.read_file(STATES_PATH)
    states = states.loc[states["adm0_a3"] == "USA", ["postal", "name", "geometry"]]
    states = states.rename(columns={"postal": "state_postal", "name": "state_name"})
    return states.to_crs(epsg=4326)


def attach_state(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing state abbreviations using point-in-polygon lookup."""
    mask = df["state_abbr"].fillna("").isin(["", "Unknown", "unknown", "UNK"])
    if not mask.any():
        df["state_name_filled"] = np.nan
        return df

    subset = df.loc[mask].copy()
    points = gpd.GeoDataFrame(
        subset,
        geometry=gpd.points_from_xy(subset["long"], subset["lat"], crs="EPSG:4326"),
    )
    states = load_state_boundaries()
    joined = gpd.sjoin(points, states, how="left", predicate="within")
    df.loc[mask, "state_abbr"] = joined["state_postal"].fillna("Unknown")
    df.loc[mask, "state_name_filled"] = joined["state_name"]
    df.loc[~mask, "state_name_filled"] = np.nan
    return df


def compute_distance_to_gulf(latitudes: Iterable[float], longitudes: Iterable[float]) -> np.ndarray:
    geod = Geod(ellps="WGS84")
    coords = np.asarray(list(zip(latitudes, longitudes)), dtype=float)
    gulf_points = np.asarray(GULF_COORDS, dtype=float)
    distances_km = np.full(len(coords), np.nan, dtype=float)
    for idx, (lat, lon) in enumerate(coords):
        if np.isnan(lat) or np.isnan(lon):
            continue
        # geodesic distance to each reference point
        dists = []
        for gulf_lat, gulf_lon in gulf_points:
            _, _, dist_m = geod.inv(lon, lat, gulf_lon, gulf_lat)
            dists.append(dist_m / 1000.0)
        distances_km[idx] = float(np.nanmin(dists))
    return distances_km


def prepare_dataset() -> pd.DataFrame:
    base = pd.read_csv(REGRESSION_PATH, low_memory=False)
    factors = pd.read_csv(
        FACTOR_PATH,
        usecols=[
            "ResponseId",
            "attachment_belonging",
            "move_intent_disamenity",
            "trust_safety",
            "social_cohesion",
            "quality_of_life",
        ],
    )
    merged = base.merge(factors, on="ResponseId", how="left", suffixes=("", "_factor"))
    merged = merged.rename(columns={"ResponseId": "response_id"})

    merged["hazard_class"] = merged["hazard_category"].map(assign_hazard_class)
    merged["homeowners_policy"] = yes_no(merged["2.10_1"])
    merged["flood_policy"] = yes_no(merged["2.10_2"])
    merged["renters_policy"] = yes_no(merged["2.10_3"])

    merged["tenure"] = merged["htype"].fillna("Unknown")
    merged["is_homeowner"] = (merged["tenure"] == "Homeowner").astype(float)
    merged["is_renter"] = (merged["tenure"] == "Renter").astype(float)

    merged["owner_home_policy"] = np.where(
        merged["is_homeowner"] == 1.0, merged["homeowners_policy"], np.nan
    )
    merged["owner_flood_policy"] = np.where(
        merged["is_homeowner"] == 1.0, merged["flood_policy"], np.nan
    )
    merged["renter_policy_indicator"] = np.where(
        merged["is_renter"] == 1.0, merged["renters_policy"], np.nan
    )
    merged["renter_flood_policy"] = np.where(
        merged["is_renter"] == 1.0, merged["flood_policy"], np.nan
    )

    def policy_max(values: Iterable[float]) -> float:
        array = np.asarray(list(values), dtype=float)
        if np.isnan(array).all():
            return np.nan
        return float(np.nanmax(array))

    def aligned_policy(row: pd.Series) -> float | np.nan:
        hazard = row["hazard_class"]
        if row["tenure"] == "Homeowner":
            if hazard == "flood":
                return row["owner_flood_policy"]
            if hazard == "hurricane":
                return policy_max([row["owner_home_policy"], row["owner_flood_policy"]])
            if hazard in {"wind", "winter"}:
                return row["owner_home_policy"]
            return row["owner_home_policy"]
        if row["tenure"] == "Renter":
            if hazard == "flood":
                return row["renter_flood_policy"]
            if hazard == "hurricane":
                return policy_max([row["renter_policy_indicator"], row["renter_flood_policy"]])
            if hazard in {"wind", "winter"}:
                return row["renter_policy_indicator"]
            return row["renter_policy_indicator"]
        return np.nan

    def aligned_policy_strict(row: pd.Series) -> float | np.nan:
        hazard = row["hazard_class"]
        if row["tenure"] == "Homeowner":
            if hazard in {"flood", "hurricane"}:
                return row["owner_flood_policy"]
            if hazard in {"wind", "winter"}:
                return row["owner_home_policy"]
            return row["owner_home_policy"]
        if row["tenure"] == "Renter":
            if hazard in {"flood", "hurricane"}:
                return row["renter_flood_policy"]
            if hazard in {"wind", "winter"}:
                return row["renter_policy_indicator"]
            return row["renter_policy_indicator"]
        return np.nan

    merged["coverage_peril_aligned"] = merged.apply(aligned_policy, axis=1)
    merged["coverage_peril_aligned_strict"] = merged.apply(aligned_policy_strict, axis=1)
    merged["coverage_original"] = pd.to_numeric(merged["insurance_yes"], errors="coerce")
    policy_cols = ["homeowners_policy", "flood_policy", "renters_policy"]
    merged["coverage_any_policy"] = merged[policy_cols].max(axis=1, skipna=True)
    missing_policy = merged[policy_cols].isna().all(axis=1)
    merged.loc[missing_policy, "coverage_any_policy"] = np.nan

    income_map = _ordered_map(
        [
            "Less than $10,000",
            "$10,000 to $19,999",
            "$20,000 to $29,999",
            "$30,000 to $39,999",
            "$40,000 to $49,999",
            "$50,000 to $59,999",
            "$60,000 to $69,999",
            "$70,000 to $79,999",
            "$80,000 to $89,999",
            "$90,000 to $99,999",
            "$100,000 to $149,999",
            "$150,000 or more",
        ]
    )
    education_map = _ordered_map(
        [
            "Less than high school degree",
            "High school graduate (high school diploma or equivalent including GED)",
            "Some college but no degree",
            "Associate degree in college (2-year)",
            "Bachelor's degree in college (4-year)",
            "Master's degree",
            "Professional degree (JD, MD)",
            "Doctoral degree",
        ]
    )
    damage_map = {
        "No damage": 0,
        "Not very extensive": 1,
        "Somewhat extensive": 2,
        "Very extensive": 3,
    }

    merged["income_ord"] = merged["1.9"].map(income_map)
    merged["education_ord"] = merged["education"].map(education_map)
    merged["damage_ord"] = merged["2.8_1"].map(damage_map)
    merged["aid_flag"] = (merged["2.12"] == "Yes").astype(float)
    merged["displacement_flag"] = pd.to_numeric(
        merged["displaced_event_flag"], errors="coerce"
    )
    merged["employed"] = (merged["1.8"] == "Employed").astype(float)
    merged["age_years"] = SURVEY_YEAR - pd.to_numeric(merged["age"], errors="coerce")
    merged["household_size"] = pd.to_numeric(merged["hsize"], errors="coerce")
    merged["female"] = (merged["gender"] == "Woman").astype(float)
    merged["nonwhite"] = (
        ~merged["race"].fillna("").str.contains("White", case=False)
    ).astype(float)
    merged["hispanic"] = (
        merged["race"].fillna("").str.contains("Hispanic", case=False).astype(float)
    )
    merged["married"] = (merged["marital"] == "Married").astype(float)
    merged["mfr_flag"] = merged["structure_category"].str.contains(
        "Multifamily", case=False, na=False
    ).astype(float)
    merged["attachment_belonging"] = pd.to_numeric(
        merged["attachment_belonging"], errors="coerce"
    )
    merged["move_intent_disamenity"] = pd.to_numeric(
        merged["move_intent_disamenity"], errors="coerce"
    )
    merged["trust_safety"] = pd.to_numeric(merged["trust_safety"], errors="coerce")
    merged["social_cohesion"] = pd.to_numeric(merged["social_cohesion"], errors="coerce")
    merged["quality_of_life"] = pd.to_numeric(
        merged["quality_of_life"], errors="coerce"
    )

    merged = attach_state(merged)
    merged["distance_to_gulf_km"] = compute_distance_to_gulf(merged["lat"], merged["long"])
    merged["coastal_county_flag"] = (merged["distance_to_gulf_km"] <= 50).astype(float)

    keep_columns = [
        "response_id",
        "tenure",
        "hazard_category",
        "hazard_class",
        "state_abbr",
        "state_name_filled",
        "lat",
        "long",
        "distance_to_gulf_km",
        "coastal_county_flag",
        "coverage_original",
        "coverage_peril_aligned",
        "coverage_peril_aligned_strict",
        "coverage_any_policy",
        "homeowners_policy",
        "flood_policy",
        "renters_policy",
        "owner_home_policy",
        "owner_flood_policy",
        "renter_policy_indicator",
        "renter_flood_policy",
        "income_ord",
        "damage_ord",
        "aid_flag",
        "displacement_flag",
        "age_years",
        "household_size",
        "employed",
        "attachment_belonging",
        "move_intent_disamenity",
        "trust_safety",
        "social_cohesion",
        "quality_of_life",
        "female",
        "nonwhite",
        "hispanic",
        "education_ord",
        "married",
        "mfr_flag",
    ]

    return merged[keep_columns]


def write_outputs(dataset: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "insurance").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "insurance" / "insurance_model_dataset.csv"
    dataset.to_csv(out_path, index=False)
    legacy_path = PROCESSED_DIR / "insurance_model_dataset.csv"
    dataset.to_csv(legacy_path, index=False)

    counts = (
        dataset.groupby(["hazard_class", "tenure"])
        .agg(
            n=("response_id", "count"),
            coverage_rate=("coverage_peril_aligned", "mean"),
            coverage_strict=("coverage_peril_aligned_strict", "mean"),
            coverage_any=("coverage_any_policy", "mean"),
            original_rate=("coverage_original", "mean"),
        )
        .reset_index()
    )
    counts = counts.sort_values(["hazard_class", "tenure"])
    counts.to_csv(RESULTS_DIR / "hazard_class_counts.csv", index=False)


def main() -> None:
    dataset = prepare_dataset()
    write_outputs(dataset)


if __name__ == "__main__":
    main()
