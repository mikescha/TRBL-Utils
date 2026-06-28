import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

COL_SITE_ID = "Site_ID"
COL_SITE_NAME = "Site_Name"
COL_DEPLOYMENT_START = "Deployment_Start"
COL_DEPLOYMENT_END = "Deployment_End"
COL_BREEDING_TYPE = "Breeding_Type"
COL_COMPLEX_TYPES = "Complex_Types"
COL_APPROX_COLONY_SIZE = "Approx_Colony_Size"
COL_COLONY_SIZE = "Colony_Size"
COL_SUBSTRATE = "Substrate"
COL_GROUP = "Group"
COL_PRETTY_SITE_NAME = "Pretty_Site_Name"
COL_SKIP_SITE = "Skip_Site"
COL_COMMENT = "Comment"
COL_PULSE_NAME = "Pulse_Name"
COL_OUTCOME = "Outcome"
COL_HATCH_DATE = "Hatch_Date"
COL_ABANDON_DATE = "Abandoned_Date"
COL_PARTIAL_ABANDON_DATE = "Partial_Abandon_Date"
STATUS_ND = "ND"

# Manual outcome labels
OUTCOME_ABANDONED = "Abandoned"
OUTCOME_PARTIALLY_ABANDONED = "Partially Abandoned"
OUTCOME_SUCCESSFUL = "Successful"
OUTCOME_UNKNOWN = "Unknown"
OUTCOME_NO_COLONY = "No Colony"
OUTCOME_NO_TRBL = "No TRBL"

# File locations
INPUT_CSV = Path(
    r"C:\Users\mikes\OneDrive\Documents\GitHub\TRBLSummarizer\TRBLSummarizer\Data\TRBL Analysis tracking - All.csv"
)
DATA_ROOT = Path(r"C:\Users\mikes\OneDrive\Documents\GitHub\TRBLSummarizer\TRBLSummarizer")
PMJ_DIR = DATA_ROOT / "PMJ Data"
DATA_DIR = DATA_ROOT / "Data"
HOURLY_PARQUET_FILES = DATA_DIR / Path("recordings_per_day_hour.parquet")
SHARING_OUTPUT_DIR = Path(r"G:\My Drive\TRBL for Wendy GDrive")


def format_date_for_output(value: date | None, missing: str = STATUS_ND) -> str:
    """Formats date values consistently for CSV output."""
    if isinstance(value, date):
        return value.isoformat()
    return missing


def normalize_one_date(value: Any) -> Any:
    preserve_values = {
        "",
        "ND",
        "NHD",
        "inf",
        "missed",
        "n/a",
        "na",
        "nan",
        "None",
        "Continuous",
    }

    if pd.isna(value):
        return value

    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip()
    has_tilde = text.startswith("~")
    cleaned = text.removeprefix("~").strip()

    if cleaned.startswith("before"):
        return cleaned
    
    if cleaned in preserve_values:
        return f"~{cleaned}" if has_tilde else cleaned

    parsed = pd.to_datetime(cleaned, errors="coerce")
    if pd.isna(parsed):
        return value

    iso_date = parsed.date().isoformat()
    return f"~{iso_date}" if has_tilde else iso_date


def normalize_output_date_columns(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """Normalizes selected date-like output columns to YYYY-MM-DD.

    Non-date status values such as NHD, ND, inf, missed, and blanks are preserved.
    Leading ~ markers are preserved while the date itself is normalized.
    """
    normalized = df.copy()

    for col in date_columns:
        if col not in normalized.columns:
            continue
        normalized[col] = normalized[col].apply(normalize_one_date)

    return normalized


def save_csv_with_retry(df: pd.DataFrame, path: Path, share = False) -> None:
    while True:
        try:
            df.to_csv(path, index=False)
            break
        except PermissionError:
            input(f"\n[!] Output file is locked in Excel: {path.name}\nClose it and press Enter to retry...")

    if share and SHARING_OUTPUT_DIR.exists():
        shutil.copy2(path, SHARING_OUTPUT_DIR / path.name)

