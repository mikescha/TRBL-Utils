from __future__ import annotations

import shutil
from datetime import date, timedelta
from functools import lru_cache
from glob import escape
from pathlib import Path
from typing import Any, cast

import pandas as pd

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
MIN_REQUIRED_DAYS = 4
START_HOUR = 7
END_HOUR = 20

BASE_DIR = Path(".")
DATA_DIR = Path(r"C:\Users\mikes\OneDrive\Documents\GitHub\TRBLSummarizer\TRBLSummarizer")
PMJ_DIR = DATA_DIR / "PMJ Data"
SHARING_OUTPUT_DIR = Path(r"G:\My Drive\TRBL for Wendy GDrive")

BREEDING_DATES_CSV = BASE_DIR / "breeding dates.csv"
OUT_FILE = BASE_DIR / "nestling-to-female-ratios.csv"
OUT_FILTERED_FILE = BASE_DIR / "nestling-to-female-ratios-filtered.csv"
OUT_ARI_FILE = BASE_DIR / "nestling-to-female-ratios-for-comparison.csv"
RESULTS_TXT = BASE_DIR / "ratios_results.txt"
OUT_DIAGNOSTIC_FILE = BASE_DIR / "nestling-to-female-ratios-diagnostics.csv"
OUT_DAILY_DIAGNOSTIC_FILE = BASE_DIR / "nestling-to-female-ratios-daily-diagnostics.csv"

AVG_FEMALE_CALLS = "Avg_Female_Calls_Per_Rec"
AVG_NESTLING_CALLS = "Avg_Nestling_Calls_Per_Rec"


# Path to the specified parquet database for total recordings
RECORDINGS_PARQUET = DATA_DIR / "Data" / "recordings_per_day_hour.parquet"

DECIMALS = 3

# Biological / descriptive thresholds
ARI_HIGH_THRESHOLD = 0.5
LOW_FEMALE_DENOMINATOR_THRESHOLD = 5

# ARI status values
STATUS_ND = "ND"
ARI_STATUS_NUMERIC = "Numeric"
ARI_STATUS_NHD = "NHD"
ARI_STATUS_MISSING_DATES = "ND_MISSING_DATES"
ARI_STATUS_INVALID_BREEDING_TYPE = "ND_INVALID_BREEDING_TYPE"
ARI_STATUS_NO_FEMALE_CALLS = "ND_NO_FEMALE_CALLS"
ARI_STATUS_INSUFFICIENT_DAYS = "ND_INSUFFICIENT_DAYS"

# ARI descriptive classes for publication-facing output
ARI_CLASS_NOT_SCORABLE = "Not ARI-scorable"
ARI_CLASS_NO_OFFSPRING_EVIDENCE = "No detected offspring acoustic evidence"
ARI_CLASS_REDUCED_OFFSPRING_ACTIVITY = "Reduced offspring acoustic activity"
ARI_CLASS_HIGH_OFFSPRING_ACTIVITY = "High offspring acoustic activity"

COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE = "ARI_Female_Denominator_Confidence"

ARI_CONFIDENCE_NOT_EVALUATED = "Not evaluated"
ARI_CONFIDENCE_NO_FEMALE_DENOMINATOR = "No female denominator"
ARI_CONFIDENCE_LOW_FEMALE_DENOMINATOR = "Low female denominator"
ARI_CONFIDENCE_MODERATE_FEMALE_DENOMINATOR = "Moderate female denominator"
ARI_CONFIDENCE_STABLE_FEMALE_DENOMINATOR = "Stable female denominator"

LOW_FEMALE_DENOMINATOR_MAX = 5
MODERATE_FEMALE_DENOMINATOR_MAX = 20

# Manual / legacy outcome labels
OUTCOME_ABANDONED = "Abandoned"
OUTCOME_PARTIALLY_ABANDONED = "Partially Abandoned"
OUTCOME_SUCCESSFUL = "Successful"
OUTCOME_UNKNOWN = "Unknown"

# Breeding type labels
BREEDING_TYPE_SIMPLE = "Simple"
BREEDING_TYPE_SEQUENTIAL = "Sequential"
BREEDING_TYPE_COMPLEX = "Complex"
BREEDING_TYPE_UNKNOWN = "Unknown"
BREEDING_TYPE_ASYNCHRONOUS = "Asynchronous"

# Call type labels used for source-file discovery
CALL_TYPE_FEMALE = "Female"
CALL_TYPE_NESTLING = "Nestling"
CALL_TYPE_FLEDGLING = "Fledgling"
VALIDATED_PRESENT = "present"

# Common CSV column names
COL_APPROX_COLONY_SIZE = "Approx Colony Size"
COL_ARI = "ARI"
COL_ARI_CLASS = "ARI_Class"
COL_ARI_CLASS_THRESHOLD = "ARI_Class_Threshold"
COL_ARI_STATUS = "ARI_Status"
COL_AVG_FLEDGLING_CALLS_DAY = "Avg_Fledgling_Calls_Day"
COL_BREEDING_TYPE = "Breeding Type"
COL_CALL_TYPE = "Call_Type"
COL_COMMENT = "Comment"
COL_COMPLEX_TYPES = "Complex Types"
COL_DATE = "Date"
COL_DEPLOYMENT_END = "Deployment End"
COL_DEPLOYMENT_START = "Deployment Start"
COL_DETECTION_RATE = "Detection_Rate"
COL_DETECTION_RECORDINGS = "Detection_Recordings"
COL_EARLIEST_REC = "Earliest_Rec"
COL_FEMALE_DETECTION_RECORDINGS = "Female_Detection_Recordings"
COL_FLEDGLING_DAYS = "Fledgling_Days"
COL_FLEDGLING_DETECTION_DATES_IN_WINDOW = "Fledgling_Detection_Dates_In_Window",
COL_FLEDGLING_DETECTION_RECORDINGS = "Fledgling_Detection_Recordings"
COL_FLEDGLINGS_PRESENT = "Fledglings_Present"
COL_FLEDGLING_RAW_DETECTION_ROWS = "Fledgling_Raw_Detection_Rows",
COL_FLEDGLING_TOTAL_RECORDINGS = "Fledgling_Total_Recordings"
COL_FLEDGLING_WINDOW_DETECTION_ROWS = "Fledgling_Window_Detection_Rows",
COL_FLEDGLING_WINDOW_END = "Fledgling_Window_End"
COL_FLEDGLING_WINDOW_START = "Fledgling_Window_Start"
COL_HAD_DETECTIONS = "Had_Detections"
COL_HAD_RECORDINGS = "Had_Recordings"
COL_HATCH_DATE = "Hatch_Date"
COL_INCUBATION_DAYS = "Incubation_Days"
COL_LATEST_FLEDGLING_REC = "Latest_Fledgling_Rec"
COL_LATEST_REC = "Latest_Rec"
COL_NESTLING_DAYS = "Nestling_Days"
COL_NESTLING_DETECTION_RECORDINGS = "Nestling_Detection_Recordings"
COL_OUTCOME = "Outcome"
COL_PULSE_NAME = "Pulse Name"
COL_SITE_ID = "Site ID"
COL_SITE_NAME = "Site_Name"
COL_SUBSTRATE = "Substrate"
COL_TOTAL_RECORDINGS = "Total_Recordings"
COL_WINDOW_DAY = "Window_Day"
COL_WINDOW_END = "Window_End"
COL_WINDOW_START = "Window_Start"
COL_WINDOW_TYPE = "Window_Type"


COL_ARI_DIAGNOSTIC = "ARI_Diagnostic"
COL_ARI_WINDOW_FEMALE_START = "ARI_Window_Female_Start"
COL_ARI_WINDOW_FEMALE_END = "ARI_Window_Female_End"
COL_ARI_WINDOW_NESTLING_START = "ARI_Window_Nestling_Start"
COL_ARI_WINDOW_NESTLING_END = "ARI_Window_Nestling_End"
COL_ARI_RECORDING_START = "ARI_Recording_Start"
COL_ARI_RECORDING_STOP = "ARI_Recording_Stop"
COL_ARI_TOTAL_FEMALE_RECORDINGS = "ARI_Total_Female_Recordings"
COL_ARI_TOTAL_NESTLING_RECORDINGS = "ARI_Total_Nestling_Recordings"
COL_ARI_RAW_FEMALE_DETECTION_ROWS = "ARI_Raw_Female_Detection_Rows"
COL_ARI_WINDOW_FEMALE_DETECTION_ROWS = "ARI_Window_Female_Detection_Rows"
COL_ARI_RAW_NESTLING_DETECTION_ROWS = "ARI_Raw_Nestling_Detection_Rows"
COL_ARI_WINDOW_NESTLING_DETECTION_ROWS = "ARI_Window_Nestling_Detection_Rows"
COL_ARI_FEMALE_DETECTION_DATES_IN_WINDOW = "ARI_Female_Detection_Dates_In_Window"
COL_ARI_NESTLING_DETECTION_DATES_IN_WINDOW = "ARI_Nestling_Detection_Dates_In_Window"
COL_ARI_EARLIEST_RAW_FEMALE_DATE = "ARI_Earliest_Raw_Female_Date"
COL_ARI_LATEST_RAW_FEMALE_DATE = "ARI_Latest_Raw_Female_Date"
COL_ARI_EARLIEST_WINDOW_FEMALE_DATE = "ARI_Earliest_Window_Female_Date"
COL_ARI_LATEST_WINDOW_FEMALE_DATE = "ARI_Latest_Window_Female_Date"
COL_ARI_EARLIEST_RAW_NESTLING_DATE = "ARI_Earliest_Raw_Nestling_Date"
COL_ARI_LATEST_RAW_NESTLING_DATE = "ARI_Latest_Raw_Nestling_Date"
COL_ARI_EARLIEST_WINDOW_NESTLING_DATE = "ARI_Earliest_Window_Nestling_Date"
COL_ARI_LATEST_WINDOW_NESTLING_DATE = "ARI_Latest_Window_Nestling_Date"


# Backward-compatibility for older test/output names. Prefer
# COL_FLEDGLING_DETECTION_RECORDINGS in new code.
LEGACY_COL_TOTAL_FLEDGLING_CALLS = "Total_Fledgling_Calls"

NO_HATCH_VALUES = {
    "",
    ARI_STATUS_NHD,
    STATUS_ND,
    "nan",
    "NaN",
    "n/a",
    "N/A",
    "na",
    "NA",
    "inf",
    "missed",
}


PUBLICATION_COLUMNS = [
    COL_SITE_ID,
    COL_SITE_NAME,
    COL_PULSE_NAME,
    COL_OUTCOME,
    COL_ARI_STATUS,
    COL_ARI_CLASS,
    COL_ARI_CLASS_THRESHOLD,
    COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE,
    COL_BREEDING_TYPE,
    COL_COMPLEX_TYPES,
    COL_HATCH_DATE,
    COL_SUBSTRATE,
    COL_APPROX_COLONY_SIZE,
    COL_DEPLOYMENT_START,
    COL_DEPLOYMENT_END,

    COL_ARI_WINDOW_FEMALE_START,
    COL_ARI_WINDOW_FEMALE_END,
    COL_INCUBATION_DAYS,
    COL_ARI_TOTAL_FEMALE_RECORDINGS,
    COL_FEMALE_DETECTION_RECORDINGS,
    AVG_FEMALE_CALLS,

    COL_ARI_WINDOW_NESTLING_START,
    COL_ARI_WINDOW_NESTLING_END,
    COL_NESTLING_DAYS,
    COL_ARI_TOTAL_NESTLING_RECORDINGS,
    COL_NESTLING_DETECTION_RECORDINGS,
    AVG_NESTLING_CALLS,

    COL_ARI,

    COL_FLEDGLING_WINDOW_START,
    COL_FLEDGLING_WINDOW_END,
    COL_FLEDGLING_DAYS,
    COL_FLEDGLING_TOTAL_RECORDINGS,
    COL_FLEDGLING_DETECTION_RECORDINGS,
    COL_AVG_FLEDGLING_CALLS_DAY,
    COL_FLEDGLINGS_PRESENT,
    COL_LATEST_FLEDGLING_REC,
]

SOURCE_AUDIT_COLUMNS = [
    "Group",
    "Pretty Name",
    "Colony Size",
    "mcstart",
    "incstart",
    "fledgestart",
    "fledgedisp",
    "abandon",
    "partial abandon",
    "Source Row",
    "Review Status",
    "Review Notes",
]

METRIC_DIAGNOSTIC_COLUMNS = [
    COL_COMMENT,
    COL_ARI_DIAGNOSTIC,
    COL_EARLIEST_REC,
    COL_LATEST_REC,
    COL_ARI_RECORDING_START,
    COL_ARI_RECORDING_STOP,
    COL_ARI_RAW_FEMALE_DETECTION_ROWS,
    COL_ARI_WINDOW_FEMALE_DETECTION_ROWS,
    COL_ARI_RAW_NESTLING_DETECTION_ROWS,
    COL_ARI_WINDOW_NESTLING_DETECTION_ROWS,
    COL_ARI_FEMALE_DETECTION_DATES_IN_WINDOW,
    COL_ARI_NESTLING_DETECTION_DATES_IN_WINDOW,
    COL_ARI_EARLIEST_RAW_FEMALE_DATE,
    COL_ARI_LATEST_RAW_FEMALE_DATE,
    COL_ARI_EARLIEST_WINDOW_FEMALE_DATE,
    COL_ARI_LATEST_WINDOW_FEMALE_DATE,
    COL_ARI_EARLIEST_RAW_NESTLING_DATE,
    COL_ARI_LATEST_RAW_NESTLING_DATE,
    COL_ARI_EARLIEST_WINDOW_NESTLING_DATE,
    COL_ARI_LATEST_WINDOW_NESTLING_DATE,
    COL_FLEDGLING_RAW_DETECTION_ROWS,
    COL_FLEDGLING_WINDOW_DETECTION_ROWS,
    COL_FLEDGLING_DETECTION_DATES_IN_WINDOW,
]


# ==============================================================================
# DATA LOADING UTILITIES & FILTER HELPERS
# ==============================================================================
def filter_by_datetime_bounds(
    df: pd.DataFrame, 
    start_date: date, 
    end_date: date, 
    date_col: str = "date", 
    hour_col: str = "hour"
) -> pd.DataFrame:
    """Filters a DataFrame strictly to rows where the date falls within [start_date, end_date]
    and the hour matches our active monitoring window [START_HOUR, END_HOUR) exclusive.
    """
    if df.empty:
        return df
    date_mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    hour_mask = (df[hour_col] >= START_HOUR) & (df[hour_col] < END_HOUR)
    return df[date_mask & hour_mask]


def inclusive_day_span(start_date: date, end_date: date) -> int:
    """Returns the number of inclusive calendar days in a date window.

    If the end date is before the start date, the usable window is empty.
    """
    if end_date < start_date:
        return 0
    return (end_date - start_date).days + 1


def format_date_for_output(value: date | None, missing: str = STATUS_ND) -> str:
    """Formats date values consistently for CSV output."""
    if isinstance(value, date):
        return value.strftime("%m/%d/%Y")
    return missing


def summarize_unique_dates(df: pd.DataFrame, date_col: str = "date", max_dates: int = 12) -> str:
    """Returns a compact date summary for diagnostics."""
    if df.empty or date_col not in df.columns:
        return ""

    dates = sorted(d for d in df[date_col].dropna().unique() if isinstance(d, date))
    if not dates:
        return ""

    formatted = [d.strftime("%m/%d/%Y") for d in dates]
    if len(formatted) <= max_dates:
        return ", ".join(formatted)

    shown = formatted[:max_dates]
    return ", ".join(shown) + f", ... (+{len(formatted) - max_dates} more)"


def normalize_hatch_date_value(value: Any) -> str:
    """Normalizes no-hatch-date values to NHD while preserving usable date strings."""
    cleaned = str(value).replace("~", "").strip()
    return ARI_STATUS_NHD if cleaned in NO_HATCH_VALUES else cleaned


def classify_female_denominator_confidence(value: Any) -> str:
    """Classifies ARI denominator stability based on female detection recordings."""
    if pd.isna(value):
        return ARI_CONFIDENCE_NOT_EVALUATED

    try:
        count = int(value)
    except (TypeError, ValueError):
        return ARI_CONFIDENCE_NOT_EVALUATED

    if count <= 0:
        return ARI_CONFIDENCE_NO_FEMALE_DENOMINATOR
    if count <= LOW_FEMALE_DENOMINATOR_MAX:
        return ARI_CONFIDENCE_LOW_FEMALE_DENOMINATOR
    if count <= MODERATE_FEMALE_DENOMINATOR_MAX:
        return ARI_CONFIDENCE_MODERATE_FEMALE_DENOMINATOR
    return ARI_CONFIDENCE_STABLE_FEMALE_DENOMINATOR


@lru_cache(maxsize=1)
def load_recordings_parquet() -> pd.DataFrame:
    """Loads and caches the entire parquet file once at startup to optimize
    row-by-row processing speed. Standardizes column names and index structures.
    """
    if not RECORDINGS_PARQUET.exists():
        print(f"Warning: Parquet file not found at {RECORDINGS_PARQUET}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(RECORDINGS_PARQUET)
        df.columns = df.columns.str.lower().str.strip()
        
        # Determine the site column dynamically
        site_col = "site_name" if "site_name" in df.columns else "site"
        if site_col in df.columns:
            df["site_clean"] = df[site_col].astype(str).str.lower().str.strip()
        else:
            df["site_clean"] = ""
            
        # Parse date column safely
        if "date" in df.columns:
            df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            
        return df
    except Exception as e:
        print(f"Error loading recordings parquet: {e}")
        return pd.DataFrame()


def get_site_recording_bounds(site_name: str) -> tuple[date | None, date | None]:
    """Finds the absolute deployment start and stop dates for an ARU site from
    the master parquet database.
    """
    df = load_recordings_parquet()
    if df.empty:
        return None, None
        
    clean_target = site_name.lower().strip()
    df_site = df[df["site_clean"] == clean_target]
    if df_site.empty:
        return None, None
        
    all_dates = df_site["date_parsed"].dropna().unique()
    if len(all_dates) == 0:
        return None, None
        
    return min(all_dates), max(all_dates)


def get_total_recordings(site_name: str, start_date: date, end_date: date) -> int:
    """Calculates the total number of recordings made at a site between start_date
    and end_date (inclusive), limited to configured daylight hours.
    """
    df = load_recordings_parquet()
    if df.empty:
        return 0
        
    clean_target = site_name.lower().strip()
    df_site = df[df["site_clean"] == clean_target]
    if df_site.empty:
        return 0
        
    df_filtered = filter_by_datetime_bounds(
        df_site, start_date, end_date, date_col="date_parsed", hour_col="hour"
    )
    
    total_recs = df_filtered["n_recordings"].sum()
    return int(total_recs)


def get_daily_recordings_by_date(
    site_name: str,
    start_date: date,
    end_date: date,
) -> dict[date, int]:
    """Returns total daylight recordings per date for a site/window.

    Dates with no rows in the parquet will not appear in the returned dict.
    """
    df = load_recordings_parquet()
    if df.empty:
        return {}

    clean_target = site_name.lower().strip()
    df_site = df[df["site_clean"] == clean_target]
    if df_site.empty:
        return {}

    df_filtered = filter_by_datetime_bounds(
        df_site,
        start_date,
        end_date,
        date_col="date_parsed",
        hour_col="hour",
    )

    if df_filtered.empty:
        return {}

    daily = df_filtered.groupby("date_parsed")["n_recordings"].sum()
    return {d: int(v) for d, v in daily.items() if isinstance(d, date)}


def build_daily_detection_rows(
    source_row: dict[str, Any],
    site_name: str,
    hatch_date: date | None,
    window_type: str,
    call_type: str,
    start_date: date,
    end_date: date,
    detections_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Builds one diagnostic row per date in an analysis window.

    Detection counts are recording-level confirmed detections.
    Detection rate is detection recordings / total recordings for that date.
    """
    if end_date < start_date:
        return []

    daily_recordings = get_daily_recordings_by_date(site_name, start_date, end_date)

    if detections_df.empty:
        detections_in_window = pd.DataFrame(columns=["date", "hour"])
    else:
        detections_in_window = filter_by_datetime_bounds(detections_df, start_date, end_date)

    if detections_in_window.empty:
        daily_detections: dict[date, int] = {}
    else:
        counts = detections_in_window.groupby("date").size()
        daily_detections = {d: int(v) for d, v in counts.items() if isinstance(d, date)}

    rows: list[dict[str, Any]] = []
    span_days = inclusive_day_span(start_date, end_date)

    for day_index in range(span_days):
        current_date = start_date + timedelta(days=day_index)
        total_recordings = daily_recordings.get(current_date, 0)
        detection_recordings = daily_detections.get(current_date, 0)

        if total_recordings > 0:
            detection_rate: Any = round(
                detection_recordings / total_recordings,
                DECIMALS,
            )
            had_recordings = "Yes"
        else:
            detection_rate = pd.NA
            had_recordings = "No"

        rows.append(
            {
                COL_SITE_ID: source_row.get(COL_SITE_ID, ""),
                COL_SITE_NAME: site_name,
                COL_PULSE_NAME: source_row.get(COL_PULSE_NAME, ""),
                COL_OUTCOME: source_row.get(COL_OUTCOME, ""),
                COL_BREEDING_TYPE: source_row.get(COL_BREEDING_TYPE, ""),
                COL_COMPLEX_TYPES: source_row.get(COL_COMPLEX_TYPES, ""),
                COL_HATCH_DATE: format_date_for_output(hatch_date, missing=ARI_STATUS_NHD),
                COL_WINDOW_TYPE: window_type,
                COL_CALL_TYPE: call_type,
                COL_WINDOW_DAY: day_index + 1,
                COL_WINDOW_START: format_date_for_output(start_date),
                COL_WINDOW_END: format_date_for_output(end_date),
                COL_DATE: format_date_for_output(current_date),
                COL_TOTAL_RECORDINGS: total_recordings,
                COL_DETECTION_RECORDINGS: detection_recordings if total_recordings > 0 else pd.NA,
                COL_DETECTION_RATE: detection_rate,
                COL_HAD_RECORDINGS: had_recordings,
                COL_HAD_DETECTIONS: "Yes" if detection_recordings > 0 else "No",
            }
        )

    return rows


def get_recording_days_count(site_name: str, start_date: date, end_date: date) -> int:
    """Finds how many unique calendar days within the target window actually had
    recorded files on the ARU within configure daytime hours.
    """
    df = load_recordings_parquet()
    if df.empty:
        return 0
        
    clean_target = site_name.lower().strip()
    df_site = df[df["site_clean"] == clean_target]
    if df_site.empty:
        return 0
        
    df_filtered = filter_by_datetime_bounds(
        df_site, start_date, end_date, date_col="date_parsed", hour_col="hour"
    )
    
    return int(df_filtered["date_parsed"].nunique())


@lru_cache(maxsize=128)
def get_raw_validated_detections(site_name: str, call_type: str) -> pd.DataFrame:
    """Finds the raw recording detection log file for a site/call_type, filters
    for validated presence, cleans hour boundaries, and returns a sanitized
    DataFrame.
    """
    site_dir = PMJ_DIR / site_name
    pattern = f"*{escape(call_type)}*.csv"
    matching_files = list(site_dir.glob(pattern))

    if len(matching_files) != 1:
        print(f"Error for {site_name} {call_type}: {len(matching_files)} matching files were found.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(matching_files[0])
        df.columns = df.columns.str.lower().str.strip()
        
        # Filter for validated present detections
        df = df[df["validated"].astype(str).str.strip().str.lower() == VALIDATED_PRESENT].copy()
        if df.empty:
            return pd.DataFrame()
            
        df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce").dt.date
        
        # Safe hour type casting - simplified since column is guaranteed to exist
        df["hour"] = df["hour"].astype(int)
            
        return df[["date", "hour"]].copy()
    except Exception as e:
        print(f"Error reading detections for {site_name} ({call_type}): {e}")
        return pd.DataFrame()


# ==============================================================================
# EXTENSIBLE METRIC FRAMEWORK
# ==============================================================================
class AcousticMetric:
    def __init__(self) -> None:
        self.daily_diagnostic_rows: list[dict[str, Any]] = []

    def calculate_row(self, row: dict[str, Any], hatch_date: date | None, site_name: str) -> dict[str, Any]:
        return {}

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class AcousticReproductiveIndex(AcousticMetric):
    DAYS_TO_COUNT = 7
    NESTLING_OFFSET_DAYS = 4
    FEMALE_OFFSET_DAYS = 3
    def calculate_row(self, row: dict[str, Any], hatch_date: date | None, site_name: str) -> dict[str, Any]:
        result = {
            COL_EARLIEST_REC: STATUS_ND,
            COL_INCUBATION_DAYS: 0,
            COL_FEMALE_DETECTION_RECORDINGS: 0,
            AVG_FEMALE_CALLS: 0.0,
            COL_LATEST_REC: STATUS_ND,
            COL_NESTLING_DAYS: 0,
            COL_NESTLING_DETECTION_RECORDINGS: 0,
            AVG_NESTLING_CALLS: 0.0,
            COL_ARI: ARI_STATUS_MISSING_DATES,
            COL_COMMENT: "",

            # Diagnostic columns.
            COL_ARI_DIAGNOSTIC: "",
            COL_ARI_WINDOW_FEMALE_START: STATUS_ND,
            COL_ARI_WINDOW_FEMALE_END: STATUS_ND,
            COL_ARI_WINDOW_NESTLING_START: STATUS_ND,
            COL_ARI_WINDOW_NESTLING_END: STATUS_ND,
            COL_ARI_RECORDING_START: STATUS_ND,
            COL_ARI_RECORDING_STOP: STATUS_ND,
            COL_ARI_TOTAL_FEMALE_RECORDINGS: 0,
            COL_ARI_TOTAL_NESTLING_RECORDINGS: 0,
            COL_ARI_RAW_FEMALE_DETECTION_ROWS: 0,
            COL_ARI_WINDOW_FEMALE_DETECTION_ROWS: 0,
            COL_ARI_RAW_NESTLING_DETECTION_ROWS: 0,
            COL_ARI_WINDOW_NESTLING_DETECTION_ROWS: 0,
            COL_ARI_FEMALE_DETECTION_DATES_IN_WINDOW: "",
            COL_ARI_NESTLING_DETECTION_DATES_IN_WINDOW: "",
            COL_ARI_EARLIEST_RAW_FEMALE_DATE: STATUS_ND,
            COL_ARI_LATEST_RAW_FEMALE_DATE: STATUS_ND,
            COL_ARI_EARLIEST_WINDOW_FEMALE_DATE: STATUS_ND,
            COL_ARI_LATEST_WINDOW_FEMALE_DATE: STATUS_ND,
            COL_ARI_EARLIEST_RAW_NESTLING_DATE: STATUS_ND,
            COL_ARI_LATEST_RAW_NESTLING_DATE: STATUS_ND,
            COL_ARI_EARLIEST_WINDOW_NESTLING_DATE: STATUS_ND,
            COL_ARI_LATEST_WINDOW_NESTLING_DATE: STATUS_ND,
        }
        comments = []

        # Validate descriptive tracking variables
        breeding_type = str(row.get(COL_BREEDING_TYPE, "")).strip()
        complex_types = str(row.get(COL_COMPLEX_TYPES, "")).strip()
        
        invalid_breeding_type = False
        if breeding_type.lower() == "unknown":
            comments.append("Breeding Type is not valid: Unknown")
            invalid_breeding_type = True
        if breeding_type.lower() == "complex":
            comments.append("Breeding Type is not valid: Complex")
            invalid_breeding_type = True
        if "asynchronous" in complex_types.lower() or "asynchronous" in breeding_type.lower():
            comments.append("Breeding Type is not valid: Asynchronous")
            invalid_breeding_type = True

        if not hatch_date:
            comments.append("No valid hatch date")
            result[COL_EARLIEST_REC] = ARI_STATUS_NHD
            result[COL_LATEST_REC] = ARI_STATUS_NHD
            result[COL_ARI] = ARI_STATUS_NHD
            result[COL_COMMENT] = "; ".join(comments)
            return result

        # Fetch recording boundaries for the site
        rec_start, rec_stop = get_site_recording_bounds(site_name)
        if not rec_start or not rec_stop:
            comments.append("No recording deployment logs found in parquet")
            result[COL_COMMENT] = "; ".join(comments)
            return result
        #Diagnostics
        result[COL_ARI_RECORDING_START] = format_date_for_output(rec_start)
        result[COL_ARI_RECORDING_STOP] = format_date_for_output(rec_stop)

        # ----------------------------------------------------------------------
        # 1. Define Windows & Query Parquet / Detections
        # ----------------------------------------------------------------------
        # Female Window: up to 7 calendar days from hatch-3 through hatch-9,
        # clipped by deployment bounds.
        female_start_offset = self.DAYS_TO_COUNT + self.FEMALE_OFFSET_DAYS - 1
        f_start = max(hatch_date - timedelta(days=female_start_offset), rec_start)
        f_end = min(hatch_date - timedelta(days=self.FEMALE_OFFSET_DAYS), rec_stop)

        # Nestling Window: 7 calendar days from hatch+5 through hatch+11,
        # clipped by deployment stop.
        n_start = hatch_date + timedelta(days=self.NESTLING_OFFSET_DAYS)
        n_end = min(rec_stop, n_start + timedelta(days=self.DAYS_TO_COUNT - 1))

        #diagnostics
        result[COL_ARI_WINDOW_FEMALE_START] = format_date_for_output(f_start)
        result[COL_ARI_WINDOW_FEMALE_END] = format_date_for_output(f_end)
        result[COL_ARI_WINDOW_NESTLING_START] = format_date_for_output(n_start)
        result[COL_ARI_WINDOW_NESTLING_END] = format_date_for_output(n_end)

        # Span days are the inclusive analysis-window lengths, not unique detection days.
        inc_days_used = inclusive_day_span(f_start, f_end)
        nest_days_used = inclusive_day_span(n_start, n_end)

        result[COL_INCUBATION_DAYS] = inc_days_used
        result[COL_NESTLING_DAYS] = nest_days_used

        # Sum up total available recordings using centralized time mask helper.
        total_recs_during_incubation = get_total_recordings(site_name, f_start, f_end)
        total_recs_during_nestling = get_total_recordings(site_name, n_start, n_end)

        #diagnostics
        result[COL_ARI_TOTAL_FEMALE_RECORDINGS] = total_recs_during_incubation
        result[COL_ARI_TOTAL_NESTLING_RECORDINGS] = total_recs_during_nestling

        # Fetch raw validation detection logs
        df_f = get_raw_validated_detections(site_name, CALL_TYPE_FEMALE)
        df_n = get_raw_validated_detections(site_name, CALL_TYPE_NESTLING)

        self.daily_diagnostic_rows.extend(
            build_daily_detection_rows(
                source_row=row,
                site_name=site_name,
                hatch_date=hatch_date,
                window_type="Female_Incubation",
                call_type="Female",
                start_date=f_start,
                end_date=f_end,
                detections_df=df_f,
            )
        )

        self.daily_diagnostic_rows.extend(
            build_daily_detection_rows(
                source_row=row,
                site_name=site_name,
                hatch_date=hatch_date,
                window_type="Nestling",
                call_type="Nestling",
                start_date=n_start,
                end_date=n_end,
                detections_df=df_n,
            )
        )


        # ----------------------------------------------------------------------
        # 2. Female Incubation Processing
        # ----------------------------------------------------------------------
        if not df_f.empty:
            #diagnostics
            result[COL_ARI_RAW_FEMALE_DETECTION_ROWS] = len(df_f)
            raw_earliest_f = df_f["date"].min()
            raw_latest_f = df_f["date"].max()
            result[COL_ARI_EARLIEST_RAW_FEMALE_DATE] = format_date_for_output(raw_earliest_f)
            result[COL_ARI_LATEST_RAW_FEMALE_DATE] = format_date_for_output(raw_latest_f)


            # Filter detections strictly matching target date and hour parameter helper
            df_f_win = filter_by_datetime_bounds(df_f, f_start, f_end)

            #diagnostics
            result[COL_ARI_WINDOW_FEMALE_DETECTION_ROWS] = len(df_f_win)
            result[COL_ARI_FEMALE_DETECTION_DATES_IN_WINDOW] = summarize_unique_dates(df_f_win)
        
        
            if not df_f_win.empty:
                earliest_f = df_f_win["date"].min()
                latest_f = df_f_win["date"].max()

                #diagnostics
                result[COL_ARI_EARLIEST_WINDOW_FEMALE_DATE] = format_date_for_output(earliest_f)
                result[COL_ARI_LATEST_WINDOW_FEMALE_DATE] = format_date_for_output(latest_f)

                result[COL_EARLIEST_REC] = format_date_for_output(earliest_f)
                result[COL_FEMALE_DETECTION_RECORDINGS] = len(df_f_win)
                
                if total_recs_during_incubation > 0:
                    result[AVG_FEMALE_CALLS] = round(result[COL_FEMALE_DETECTION_RECORDINGS] / 
                                                     total_recs_during_incubation, DECIMALS)

        # ----------------------------------------------------------------------
        # 3. Nestling Validation Processing
        # ----------------------------------------------------------------------
        if not df_n.empty:
            # Filter nestling detections strictly matching target date and hour parameter helper
            df_n_win = filter_by_datetime_bounds(df_n, n_start, n_end)


            #diagnostics
            result[COL_ARI_RAW_NESTLING_DETECTION_ROWS] = len(df_n)
            raw_earliest_n = df_n["date"].min()
            raw_latest_n = df_n["date"].max()
            result[COL_ARI_EARLIEST_RAW_NESTLING_DATE] = format_date_for_output(raw_earliest_n)
            result[COL_ARI_LATEST_RAW_NESTLING_DATE] = format_date_for_output(raw_latest_n)
            result[COL_ARI_WINDOW_NESTLING_DETECTION_ROWS] = len(df_n_win)
            result[COL_ARI_NESTLING_DETECTION_DATES_IN_WINDOW] = summarize_unique_dates(df_n_win)


            if not df_n_win.empty:
                latest_n = df_n_win["date"].max()
                result[COL_LATEST_REC] = format_date_for_output(latest_n)
                result[COL_NESTLING_DETECTION_RECORDINGS] = len(df_n_win)                


                #diagnostics
                earliest_n = df_n_win["date"].min()
                result[COL_ARI_EARLIEST_WINDOW_NESTLING_DATE] = format_date_for_output(earliest_n)
                result[COL_ARI_LATEST_WINDOW_NESTLING_DATE] = format_date_for_output(latest_n)


                if total_recs_during_nestling > 0:
                    result[AVG_NESTLING_CALLS] = round(result[COL_NESTLING_DETECTION_RECORDINGS] / total_recs_during_nestling, DECIMALS)

        # ----------------------------------------------------------------------
        # 4. Biological Quality Guardrails & Outcome Scoring
        # ----------------------------------------------------------------------
        if inc_days_used < MIN_REQUIRED_DAYS:
            comments.append("Incubation days less than 4")
            result[COL_ARI] = ARI_STATUS_INSUFFICIENT_DAYS
            
        if nest_days_used < MIN_REQUIRED_DAYS:
            comments.append("Nestling days less than 4")
            result[COL_ARI] = ARI_STATUS_INSUFFICIENT_DAYS

        if invalid_breeding_type:
            result[COL_ARI] = ARI_STATUS_INVALID_BREEDING_TYPE
        elif result[COL_ARI] != ARI_STATUS_INSUFFICIENT_DAYS:
            if result[AVG_FEMALE_CALLS] == 0:
                result[COL_ARI] = ARI_STATUS_NO_FEMALE_CALLS
            else:
                result[COL_ARI] = round(result[AVG_NESTLING_CALLS] / result[AVG_FEMALE_CALLS], DECIMALS)


        #diagnostics
        if result[COL_ARI] == ARI_STATUS_NHD:
            result[COL_ARI_DIAGNOSTIC] = "No hatch date; ARI windows cannot be defined"
        elif result[COL_ARI] == ARI_STATUS_INVALID_BREEDING_TYPE:
            result[COL_ARI_DIAGNOSTIC] = (
                "Invalid breeding type; call dates/counts/rates calculated, "
                "but numeric ARI not calculated"
        )
        elif result[COL_ARI] == ARI_STATUS_INSUFFICIENT_DAYS:
            result[COL_ARI_DIAGNOSTIC] = (
                f"Insufficient span days: incubation={result['Incubation_Days']}, "
                f"nestling={result['Nestling_Days']}"
            )
        elif result[COL_ARI] == ARI_STATUS_NO_FEMALE_CALLS:
            result[COL_ARI_DIAGNOSTIC] = (
                f"No female detections in female window; "
                f"female_window={result['ARI_Window_Female_Start']} to {result['ARI_Window_Female_End']}; "
                f"raw_female_rows={result['ARI_Raw_Female_Detection_Rows']}"
            )
        elif isinstance(result[COL_ARI], float):
            result[COL_ARI_DIAGNOSTIC] = (
                f"ARI={result['ARI']} from "
                f"nestling_rate={result[AVG_NESTLING_CALLS]} / "
                f"female_rate={result[AVG_FEMALE_CALLS]}"
            )
        else:
            result[COL_ARI_DIAGNOSTIC] = str(result[COL_ARI])

        result[COL_COMMENT] = "; ".join(comments)
        return result

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_ari = pd.to_numeric(df[COL_ARI], errors="coerce")
        numeric_mask = numeric_ari.notna()

        # Use the current fledgling detection column, while still tolerating the
        # older column name in tests or older output files.
        if COL_FLEDGLING_DETECTION_RECORDINGS in df.columns:
            fledgling_calls = pd.to_numeric(
                df[COL_FLEDGLING_DETECTION_RECORDINGS], errors="coerce"
            ).fillna(0)
        elif LEGACY_COL_TOTAL_FLEDGLING_CALLS in df.columns:
            fledgling_calls = pd.to_numeric(
                df[LEGACY_COL_TOTAL_FLEDGLING_CALLS], errors="coerce"
            ).fillna(0)
        else:
            fledgling_calls = pd.Series(0, index=df.index, dtype="float64")

        zero_ari_with_fledglings = (
            numeric_mask
            & (numeric_ari == 0)
            & (fledgling_calls > 0)
        )

        # Publication-facing ARI status and descriptive class. These fields are
        # intentionally about the acoustic evidence, not a definitive colony fate.
        df[COL_ARI_STATUS] = df[COL_ARI].astype(str)
        df.loc[numeric_mask, COL_ARI_STATUS] = ARI_STATUS_NUMERIC

        df[COL_ARI_CLASS] = ARI_CLASS_NOT_SCORABLE
        df[COL_ARI_CLASS_THRESHOLD] = ARI_HIGH_THRESHOLD
        df.loc[
            numeric_mask & (numeric_ari == 0) & (fledgling_calls <= 0),
            COL_ARI_CLASS,
        ] = ARI_CLASS_NO_OFFSPRING_EVIDENCE
        df.loc[
            zero_ari_with_fledglings
            | (numeric_mask & (numeric_ari > 0) & (numeric_ari <= ARI_HIGH_THRESHOLD)),
            COL_ARI_CLASS,
        ] = ARI_CLASS_REDUCED_OFFSPRING_ACTIVITY
        df.loc[
            numeric_mask & (numeric_ari > ARI_HIGH_THRESHOLD),
            COL_ARI_CLASS,
        ] = ARI_CLASS_HIGH_OFFSPRING_ACTIVITY

        if COL_FEMALE_DETECTION_RECORDINGS in df.columns:
            female_detection_recordings = df[COL_FEMALE_DETECTION_RECORDINGS]
        else:
            female_detection_recordings = pd.Series(pd.NA, index=df.index)

        df[COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE] = female_detection_recordings.apply(
            classify_female_denominator_confidence
        )

        return df


class FledglingMetrics(AcousticMetric):
    FLEDGLING_OFFSET_DAYS = 9
    FLEDGLING_LATEST_DAY_OFFSET = FLEDGLING_OFFSET_DAYS + 7

    def calculate_row(self, row: dict[str, Any], hatch_date: date | None, site_name: str) -> dict[str, Any]:
        result = {
            COL_LATEST_FLEDGLING_REC: STATUS_ND,
            COL_FLEDGLING_DAYS: 0,
            COL_FLEDGLING_DETECTION_RECORDINGS: 0,
            COL_AVG_FLEDGLING_CALLS_DAY: 0.0,
            COL_FLEDGLINGS_PRESENT: "No",

            # Diagnostic columns.
            "Fledgling_Window_Start": STATUS_ND,
            "Fledgling_Window_End": STATUS_ND,
            "Fledgling_Total_Recordings": 0,
            "Fledgling_Raw_Detection_Rows": 0,
            "Fledgling_Window_Detection_Rows": 0,
            "Fledgling_Detection_Dates_In_Window": "",
        }

        if not hatch_date:
            result[COL_LATEST_FLEDGLING_REC] = ARI_STATUS_NHD
            return result

        rec_start, rec_stop = get_site_recording_bounds(site_name)
        if not rec_start or not rec_stop:
            return result

        fledge_start = hatch_date + timedelta(days=self.FLEDGLING_OFFSET_DAYS)
        fledge_end = min(rec_stop, hatch_date + timedelta(days=self.FLEDGLING_LATEST_DAY_OFFSET))

        #diagnostics
        result["Fledgling_Window_Start"] = format_date_for_output(fledge_start)
        result["Fledgling_Window_End"] = format_date_for_output(fledge_end)


        result[COL_FLEDGLING_DAYS] = inclusive_day_span(fledge_start, fledge_end)

        total_recs_during_fledging = get_total_recordings(site_name, fledge_start, fledge_end)


        #diagnostics
        result["Fledgling_Total_Recordings"] = total_recs_during_fledging

        df_fld = get_raw_validated_detections(site_name, CALL_TYPE_FLEDGLING)

        self.daily_diagnostic_rows.extend(
            build_daily_detection_rows(
                source_row=row,
                site_name=site_name,
                hatch_date=hatch_date,
                window_type="Fledgling",
                call_type="Fledgling",
                start_date=fledge_start,
                end_date=fledge_end,
                detections_df=df_fld,
            )
        )

        if not df_fld.empty:
            df_fld_win = filter_by_datetime_bounds(df_fld, fledge_start, fledge_end)

            #diagnostics
            result["Fledgling_Raw_Detection_Rows"] = len(df_fld)
            result["Fledgling_Window_Detection_Rows"] = len(df_fld_win)
            result["Fledgling_Detection_Dates_In_Window"] = summarize_unique_dates(df_fld_win)


            if not df_fld_win.empty:
                latest_f = df_fld_win["date"].max()
                result[COL_LATEST_FLEDGLING_REC] = format_date_for_output(latest_f)
                result[COL_FLEDGLING_DETECTION_RECORDINGS] = len(df_fld_win)
                result[COL_FLEDGLINGS_PRESENT] = "Yes"
                
                if total_recs_during_fledging > 0:
                    result[COL_AVG_FLEDGLING_CALLS_DAY] = round(
                        result[COL_FLEDGLING_DETECTION_RECORDINGS] / total_recs_during_fledging, 
                        DECIMALS)

        return result


# ==============================================================================
# MAIN RUNTIME PIPELINE
# ==============================================================================
def save_csv_with_retry(df: pd.DataFrame, path: Path, share = False) -> None:
    while True:
        try:
            df.to_csv(path, index=False)
            break
        except PermissionError:
            input(f"\n[!] Output file is locked in Excel: {path.name}\nClose it and press Enter to retry...")

    if share:
        if SHARING_OUTPUT_DIR.exists():
            shutil.copy2(path, SHARING_OUTPUT_DIR / path.name)


def main() -> None:
    if not BREEDING_DATES_CSV.exists():
        print(f"Error: Required file missing: {BREEDING_DATES_CSV}")
        return

    active_metrics: list[AcousticMetric] = [
        AcousticReproductiveIndex(),
        FledglingMetrics()
    ]

    source_df = pd.read_csv(BREEDING_DATES_CSV)
    source_df.columns = source_df.columns.astype(str).str.strip()
    
    if "Name" in source_df.columns:
        source_df.rename(columns={"Name": COL_SITE_NAME}, inplace=True)
    if "hatch" in source_df.columns:
        source_df.rename(columns={"hatch": COL_HATCH_DATE}, inplace=True)
    
    # Clean and normalize Hatch_Date values for output and downstream datetime parsing.
    source_df[COL_HATCH_DATE] = source_df[COL_HATCH_DATE].apply(normalize_hatch_date_value)

    processed_records = []
    records = source_df.to_dict("records")
    print(f"Processing {len(records)} records", end="", flush=True)

    for idx, raw_row in enumerate(records, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(records):
            print(".", end="", flush=True)

        row = cast(dict[str, Any], raw_row)
        out_row: dict[str, Any] = row.copy()
        
        site_name = str(row.get(COL_SITE_NAME, "")).strip()
        hatch_str = str(row.get(COL_HATCH_DATE, "")).strip()

        if hatch_str == ARI_STATUS_NHD:
            hatch_date = None
        else:
            try:
                hatch_date = pd.to_datetime(hatch_str).date()
            except Exception:
                hatch_date = None

        for metric in active_metrics:
            out_row.update(metric.calculate_row(row, hatch_date, site_name))
                
        processed_records.append(out_row)

    full_results_df = pd.DataFrame(processed_records)
    for metric in active_metrics:
        full_results_df = metric.post_process(full_results_df)

    print(" done.")

    print("Saving files...")
    publication_cols = [c for c in PUBLICATION_COLUMNS if c in full_results_df.columns]
    publication_df = full_results_df[publication_cols].copy()
    save_csv_with_retry(publication_df, OUT_FILE, share=True)
 
    numeric_mask = pd.to_numeric(full_results_df[COL_ARI], errors="coerce").notna()
    save_csv_with_retry(full_results_df[numeric_mask], OUT_FILTERED_FILE)

    comparison_mask = (
        ~full_results_df[COL_HATCH_DATE].astype(str).str.strip().isin(
            {ARI_STATUS_NHD, STATUS_ND, "inf", "missed", "n/a", "na", ""}
        )
        & ~full_results_df.get(COL_OUTCOME, pd.Series(dtype=str)).astype(str).isin({OUTCOME_UNKNOWN})
        & full_results_df.get(COL_BREEDING_TYPE, pd.Series(dtype=str))
            .astype(str)
            .str.strip()
            .isin({BREEDING_TYPE_SIMPLE, BREEDING_TYPE_SEQUENTIAL})
    )
    save_csv_with_retry(
        publication_df.loc[comparison_mask].reset_index(drop=True),
        OUT_ARI_FILE,
    )

    DIAGNOSTIC_COLUMNS = (
        PUBLICATION_COLUMNS
        + METRIC_DIAGNOSTIC_COLUMNS
        + SOURCE_AUDIT_COLUMNS
    )
    diagnostic_cols = [c for c in DIAGNOSTIC_COLUMNS if c in full_results_df.columns]
    diagnostic_df = full_results_df[diagnostic_cols].copy()
    save_csv_with_retry(diagnostic_df, OUT_DIAGNOSTIC_FILE)

    daily_diagnostic_rows: list[dict[str, Any]] = []
    for metric in active_metrics:
        daily_diagnostic_rows.extend(metric.daily_diagnostic_rows)

    if daily_diagnostic_rows:
        daily_diagnostics_df = pd.DataFrame(daily_diagnostic_rows)

        desired_daily_order = [
            COL_SITE_ID,
            COL_SITE_NAME,
            COL_PULSE_NAME,
            COL_OUTCOME,
            COL_BREEDING_TYPE,
            COL_COMPLEX_TYPES,
            COL_HATCH_DATE,
            COL_WINDOW_TYPE,
            COL_CALL_TYPE,
            COL_WINDOW_DAY,
            COL_WINDOW_START,
            COL_WINDOW_END,
            COL_DATE,
            COL_TOTAL_RECORDINGS,
            COL_DETECTION_RECORDINGS,
            COL_DETECTION_RATE,
            COL_HAD_RECORDINGS,
            COL_HAD_DETECTIONS,
        ]

        existing_daily_cols = [c for c in desired_daily_order if c in daily_diagnostics_df.columns]
        daily_remainder = [c for c in daily_diagnostics_df.columns if c not in existing_daily_cols]
        daily_diagnostics_df = daily_diagnostics_df[existing_daily_cols + daily_remainder]

        save_csv_with_retry(daily_diagnostics_df, OUT_DAILY_DIAGNOSTIC_FILE, share=True)
    
    log_text = (
        f"Processing Complete\n===================\n"
        f"Rows processed: {len(full_results_df)}\n"
        f"Outcomes:\n---------------------------\n"
        f"{full_results_df[COL_OUTCOME].value_counts().to_string()}\n\n"
        f"ARI classes:\n------------\n{full_results_df[COL_ARI_CLASS].value_counts().to_string()}\n"
    )
    RESULTS_TXT.write_text(log_text, encoding="utf-8")
    print(f"\nProcessing Complete. Metrics updated successfully.\n{log_text}")


if __name__ == "__main__":
    main()