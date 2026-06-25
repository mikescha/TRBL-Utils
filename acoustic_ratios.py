from __future__ import annotations

import shutil
from datetime import date, timedelta
from functools import lru_cache
from glob import escape
from pathlib import Path
from typing import Any, cast

import numpy as np
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

AVG_FEMALE_CALLS = "Avg_Female_Calls_Per_Rec"
AVG_NESTLING_CALLS = "Avg_Nestling_Calls_Per_Rec"


# Path to the specified parquet database for total recordings
RECORDINGS_PARQUET = DATA_DIR / "Data" / "recordings_per_day_hour.parquet"

DECIMALS = 3

# Biological / descriptive thresholds
ARI_HIGH_THRESHOLD = 0.5
LOW_FEMALE_DENOMINATOR_THRESHOLD = 5

# ARI status values
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
COL_SITE_ID = "Site ID"
COL_SITE_NAME = "Site_Name"
COL_PULSE_NAME = "Pulse Name"
COL_OUTCOME = "Outcome"
COL_CALCULATED_OUTCOME = "Calculated_Outcome"
COL_BREEDING_TYPE = "Breeding Type"
COL_COMPLEX_TYPES = "Complex Types"
COL_HATCH_DATE = "Hatch_Date"
COL_EARLIEST_REC = "Earliest_Rec"
COL_INCUBATION_DAYS = "Incubation_Days"
COL_FEMALE_DETECTION_RECORDINGS = "Female_Detection_Recordings"
COL_LATEST_REC = "Latest_Rec"
COL_NESTLING_DAYS = "Nestling_Days"
COL_NESTLING_DETECTION_RECORDINGS = "Nestling_Detection_Recordings"
COL_ARI = "ARI"
COL_ARI_STATUS = "ARI_Status"
COL_ARI_CLASS = "ARI_Class"
COL_ARI_CLASS_THRESHOLD = "ARI_Class_Threshold"
COL_ARI_FEMALE_DENOMINATOR_FLAG = "ARI_Female_Denominator_Flag"
COL_LATEST_FLEDGLING_REC = "Latest_Fledgling_Rec"
COL_FLEDGLING_DAYS = "Fledgling_Days"
COL_FLEDGLING_DETECTION_RECORDINGS = "Fledgling_Detection_Recordings"
COL_AVG_FLEDGLING_CALLS_DAY = "Avg_Fledgling_Calls_Day"
COL_FLEDGLINGS_PRESENT = "Fledglings_Present"
COL_SUBSTRATE = "Substrate"
COL_APPROX_COLONY_SIZE = "Approx Colony Size"
COL_COMMENT = "Comment"
COL_OUTCOME_MISMATCH = "Outcome_Mismatch"
COL_OUTCOME_MISMATCH_TYPE = "Outcome_Mismatch_Type"
COL_OUTCOME_DIAGNOSTIC = "Outcome_Diagnostic"

# Backward-compatibility for older test/output names. Prefer
# COL_FLEDGLING_DETECTION_RECORDINGS in new code.
LEGACY_COL_TOTAL_FLEDGLING_CALLS = "Total_Fledgling_Calls"

NO_HATCH_VALUES = {
    "",
    ARI_STATUS_NHD,
    "ND",
    "nan",
    "NaN",
    "n/a",
    "N/A",
    "na",
    "NA",
    "inf",
    "missed",
}


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

def format_date_for_output(value: date | None, missing: str = "ND") -> str:
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
    def calculate_row(self, row: dict[str, Any], hatch_date: date | None, site_name: str) -> dict[str, Any]:
        return {}

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class AcousticReproductiveIndex(AcousticMetric):
    DAYS_TO_COUNT = 10
    NESTLING_OFFSET_DAYS = 2
    def calculate_row(self, row: dict[str, Any], hatch_date: date | None, site_name: str) -> dict[str, Any]:
        result = {
            "Earliest_Rec": "ND",
            "Incubation_Days": 0,
            "Female_Detection_Recordings": 0,
            AVG_FEMALE_CALLS: 0.0,
            "Latest_Rec": "ND",
            "Nestling_Days": 0,
            "Nestling_Detection_Recordings": 0,
            AVG_NESTLING_CALLS: 0.0,
            "ARI": "ND_MISSING_DATES",
            "Comment": "",

            # Diagnostic columns.
            "ARI_Diagnostic": "",
            "ARI_Window_Female_Start": "ND",
            "ARI_Window_Female_End": "ND",
            "ARI_Window_Nestling_Start": "ND",
            "ARI_Window_Nestling_End": "ND",
            "ARI_Recording_Start": "ND",
            "ARI_Recording_Stop": "ND",
            "ARI_Total_Female_Recordings": 0,
            "ARI_Total_Nestling_Recordings": 0,
            "ARI_Raw_Female_Detection_Rows": 0,
            "ARI_Window_Female_Detection_Rows": 0,
            "ARI_Raw_Nestling_Detection_Rows": 0,
            "ARI_Window_Nestling_Detection_Rows": 0,
            "ARI_Female_Detection_Dates_In_Window": "",
            "ARI_Nestling_Detection_Dates_In_Window": "",
            "ARI_Earliest_Raw_Female_Date": "ND",
            "ARI_Latest_Raw_Female_Date": "ND",
            "ARI_Earliest_Window_Female_Date": "ND",
            "ARI_Latest_Window_Female_Date": "ND",
            "ARI_Earliest_Raw_Nestling_Date": "ND",
            "ARI_Latest_Raw_Nestling_Date": "ND",
            "ARI_Earliest_Window_Nestling_Date": "ND",
            "ARI_Latest_Window_Nestling_Date": "ND",
        }

        comments = []

        # Validate descriptive tracking variables
        breeding_type = str(row.get("Breeding Type", "")).strip()
        complex_types = str(row.get("Complex Types", "")).strip()
        
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
            result["Earliest_Rec"] = "NHD"
            result["Latest_Rec"] = "NHD"
            result["ARI"] = "NHD"
            result["Comment"] = "; ".join(comments)
            return result

        # Fetch recording boundaries for the site
        rec_start, rec_stop = get_site_recording_bounds(site_name)
        if not rec_start or not rec_stop:
            comments.append("No recording deployment logs found in parquet")
            result["Comment"] = "; ".join(comments)
            return result
        #Diagnostics
        result["ARI_Recording_Start"] = format_date_for_output(rec_start)
        result["ARI_Recording_Stop"] = format_date_for_output(rec_stop)

        # ----------------------------------------------------------------------
        # 1. Define Windows & Query Parquet / Detections
        # ----------------------------------------------------------------------
        # Female Window: up to 10 calendar days ending the day before hatch,
        # clipped by deployment bounds.
        f_start = max(hatch_date - timedelta(days=self.DAYS_TO_COUNT), rec_start)
        f_end = min(hatch_date - timedelta(days=1), rec_stop)

        # Nestling Window: 10 calendar days from hatch+2 through hatch+11,
        # clipped by deployment stop.
        n_start = hatch_date + timedelta(days=self.NESTLING_OFFSET_DAYS)
        n_end = min(rec_stop, n_start + timedelta(days=self.DAYS_TO_COUNT - 1))

        #diagnostics
        result["ARI_Window_Female_Start"] = format_date_for_output(f_start)
        result["ARI_Window_Female_End"] = format_date_for_output(f_end)
        result["ARI_Window_Nestling_Start"] = format_date_for_output(n_start)
        result["ARI_Window_Nestling_End"] = format_date_for_output(n_end)

        # Span days are the inclusive analysis-window lengths, not unique detection days.
        inc_days_used = inclusive_day_span(f_start, f_end)
        nest_days_used = inclusive_day_span(n_start, n_end)

        result["Incubation_Days"] = inc_days_used
        result["Nestling_Days"] = nest_days_used

        # Sum up total available recordings using centralized time mask helper.
        total_recs_during_incubation = get_total_recordings(site_name, f_start, f_end)
        total_recs_during_nestling = get_total_recordings(site_name, n_start, n_end)

        #diagnostics
        result["ARI_Total_Female_Recordings"] = total_recs_during_incubation
        result["ARI_Total_Nestling_Recordings"] = total_recs_during_nestling

        # Fetch raw validation detection logs
        df_f = get_raw_validated_detections(site_name, CALL_TYPE_FEMALE)
        df_n = get_raw_validated_detections(site_name, CALL_TYPE_NESTLING)

        # ----------------------------------------------------------------------
        # 2. Female Incubation Processing
        # ----------------------------------------------------------------------
        if not df_f.empty:
            #diagnostics
            result["ARI_Raw_Female_Detection_Rows"] = len(df_f)
            raw_earliest_f = df_f["date"].min()
            raw_latest_f = df_f["date"].max()
            result["ARI_Earliest_Raw_Female_Date"] = format_date_for_output(raw_earliest_f)
            result["ARI_Latest_Raw_Female_Date"] = format_date_for_output(raw_latest_f)


            # Filter detections strictly matching target date and hour parameter helper
            df_f_win = filter_by_datetime_bounds(df_f, f_start, f_end)

            #diagnostics
            result["ARI_Window_Female_Detection_Rows"] = len(df_f_win)
            result["ARI_Female_Detection_Dates_In_Window"] = summarize_unique_dates(df_f_win)
        
        
            if not df_f_win.empty:
                earliest_f = df_f_win["date"].min()
                latest_f = df_f_win["date"].max()

                #diagnostics
                result["ARI_Earliest_Window_Female_Date"] = format_date_for_output(earliest_f)
                result["ARI_Latest_Window_Female_Date"] = format_date_for_output(latest_f)

                result["Earliest_Rec"] = format_date_for_output(earliest_f)
                result["Female_Detection_Recordings"] = len(df_f_win)
                
                if total_recs_during_incubation > 0:
                    result[AVG_FEMALE_CALLS] = round(result["Female_Detection_Recordings"] / 
                                                     total_recs_during_incubation, DECIMALS)

        # ----------------------------------------------------------------------
        # 3. Nestling Validation Processing
        # ----------------------------------------------------------------------
        if not df_n.empty:
            # Filter nestling detections strictly matching target date and hour parameter helper
            df_n_win = filter_by_datetime_bounds(df_n, n_start, n_end)


            #diagnostics
            result["ARI_Raw_Nestling_Detection_Rows"] = len(df_n)
            raw_earliest_n = df_n["date"].min()
            raw_latest_n = df_n["date"].max()
            result["ARI_Earliest_Raw_Nestling_Date"] = format_date_for_output(raw_earliest_n)
            result["ARI_Latest_Raw_Nestling_Date"] = format_date_for_output(raw_latest_n)
            result["ARI_Window_Nestling_Detection_Rows"] = len(df_n_win)
            result["ARI_Nestling_Detection_Dates_In_Window"] = summarize_unique_dates(df_n_win)


            if not df_n_win.empty:
                latest_n = df_n_win["date"].max()
                result["Latest_Rec"] = format_date_for_output(latest_n)
                result["Nestling_Detection_Recordings"] = len(df_n_win)                


                #diagnostics
                earliest_n = df_n_win["date"].min()
                result["ARI_Earliest_Window_Nestling_Date"] = format_date_for_output(earliest_n)
                result["ARI_Latest_Window_Nestling_Date"] = format_date_for_output(latest_n)


                if total_recs_during_nestling > 0:
                    result[AVG_NESTLING_CALLS] = round(result["Nestling_Detection_Recordings"] / total_recs_during_nestling, DECIMALS)

        # ----------------------------------------------------------------------
        # 4. Biological Quality Guardrails & Outcome Scoring
        # ----------------------------------------------------------------------
        if inc_days_used < MIN_REQUIRED_DAYS:
            comments.append("Incubation days less than 4")
            result["ARI"] = "ND_INSUFFICIENT_DAYS"
            
        if nest_days_used < MIN_REQUIRED_DAYS:
            comments.append("Nestling days less than 4")
            result["ARI"] = "ND_INSUFFICIENT_DAYS"

        if invalid_breeding_type:
            result["ARI"] = "ND_INVALID_BREEDING_TYPE"
        elif result["ARI"] != "ND_INSUFFICIENT_DAYS":
            if result[AVG_FEMALE_CALLS] == 0:
                result["ARI"] = "ND_NO_FEMALE_CALLS"
            else:
                result["ARI"] = round(result[AVG_NESTLING_CALLS] / result[AVG_FEMALE_CALLS], DECIMALS)


        #diagnostics
        if result["ARI"] == "NHD":
            result["ARI_Diagnostic"] = "No hatch date; ARI windows cannot be defined"
        elif result["ARI"] == "ND_INVALID_BREEDING_TYPE":
            result["ARI_Diagnostic"] = (
                "Invalid breeding type; call dates/counts/rates calculated, "
                "but numeric ARI not calculated"
        )
        elif result["ARI"] == "ND_INSUFFICIENT_DAYS":
            result["ARI_Diagnostic"] = (
                f"Insufficient span days: incubation={result['Incubation_Days']}, "
                f"nestling={result['Nestling_Days']}"
            )
        elif result["ARI"] == "ND_NO_FEMALE_CALLS":
            result["ARI_Diagnostic"] = (
                f"No female detections in female window; "
                f"female_window={result['ARI_Window_Female_Start']} to {result['ARI_Window_Female_End']}; "
                f"raw_female_rows={result['ARI_Raw_Female_Detection_Rows']}"
            )
        elif isinstance(result["ARI"], float):
            result["ARI_Diagnostic"] = (
                f"ARI={result['ARI']} from "
                f"nestling_rate={result[AVG_NESTLING_CALLS]} / "
                f"female_rate={result[AVG_FEMALE_CALLS]}"
            )
        else:
            result["ARI_Diagnostic"] = str(result["ARI"])

        result["Comment"] = "; ".join(comments)
        return result

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_ari = pd.to_numeric(df[COL_ARI], errors="coerce")
        numeric_mask = numeric_ari.notna()

        valid_ari = (
            df.loc[numeric_mask & (numeric_ari < 0.8), COL_ARI]
            .sort_values()
            .to_numpy(dtype=float)
        )

        cutoff = 0.15
        if len(valid_ari) >= 2:
            diffs = np.diff(valid_ari)
            max_gap_idx = int(np.argmax(diffs))
            cutoff = float(valid_ari[max_gap_idx] + (diffs[max_gap_idx] / 2))

        # Legacy dynamic outcome classification retained for backward compatibility.
        df[COL_CALCULATED_OUTCOME] = OUTCOME_UNKNOWN
        df.loc[numeric_mask & (numeric_ari == 0), COL_CALCULATED_OUTCOME] = OUTCOME_ABANDONED
        df.loc[
            numeric_mask & (numeric_ari > 0) & (numeric_ari <= cutoff),
            COL_CALCULATED_OUTCOME,
        ] = OUTCOME_PARTIALLY_ABANDONED
        df.loc[numeric_mask & (numeric_ari > cutoff), COL_CALCULATED_OUTCOME] = OUTCOME_SUCCESSFUL

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

        df.loc[zero_ari_with_fledglings, COL_CALCULATED_OUTCOME] = OUTCOME_PARTIALLY_ABANDONED

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
            female_detection_recordings = pd.to_numeric(
                df[COL_FEMALE_DETECTION_RECORDINGS], errors="coerce"
            ).fillna(0)
        else:
            female_detection_recordings = pd.Series(0, index=df.index, dtype="float64")

        df[COL_ARI_FEMALE_DENOMINATOR_FLAG] = (
            numeric_mask
            & (female_detection_recordings > 0)
            & (female_detection_recordings <= LOW_FEMALE_DENOMINATOR_THRESHOLD)
        )

        # Diagnostics for the legacy dynamic cutoff.
        df["ARI_Cutoff"] = cutoff
        df["ARI_Margin_To_Cutoff"] = pd.NA

        df.loc[numeric_mask, "ARI_Margin_To_Cutoff"] = (
            numeric_ari.loc[numeric_mask] - cutoff
        ).round(DECIMALS)

        if COL_OUTCOME in df.columns:
            human_outcome = df[COL_OUTCOME].astype(str).str.strip()
            calculated_outcome = df[COL_CALCULATED_OUTCOME].astype(str).str.strip()

            df[COL_OUTCOME_MISMATCH] = (
                human_outcome.ne("")
                & human_outcome.ne("nan")
                & human_outcome.ne(OUTCOME_UNKNOWN)
                & calculated_outcome.ne(OUTCOME_UNKNOWN)
                & human_outcome.ne(calculated_outcome)
            )

            df[COL_OUTCOME_MISMATCH_TYPE] = ""
            df.loc[
                df[COL_OUTCOME_MISMATCH],
                COL_OUTCOME_MISMATCH_TYPE,
            ] = human_outcome + " -> " + calculated_outcome

            df[COL_OUTCOME_DIAGNOSTIC] = ""

            df.loc[
                zero_ari_with_fledglings,
                COL_OUTCOME_DIAGNOSTIC,
            ] = (
                "ARI was zero because no nestling detections were found, but fledgling "
                "detections were present in the fledgling window; classified as "
                "Partially Abandoned in the legacy Calculated_Outcome field"
            )

            df.loc[
                df[COL_OUTCOME_MISMATCH]
                & human_outcome.eq(OUTCOME_SUCCESSFUL)
                & calculated_outcome.eq(OUTCOME_PARTIALLY_ABANDONED)
                & df[COL_OUTCOME_DIAGNOSTIC].eq(""),
                COL_OUTCOME_DIAGNOSTIC,
            ] = (
                "Human Successful but ARI below dynamic cutoff; inspect NBC detections, "
                "female denominator, fledgling detections, and whether the legacy dynamic "
                "cutoff is biologically appropriate"
            )
        else:
            df[COL_OUTCOME_MISMATCH] = False
            df[COL_OUTCOME_MISMATCH_TYPE] = ""
            df[COL_OUTCOME_DIAGNOSTIC] = ""

        self.calculated_cutoff = cutoff
        return df


class FledglingMetrics(AcousticMetric):
    FLEDGLING_OFFSET_DAYS = 9
    FLEDGLING_LATEST_DAY_OFFSET = FLEDGLING_OFFSET_DAYS + 7

    def calculate_row(self, row: dict[str, Any], hatch_date: date | None, site_name: str) -> dict[str, Any]:
        result = {
            "Latest_Fledgling_Rec": "ND",
            "Fledgling_Days": 0,
            "Fledgling_Detection_Recordings": 0,
            "Avg_Fledgling_Calls_Day": 0.0,
            "Fledglings_Present": "No",

            # Diagnostic columns.
            "Fledgling_Window_Start": "ND",
            "Fledgling_Window_End": "ND",
            "Fledgling_Total_Recordings": 0,
            "Fledgling_Raw_Detection_Rows": 0,
            "Fledgling_Window_Detection_Rows": 0,
            "Fledgling_Detection_Dates_In_Window": "",
        }

        if not hatch_date:
            result["Latest_Fledgling_Rec"] = "NHD"
            return result

        rec_start, rec_stop = get_site_recording_bounds(site_name)
        if not rec_start or not rec_stop:
            return result

        fledge_start = hatch_date + timedelta(days=self.FLEDGLING_OFFSET_DAYS)
        fledge_end = min(rec_stop, hatch_date + timedelta(days=self.FLEDGLING_LATEST_DAY_OFFSET))

        #diagnostics
        result["Fledgling_Window_Start"] = format_date_for_output(fledge_start)
        result["Fledgling_Window_End"] = format_date_for_output(fledge_end)


        result["Fledgling_Days"] = inclusive_day_span(fledge_start, fledge_end)

        total_recs_during_fledging = get_total_recordings(site_name, fledge_start, fledge_end)


        #diagnostics
        result["Fledgling_Total_Recordings"] = total_recs_during_fledging

        df_fld = get_raw_validated_detections(site_name, CALL_TYPE_FLEDGLING)
        if not df_fld.empty:
            df_fld_win = filter_by_datetime_bounds(df_fld, fledge_start, fledge_end)

            #diagnostics
            result["Fledgling_Raw_Detection_Rows"] = len(df_fld)
            result["Fledgling_Window_Detection_Rows"] = len(df_fld_win)
            result["Fledgling_Detection_Dates_In_Window"] = summarize_unique_dates(df_fld_win)


            if not df_fld_win.empty:
                latest_f = df_fld_win["date"].max()
                result["Latest_Fledgling_Rec"] = format_date_for_output(latest_f)
                result["Fledgling_Detection_Recordings"] = len(df_fld_win)
                result["Fledglings_Present"] = "Yes"
                
                if total_recs_during_fledging > 0:
                    result["Avg_Fledgling_Calls_Day"] = round(
                        result["Fledgling_Detection_Recordings"] / total_recs_during_fledging, 
                        DECIMALS)

        return result


# ==============================================================================
# MAIN RUNTIME PIPELINE
# ==============================================================================
def save_csv_with_retry(df: pd.DataFrame, path: Path) -> None:
    while True:
        try:
            df.to_csv(path, index=False)
            break
        except PermissionError:
            input(f"\n[!] Output file is locked in Excel: {path.name}\nClose it and press Enter to retry...")


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
        source_df.rename(columns={"Name": "Site_Name"}, inplace=True)
    if "hatch" in source_df.columns:
        source_df.rename(columns={"hatch": "Hatch_Date"}, inplace=True)
    
    # Clean and normalize Hatch_Date values for output and downstream datetime parsing.
    source_df["Hatch_Date"] = source_df["Hatch_Date"].apply(normalize_hatch_date_value)

    processed_records = []
    records = source_df.to_dict("records")
    print(f"Processing {len(records)} records", end="", flush=True)

    for idx, raw_row in enumerate(records, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(records):
            print(".", end="", flush=True)

        row = cast(dict[str, Any], raw_row)
        out_row: dict[str, Any] = row.copy()
        
        site_name = str(row.get("Site_Name", "")).strip()
        hatch_str = str(row.get("Hatch_Date", "")).strip()

        if hatch_str == "NHD":
            hatch_date = None
        else:
            try:
                hatch_date = pd.to_datetime(hatch_str).date()
            except Exception:
                hatch_date = None

        for metric in active_metrics:
            out_row.update(metric.calculate_row(row, hatch_date, site_name))
                
        processed_records.append(out_row)

    results_df = pd.DataFrame(processed_records)
    for metric in active_metrics:
        results_df = metric.post_process(results_df)

    print(" done.")

    print("Saving files...")
    # Establish sequence ordering
    desired_order = [
        COL_SITE_ID, COL_SITE_NAME, COL_PULSE_NAME, COL_OUTCOME, COL_CALCULATED_OUTCOME,
        COL_ARI_STATUS, COL_ARI_CLASS, COL_ARI_CLASS_THRESHOLD, COL_ARI_FEMALE_DENOMINATOR_FLAG,
        COL_BREEDING_TYPE, COL_HATCH_DATE, COL_EARLIEST_REC, COL_INCUBATION_DAYS,
        COL_FEMALE_DETECTION_RECORDINGS, AVG_FEMALE_CALLS, COL_LATEST_REC, COL_NESTLING_DAYS,
        COL_NESTLING_DETECTION_RECORDINGS, AVG_NESTLING_CALLS, COL_ARI, COL_LATEST_FLEDGLING_REC,
        COL_FLEDGLING_DAYS, COL_FLEDGLING_DETECTION_RECORDINGS, COL_AVG_FLEDGLING_CALLS_DAY,
        COL_FLEDGLINGS_PRESENT, COL_SUBSTRATE, COL_APPROX_COLONY_SIZE, COL_COMMENT,
    ]
    
    existing_cols = [c for c in desired_order if c in results_df.columns]
    remainder = [c for c in results_df.columns if c not in existing_cols]
    results_df = results_df[existing_cols + remainder]

    save_csv_with_retry(results_df, OUT_FILE)
    
    if SHARING_OUTPUT_DIR.exists():
        shutil.copy2(OUT_FILE, SHARING_OUTPUT_DIR / OUT_FILE.name)

    numeric_mask = pd.to_numeric(results_df["ARI"], errors="coerce").notna()
    save_csv_with_retry(results_df[numeric_mask], OUT_FILTERED_FILE)

    comparison_mask = (
        ~results_df["Hatch_Date"].astype(str).str.strip().isin({"NHD", "ND", "inf", "missed", "n/a", "na", ""}) &
        ~results_df.get("Outcome", pd.Series(dtype=str)).astype(str).isin({"Unknown"}) &
        results_df.get("Breeding Type", pd.Series(dtype=str)).astype(str).str.strip().isin({"Simple", "Sequential"})
    )
    save_csv_with_retry(results_df.loc[comparison_mask].reset_index(drop=True), OUT_ARI_FILE)

    # Calculate dynamic post-process metrics for text logs
    ari_instance = next((m for m in active_metrics if isinstance(m, AcousticReproductiveIndex)), None)
    cutoff_val = getattr(ari_instance, "calculated_cutoff", "N/A")

    log_text = (
        f"Processing Complete\n===================\n"
        f"Rows processed: {len(results_df)}\n"
        f"ARI Dynamic Cutoff Threshold: {cutoff_val}\n"
        f"ARI Class Threshold: {ARI_HIGH_THRESHOLD}\n\n"
        f"Legacy calculated outcomes:\n---------------------------\n"
        f"{results_df[COL_CALCULATED_OUTCOME].value_counts().to_string()}\n\n"
        f"ARI classes:\n------------\n{results_df[COL_ARI_CLASS].value_counts().to_string()}\n"
    )
    RESULTS_TXT.write_text(log_text, encoding="utf-8")
    print(f"\nProcessing Complete. Metrics updated successfully.\n{log_text}")


    diagnostic_cols = [
        col for col in results_df.columns
        if (
            col.startswith("ARI_")
            or col.startswith("Fledgling_")
            or col in {
                "Site ID",
                "Site_Name",
                "Pulse Name",
                "Outcome",
                "Calculated_Outcome",
                "Outcome_Mismatch",
                "Outcome_Mismatch_Type",
                "Outcome_Diagnostic",
                "Breeding Type",
                "Complex Types",
                "Hatch_Date",
                "Earliest_Rec",
                "Latest_Rec",
                "Incubation_Days",
                "Nestling_Days",
                "Female_Detection_Recordings",
                "Nestling_Detection_Recordings",
                AVG_FEMALE_CALLS,
                AVG_NESTLING_CALLS,
                "ARI",
                "Latest_Fledgling_Rec",
                "Fledglings_Present",
                "Fledgling_Detection_Recordings",
                "Avg_Fledgling_Calls_Day",
                "Comment",
            }
        )
    ]

    save_csv_with_retry(results_df[diagnostic_cols], OUT_DIAGNOSTIC_FILE)


if __name__ == "__main__":
    main()