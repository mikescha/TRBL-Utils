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

AVG_FEMALE_CALLS = "Avg_Female_Calls_Per_Rec"
AVG_NESTLING_CALLS = "Avg_Nestling_Calls_Per_Rec"


# Path to the specified parquet database for total recordings
RECORDINGS_PARQUET = DATA_DIR / "Data" / "recordings_per_day_hour.parquet"

DECIMALS = 3


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
        df = df[df["validated"].astype(str).str.strip().str.lower() == "present"].copy()
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
            "Comment": ""
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

        if invalid_breeding_type:
            result["ARI"] = "ND_INVALID_BREEDING_TYPE"
            result["Comment"] = "; ".join(comments)
            return result

        if not hatch_date:
            comments.append("No valid hatch date")
            result["Comment"] = "; ".join(comments)
            return result

        # Fetch recording boundaries for the site
        rec_start, rec_stop = get_site_recording_bounds(site_name)
        if not rec_start or not rec_stop:
            comments.append("No recording deployment logs found in parquet")
            result["Comment"] = "; ".join(comments)
            return result

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

        # Span days are the inclusive analysis-window lengths, not unique detection days.
        inc_days_used = inclusive_day_span(f_start, f_end)
        nest_days_used = inclusive_day_span(n_start, n_end)

        result["Incubation_Days"] = inc_days_used
        result["Nestling_Days"] = nest_days_used

        # Sum up total available recordings using centralized time mask helper.
        total_recs_during_incubation = get_total_recordings(site_name, f_start, f_end)
        total_recs_during_nestling = get_total_recordings(site_name, n_start, n_end)

        # Fetch raw validation detection logs
        df_f = get_raw_validated_detections(site_name, "Female")
        df_n = get_raw_validated_detections(site_name, "Nestling")

        # ----------------------------------------------------------------------
        # 2. Female Incubation Processing
        # ----------------------------------------------------------------------
        if not df_f.empty:
            # Filter detections strictly matching target date and hour parameter helper
            df_f_win = filter_by_datetime_bounds(df_f, f_start, f_end)
            if not df_f_win.empty:
                earliest_f = df_f_win["date"].min()
                result["Earliest_Rec"] = earliest_f.strftime("%m/%d/%Y") if isinstance(earliest_f, date) else str(earliest_f)
                result["Female_Detection_Recordings"] = len(df_f_win)
                
                if total_recs_during_incubation > 0:
                    result[AVG_FEMALE_CALLS] = round(result["Female_Detection_Recordings"] / total_recs_during_incubation, DECIMALS)

        # ----------------------------------------------------------------------
        # 3. Nestling Validation Processing
        # ----------------------------------------------------------------------
        if not df_n.empty:
            # Filter nestling detections strictly matching target date and hour parameter helper
            df_n_win = filter_by_datetime_bounds(df_n, n_start, n_end)
            if not df_n_win.empty:
                latest_n = df_n_win["date"].max()
                result["Latest_Rec"] = latest_n.strftime("%m/%d/%Y") if isinstance(latest_n, date) else str(latest_n)
                result["Nestling_Detection_Recordings"] = len(df_n_win)
                
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

        if result["ARI"] != "ND_INSUFFICIENT_DAYS":
            if result[AVG_FEMALE_CALLS] == 0:
                result["ARI"] = "ND_NO_FEMALE_CALLS"
            else:
                result["ARI"] = round(result[AVG_NESTLING_CALLS] / result[AVG_FEMALE_CALLS], DECIMALS)
        
        result["Comment"] = "; ".join(comments)
        return result

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_ari = pd.to_numeric(df["ARI"], errors="coerce")
        numeric_mask = numeric_ari.notna()

        valid_ari = df.loc[numeric_mask & (numeric_ari < 0.8), "ARI"].sort_values().to_numpy(dtype=float)

        cutoff = 0.15
        if len(valid_ari) >= 2:
            diffs = np.diff(valid_ari)
            max_gap_idx = int(np.argmax(diffs))
            cutoff = float(valid_ari[max_gap_idx] + (diffs[max_gap_idx] / 2))

        df["Calculated_Outcome"] = "Unknown"
        df.loc[numeric_mask & (numeric_ari == 0), "Calculated_Outcome"] = "Abandoned"
        df.loc[numeric_mask & (numeric_ari > 0) & (numeric_ari <= cutoff), "Calculated_Outcome"] = "Partially Abandoned"
        df.loc[numeric_mask & (numeric_ari > cutoff), "Calculated_Outcome"] = "Successful"
        
        self.calculated_cutoff = cutoff
        return df


class FledglingMetrics(AcousticMetric):
    FLEDGLING_OFFSET_DAYS = 11
    FLEDGLING_LATEST_DAY_OFFSET = 15

    def calculate_row(self, row: dict[str, Any], hatch_date: date | None, site_name: str) -> dict[str, Any]:
        result = {
            "Latest_Fledgling_Rec": "ND",
            "Fledgling_Days": 0,
            "Fledgling_Detection_Recordings": 0,
            "Avg_Fledgling_Calls_Day": 0.0,
            "Fledglings_Present": "No"
        }

        if not hatch_date:
            return result

        rec_start, rec_stop = get_site_recording_bounds(site_name)
        if not rec_start or not rec_stop:
            return result

        fledge_start = hatch_date + timedelta(days=self.FLEDGLING_OFFSET_DAYS)
        fledge_end = min(rec_stop, hatch_date + timedelta(days=self.FLEDGLING_LATEST_DAY_OFFSET))

        result["Fledgling_Days"] = inclusive_day_span(fledge_start, fledge_end)

        total_recs_during_fledging = get_total_recordings(site_name, fledge_start, fledge_end)
        df_fld = get_raw_validated_detections(site_name, "Fledgling")

        if not df_fld.empty:
            df_fld_win = filter_by_datetime_bounds(df_fld, fledge_start, fledge_end)
            if not df_fld_win.empty:
                latest_f = df_fld_win["date"].max()
                result["Latest_Fledgling_Rec"] = latest_f.strftime("%m/%d/%Y") if isinstance(latest_f, date) else str(latest_f)
                result["Fledgling_Detection_Recordings"] = len(df_fld_win)
                result["Fledglings_Present"] = "Yes"
                
                if total_recs_during_fledging > 0:
                    result["Avg_Fledgling_Calls_Day"] = round(result["Fledgling_Detection_Recordings"] / total_recs_during_fledging, DECIMALS)

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
    
    # Clean "~" out of Hatch_Date to enable seamless datetime math
    source_df["Hatch_Date"] = source_df["Hatch_Date"].astype(str).str.replace("~", "", regex=False).str.strip()

    processed_records = []

    records = source_df.to_dict("records")
    print(f"Processing {len(records)} records", end="", flush=True)

    for idx, raw_row in enumerate(records, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(records):
            print(".", end="", flush=True)

        row = cast(dict[str, Any], raw_row)
        out_row: dict[str, Any] = row.copy()
        
        site_name = str(row.get("Site_Name", "")).strip()
        hatch_str = str(row.get("Hatch_Date", ""))
        
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
        "Site ID", "Site_Name", "Pulse Name", "Outcome", "Calculated_Outcome", 
        "Breeding Type", "Hatch_Date", "Earliest_Rec", "Incubation_Days", 
        "Female_Detection_Recordings", AVG_FEMALE_CALLS, "Latest_Rec", "Nestling_Days", 
        "Nestling_Detection_Recordings", AVG_NESTLING_CALLS, "ARI", "Latest_Fledgling_Rec", 
        "Fledgling_Days", "Fledgling_Detection_Recordings", "Avg_Fledgling_Calls_Day", 
        "Fledglings_Present", "Substrate", "Approx Colony Size", "Comment"
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
        f"ARI Cutoff Threshold: {cutoff_val}\n\n"
        f"Outcomes:\n---------\n{results_df['Calculated_Outcome'].value_counts().to_string()}\n"
    )
    RESULTS_TXT.write_text(log_text, encoding="utf-8")
    print(f"\nProcessing Complete. Metrics updated successfully.\n{log_text}")


if __name__ == "__main__":
    main()