from __future__ import annotations

import shutil
from datetime import date, datetime, timedelta
from functools import lru_cache
from glob import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
MIN_REQUIRED_DAYS = 4

BASE_DIR = Path(".")
DATA_DIR = Path(r"C:\Users\mikes\OneDrive\Documents\GitHub\TRBLSummarizer\TRBLSummarizer")
PMJ_DIR = DATA_DIR / "PMJ Data"
SHARING_OUTPUT_DIR = Path(r"G:\My Drive\TRBL for Wendy GDrive")

BREEDING_DATES_CSV = BASE_DIR / "breeding dates.csv"
OUT_FILE = BASE_DIR / "nestling-to-female-ratios.csv"
OUT_FILTERED_FILE = BASE_DIR / "nestling-to-female-ratios-filtered.csv"
OUT_ARI_FILE = BASE_DIR / "nestling-to-female-ratios-for-comparison.csv"
RESULTS_TXT = BASE_DIR / "ratios_results.txt"

DECIMALS = 3

# ==============================================================================
# DATA LOADING UTILITIES
# ==============================================================================
@lru_cache(maxsize=128)
def get_daily_validated_counts(site_name: str, call_type: str) -> pd.Series:
    """
    Finds the raw recording logs for a site name/call_type, filters for validated 
    presence, and returns a Pandas Series of positive recordings indexed by Date.
    """
    # Using site_name to match filenames like '*2024 Baja Rancho Cinega Redonda 1*Female*.csv'
    safe_name = site_name.replace("?", "").replace("*", "")
    #BUG_FIX changed next 3 lines to find files correctly
    site_dir = PMJ_DIR / safe_name
    pattern = f"*{escape(call_type)}*.csv"
    matching_files = list(site_dir.glob(pattern))

    if not matching_files:
        return pd.Series(dtype=int)

    try:
        df = pd.read_csv(matching_files[0])
        df.columns = df.columns.str.lower().str.strip()
        df = df[df["validated"].astype(str).str.strip().str.lower() == "present"].copy()
        
        if df.empty:
            return pd.Series(dtype=int)

        df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce").dt.date
        return df.groupby("date").size()
        
    except Exception as e:
        print(f"Error reading {matching_files[0].name}: {e}")
        return pd.Series(dtype=int)

#Helper for date formatting for compatibility with last version
#Remove after equivalence is checked
def format_mdy(value):
    if isinstance(value, datetime):
        value = value.date()

    if isinstance(value, date):
        return f"{value.month}/{value.day}/{value.year}"

    return value

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
            "Total_Female_Calls": 0,
            "Avg_Female_Calls_Day": 0,
            "Latest_Rec": "ND",
            "Nestling_Days": 0,
            "Total_Nestling_Calls": 0,
            "Avg_Nestling_Calls_Day": 0,
            "ARI": "ND",
            "Comment": ""
        }
        
        comments = []

        # Enforce Breeding Type business logic validations
        breeding_type = str(row.get("Breeding Type", "")).strip()
        complex_types = str(row.get("Complex Types", "")).strip()
        
        if breeding_type.lower() == "unknown":
            comments.append("Breeding Type is not valid: Unknown")
        if breeding_type.lower() == "complex":
            comments.append("Breeding Type is not valid: Complex")
        if "asynchronous" in complex_types.lower() or "asynchronous" in breeding_type.lower():
            comments.append("Breeding Type is not valid: Asynchronous")

        if not hatch_date:
            comments.append("No valid hatch date")
            result["Comment"] = "; ".join(comments)
            return result

        # Define specific calendar windows based on biological offsets
        #BUG_FIX: changed how this is calculated to ensure we don't included days when the recorder wasn't running
        rec_start = datetime.strptime(row["Deployment Start"], "%m/%d/%Y").date()
        f_earliest = hatch_date - timedelta(days=self.DAYS_TO_COUNT)
        f_start = max(rec_start, f_earliest) #only count days where there was an active recording
        f_end = hatch_date - timedelta(days=1)
        
        #BUG_FIX: changed how this is calculated to ensure we don't included days when the recorder wasn't running
        rec_end = datetime.strptime(row["Deployment End"], "%m/%d/%Y").date()
        n_start = hatch_date + timedelta(days=self.NESTLING_OFFSET_DAYS)
        n_latest = n_start + timedelta(days=self.DAYS_TO_COUNT - 1)
        n_end = min(n_latest, rec_end)

        f_counts = get_daily_validated_counts(site_name, "Female")
        n_counts = get_daily_validated_counts(site_name, "Nestling")

        f_win = f_counts[(f_counts.index >= f_start) & (f_counts.index <= f_end)]
        n_win = n_counts[(n_counts.index >= n_start) & (n_counts.index <= n_end)]

        #BUG_FIX: changed how this is calculated and moved it outside logic below to 
        # ensure days with 0 detections are included
        result["Incubation_Days"] = (f_end - f_start).days + 1
        result["Nestling_Days"] = (n_end - n_start).days + 1

        # --- FEMALE LOGIC ---
        if not f_win.empty and f_win.sum() > 0:
            earliest_f = f_win.index.min()
            result["Earliest_Rec"] = format_mdy(earliest_f)
            result["Total_Female_Calls"] = int(f_win.sum())
            if result["Incubation_Days"] > 0:
                result["Avg_Female_Calls_Day"] = round(result["Total_Female_Calls"] / result["Incubation_Days"], DECIMALS)
        
        # --- NESTLING LOGIC ---
        if not n_win.empty and n_win.sum() > 0:
            latest_n = n_win.index.max()
            result["Latest_Rec"] = format_mdy(latest_n)
            result["Total_Nestling_Calls"] = int(n_win.sum())
            if result["Nestling_Days"] > 0:
                result["Avg_Nestling_Calls_Day"] = round(result["Total_Nestling_Calls"] / result["Nestling_Days"], DECIMALS)

        # --- ARI VALIDATION LOGIC ---
        if result["Incubation_Days"] < MIN_REQUIRED_DAYS:
            comments.append("Incubation days less than 4")
            result["ARI"] = "ND_INSUFFICIENT_DAYS"
        
        if result["Nestling_Days"] < MIN_REQUIRED_DAYS:
            comments.append("Nestling days less than 4")
            result["ARI"] = "ND_INSUFFICIENT_DAYS"

        if result["ARI"] != "ND_INSUFFICIENT_DAYS":
            if result["Avg_Female_Calls_Day"] == 0:
                result["ARI"] = "ND_DIV_ZERO" if result["Avg_Nestling_Calls_Day"] > 0 else 0
            else:
                result["ARI"] = round(result["Avg_Nestling_Calls_Day"] / result["Avg_Female_Calls_Day"], DECIMALS)

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
            "Total_Fledgling_Calls": 0,
            "Avg_Fledgling_Calls_Day": 0.0,
            "Fledglings_Present": "No"
        }

        if not hatch_date:
            return result

        fledge_start = hatch_date + timedelta(days=self.FLEDGLING_OFFSET_DAYS)
        fledge_end = hatch_date + timedelta(days=self.FLEDGLING_LATEST_DAY_OFFSET)

        f_counts = get_daily_validated_counts(site_name, "Fledgling")
        f_win = f_counts[(f_counts.index >= fledge_start) & (f_counts.index <= fledge_end)]

        if not f_win.empty and f_win.sum() > 0:
            latest_f = f_win.index.max()
            result["Latest_Fledgling_Rec"] = latest_f
            result["Fledgling_Days"] = (latest_f - fledge_start).days + 1
            result["Total_Fledgling_Calls"] = int(f_win.sum())
            result["Fledglings_Present"] = "Yes"
            
            if result["Fledgling_Days"] > 0:
                result["Avg_Fledgling_Calls_Day"] = round(result["Total_Fledgling_Calls"] / result["Fledgling_Days"], 4)

        return result


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def save_csv_with_retry(df: pd.DataFrame, path: Path) -> None:
    while True:
        try:
            df.to_csv(path, index=False)
            break
        except PermissionError:
            input(f"\n[!] File is open in Excel: {path.name}\nClose it and press Enter to retry...")


def main() -> None:
    if not BREEDING_DATES_CSV.exists():
        print(f"Error: {BREEDING_DATES_CSV} not found.")
        return

    active_metrics: list[AcousticMetric] = [
        AcousticReproductiveIndex(),
        FledglingMetrics()
    ]

    source_df = pd.read_csv(BREEDING_DATES_CSV)
    
    # 1. Column Renames & Pre-Cleaning
    if "Name" in source_df.columns:
        source_df.rename(columns={"Name": "Site_Name"}, inplace=True)
    if "hatch" in source_df.columns:
        source_df.rename(columns={"hatch": "Hatch_Date"}, inplace=True)
    
    # Clean "~" out of Hatch_Date so it propagates natively to output
    source_df["Hatch_Date"] = source_df["Hatch_Date"].astype(str).str.replace("~", "", regex=False).str.strip()

    processed_records = []

    # 2. Row-by-Row processing
    for row in source_df.to_dict("records"):
        out_row = row.copy()
        
        # Site_Name is now the driving key for filename globs
        site_name = str(row.get("Site_Name", "")).strip()
        hatch_str = str(row.get("Hatch_Date", ""))
        
        try:
            hatch_date = pd.to_datetime(hatch_str).date()
        except Exception:
            hatch_date = None

        for metric in active_metrics:
            out_row.update(metric.calculate_row(row, hatch_date, site_name))
                
        processed_records.append(out_row)

    # 3. Post-Processing & Output Schema Ordering
    results_df = pd.DataFrame(processed_records)
    for metric in active_metrics:
        results_df = metric.post_process(results_df)

    desired_order = [
        "Site ID", "Site_Name", "Pulse Name", "Outcome", "Calculated_Outcome", 
        "Breeding Type", "Hatch_Date", "Earliest_Rec", "Incubation_Days", 
        "Total_Female_Calls", "Avg_Female_Calls_Day", "Latest_Rec", "Nestling_Days", 
        "Total_Nestling_Calls", "Avg_Nestling_Calls_Day", "ARI", "Latest_Fledgling_Rec", 
        "Fledgling_Days", "Total_Fledgling_Calls", "Avg_Fledgling_Calls_Day", 
        "Fledglings_Present", "Substrate", "Approx Colony Size", "Comment"
    ]
    
    # Reorganize columns (Desired up front, remainder appended in existing order)
    existing_cols = [c for c in desired_order if c in results_df.columns]
    remainder = [c for c in results_df.columns if c not in existing_cols]
    results_df = results_df[existing_cols + remainder]

    # 4. Exporting Data
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

    print(f"Successfully processed {len(results_df)} records.")

if __name__ == "__main__":
    main()