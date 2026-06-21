from __future__ import annotations

import shutil
from datetime import date, timedelta
from functools import lru_cache
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


# ==============================================================================
# DATA LOADING UTILITIES
# ==============================================================================
@lru_cache(maxsize=128)
def get_daily_validated_counts(site_id: str, call_type: str) -> pd.Series:
    """
    Finds the raw recording logs for a site/call_type, filters for validated 
    presence, and returns a Pandas Series of positive recordings indexed by Date.
    """
    pattern = f"*{site_id}*{call_type}*.csv"
    matching_files = list(PMJ_DIR.glob(pattern))

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


# ==============================================================================
# EXTENSIBLE METRIC FRAMEWORK
# ==============================================================================
class AcousticMetric:
    """Base class for all acoustic calculations."""
    
    def calculate_row(self, row: dict[Any, Any], hatch_date: date, site_id: str) -> dict[str, Any]:
        """Runs per row. Returns a dict of new column keys and values to append."""
        return {}

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs once over the final dataset. Used for dataset-wide statistics."""
        return df


class AcousticReproductiveIndex(AcousticMetric):
    """Calculates the ARI and performs post-processing gap analysis."""
    
    DAYS_TO_COUNT = 10
    NESTLING_OFFSET_DAYS = 2

    def calculate_row(self, row: dict[Any, Any], hatch_date: date, site_id: str) -> dict[str, Any]:
        f_start = hatch_date - timedelta(days=self.DAYS_TO_COUNT)
        f_end = hatch_date - timedelta(days=1)
        
        n_start = hatch_date + timedelta(days=self.NESTLING_OFFSET_DAYS)
        n_end = n_start + timedelta(days=self.DAYS_TO_COUNT - 1)
        
        female_counts = get_daily_validated_counts(site_id, "Female")
        nestling_counts = get_daily_validated_counts(site_id, "Nestling")

        f_win = female_counts[(female_counts.index >= f_start) & (female_counts.index <= f_end)]
        n_win = nestling_counts[(nestling_counts.index >= n_start) & (nestling_counts.index <= n_end)]

        result = {"ARI_num": "ND_MISSING_DATES", "ARI_Debug": ""}

        if len(f_win) < MIN_REQUIRED_DAYS or len(n_win) < MIN_REQUIRED_DAYS:
            result["ARI_num"] = "ND_INSUFFICIENT_DAYS"
            result["ARI_Debug"] = f"F_Days:{len(f_win)}, N_Days:{len(n_win)} (Req {MIN_REQUIRED_DAYS})"
            return result

        avg_female = f_win.sum() / len(f_win)
        avg_nestling = n_win.sum() / len(n_win)

        if avg_female == 0:
            result["ARI_num"] = "ND_DIV_ZERO" if avg_nestling > 0 else "0"
        else:
            result["ARI_num"] = round(avg_nestling / avg_female, 4)
            
        return result

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finds the cutoff gap and assigns categorical outcomes."""
        
        # 1. Safely convert to a purely numeric series for mathematical comparisons
        # Strings like 'ND_INSUFFICIENT_DAYS' become NaN (Not a Number)
        numeric_ari = pd.to_numeric(df["ARI_num"], errors="coerce")
        numeric_mask = numeric_ari.notna()

        # 2. Use the safe numeric_ari for the < 0.8 comparison
        valid_ari = df.loc[
            numeric_mask & (numeric_ari < 0.8),
            "ARI_num"
        ].sort_values().to_numpy(dtype=float)

        # 3. Calculate Cutoff
        cutoff = 0.15
        if len(valid_ari) >= 2:
            diffs = np.diff(valid_ari)
            max_gap_idx = int(np.argmax(diffs))
            cutoff = float(valid_ari[max_gap_idx] + (diffs[max_gap_idx] / 2))

        df["Calculated_Outcome"] = "Unknown"
        
        # 4. Apply outcomes using the safe numeric_ari for all evaluations
        df.loc[numeric_mask & (numeric_ari == 0), "Calculated_Outcome"] = "Abandoned"
        df.loc[numeric_mask & (numeric_ari > 0) & (numeric_ari <= cutoff), "Calculated_Outcome"] = "Partially Abandoned"
        df.loc[numeric_mask & (numeric_ari > cutoff), "Calculated_Outcome"] = "Successful"
        
        self.calculated_cutoff = cutoff  # Save for text reporting
        return df

class FledglingMetrics(AcousticMetric):
    """Tracks Fledgling output without conflating it with the core ARI math."""
    
    FLEDGLING_OFFSET_DAYS = 11
    FLEDGLING_LATEST_DAY_OFFSET = 15

    def calculate_row(self, row: dict[Any, Any], hatch_date: date, site_id: str) -> dict[str, Any]:
        fledge_start = hatch_date + timedelta(days=self.FLEDGLING_OFFSET_DAYS)
        fledge_end = hatch_date + timedelta(days=self.FLEDGLING_LATEST_DAY_OFFSET)

        f_counts = get_daily_validated_counts(site_id, "Fledgling")
        f_win = f_counts[(f_counts.index >= fledge_start) & (f_counts.index <= fledge_end)]

        present = "Yes" if not f_win.empty and f_win.sum() > 0 else "No"
        total = f_win.sum() if not f_win.empty else 0

        return {
            "Fledglings_Present": present,
            "Fledgling_Total": total
        }


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

    # 1. Initialize our desired metrics
    active_metrics: list[AcousticMetric] = [
        AcousticReproductiveIndex(),
        FledglingMetrics()
    ]

    source_df = pd.read_csv(BREEDING_DATES_CSV)
    processed_records = []

    # 2. Row-by-Row processing
    for row in source_df.to_dict("records"):
        out_row = row.copy()
        site_id = str(row.get("Site ID", "")).strip()
        hatch_str = str(row.get("hatch", "")).strip()
        
        try:
            hatch_date = pd.to_datetime(hatch_str.replace("~", "")).date()
            is_valid_date = True
        except Exception:
            is_valid_date = False

        # Run all metric plugins on this row
        for metric in active_metrics:
            if not site_id or not is_valid_date:
                # Provide a blank fallback if dates are unparseable
                out_row.update(metric.calculate_row({}, date.today(), ""))
                out_row["ARI_Debug"] = "Invalid or missing hatch date."
            else:
                out_row.update(metric.calculate_row(row, hatch_date, site_id))
                
        processed_records.append(out_row)

    # 3. Post-Processing (Statistical analysis via Pandas)
    results_df = pd.DataFrame(processed_records)
    for metric in active_metrics:
        results_df = metric.post_process(results_df)

    # 4. Exporting Data
    save_csv_with_retry(results_df, OUT_FILE)
    
    if SHARING_OUTPUT_DIR.exists():
        shutil.copy2(OUT_FILE, SHARING_OUTPUT_DIR / OUT_FILE.name)

    # Specific Filtering Logic
    numeric_mask = pd.to_numeric(results_df["ARI_num"], errors="coerce").notna()
    save_csv_with_retry(results_df[numeric_mask], OUT_FILTERED_FILE)

    comparison_mask = (
        ~results_df["hatch"].astype(str).str.strip().isin({"NHD", "ND", "inf", "missed", "n/a", "na", ""}) &
        ~results_df["Outcome"].astype(str).str.strip().isin({"Unknown"}) &
        results_df["Breeding Type"].astype(str).str.strip().isin({"Simple", "Sequential"})
    )
    save_csv_with_retry(results_df.loc[comparison_mask].reset_index(drop=True), OUT_ARI_FILE)

    # Extract dynamic properties for reporting (if they exist)
    ari_instance = next((m for m in active_metrics if isinstance(m, AcousticReproductiveIndex)), None)
    cutoff_val = getattr(ari_instance, "calculated_cutoff", "N/A")

    log_text = (
        f"Processing Complete\n===================\n"
        f"Rows processed: {len(results_df)}\n"
        f"ARI Cutoff Threshold: {cutoff_val}\n\n"
        f"Outcomes:\n---------\n{results_df['Calculated_Outcome'].value_counts().to_string()}\n"
    )
    
    RESULTS_TXT.write_text(log_text, encoding="utf-8")
    print(f"\nDone! Logs and exports generated.\n{log_text}")


if __name__ == "__main__":
    main()