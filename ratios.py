from __future__ import annotations
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import re
from typing import Any, Callable
from copy import deepcopy
import time


DAYS_TO_COUNT = 10
NESTLING_OFFSET_DAYS = 2    #How many days after hatching we start looking for NESTLING calls
FLEDGLING_OFFSET_DAYS = 11  #How many days after hatching we start looking for FLEDGLING calls
FLEDGLING_LATEST_DAY_OFFSET = 15  #How many days after Fledging Earliest do we stop looking for calls

BASE_DIR = Path(".")
DATA_DIR = Path("C:\\Users\\mikes\\OneDrive\\Documents\\GitHub\\TRBLSummarizer\\TRBLSummarizer\\")
BREEDING_DATES_CSV = BASE_DIR / "breeding dates.csv"
PMJ_DIR = DATA_DIR / "PMJ Data"
SHARING_OUTPUT_DIR = "G:\\My Drive\\TRBL for Wendy GDrive\\"
OUT_FILE = Path("nestling-to-female-ratios.csv")
OUTPUT_CSV = DATA_DIR / OUT_FILE
SHARING_OUTPUT_CSV = SHARING_OUTPUT_DIR / OUT_FILE

FILTERED_OUTPUT_CSV = DATA_DIR / "nestling-to-female-ratios-filtered.csv"
# Source column name for site in the tracking CSV (row-2 headers)
NAME_SOURCE_COL = "Name"
HATCH_COL = "hatch"
OUTCOME_COL = "Outcome"
PULSE_COL = "Pulse Name"


VALID_OUTCOMES = {
    "Successful",
    "Partially Abandoned",
    "Abandoned",
}
VALID_BT = {
    "Simple",
    "Sequential",
}


def make_base_result_row(row: pd.Series) -> dict[str, Any]:
    """
    Single source of truth: every output column appears here exactly once.
    """
    return {
        "Site_ID": row["Site ID"],
        "Site_Name": row["Name"],
        PULSE_COL: row[PULSE_COL],
        OUTCOME_COL: row[OUTCOME_COL],
        "Calculated_Outcome":"Unknown",
        "Breeding_Type": row["Breeding Type"],

        # Per-line / per-hatch-date fields
        "Hatch_Date": "NHD",          # overwritten later
        "Earliest_Rec": "n/a",
        "Incubation_Days": 0,
        "Total_Female_Calls": 0,
        "Avg_Female_Calls_Day": "n/a",
        "Latest_Rec": "n/a",
        "Nestling_Days": 0,
        "Total_Nestling_Calls": 0,
        "Avg_Nestling_Calls_Day": "n/a",
        "ARI": "n/a",

        #Fledgling stats
        "Latest_Fledgling_Rec":"n/a",
        "Fledgling_Days":0,  
        "Total_Fledgling_Calls":0,
        "Avg_Fledgling_Calls_Day":0,
        "Fledglings_Present":False,

        # Static site metadata copied to every output row
        "Substrate": row["Substrate"],
        "Approx_Size": row["Colony Size"],
        "Comment" : "",
    }

def parse_hatch_date_line(name: str, raw_line: str) -> datetime | None:
    """Parse a single line from the Hatch Date cell.
    Returns a datetime (normalized date) or None if invalid.
    Logs an error (print) if it can't be parsed.
    """
    if raw_line is None:
        return None

    s = str(raw_line).strip()
    if not s:
        return None

    # Skip ND
    if s.upper() == "ND":
        return None

    # Strip leading '~' and trailing '*' markers around dates
    # e.g. "~8/24/2024" -> "8/24/2024", "7/14/2024*" -> "7/14/2024"
    s = s.lstrip("~").rstrip("*").strip()

    # Drop trailing codes like " (r)", " (m)", " (m1)" etc. at the end of the string
    # Code length is 1–2 alphanumeric characters
    s = re.sub(r"\s*\([A-Za-z0-9]{1,2}\)$", "", s).strip()

    if s not in ["pre", "post", "ND", "n/a"]:
        # Try to parse as date (month/day/year style)
        try:
            dt = pd.to_datetime(s, errors="raise", dayfirst=False)
        except Exception:
            print(f"[ERROR] Bad hatch date for '{name}': {raw_line!r}")
            return None
    else:
        return None
    
    # Normalize to midnight
    return dt.normalize()


def load_breeding_dates_table() -> pd.DataFrame:
    """Load the tracking CSV, using row 2 as headers (skip row 1),
    and keep only Name + Hatch Date (from column B).
    """
    df = pd.read_csv(
        BREEDING_DATES_CSV,
        dtype=str,
    )
    expected_cols = [NAME_SOURCE_COL, HATCH_COL, OUTCOME_COL, PULSE_COL]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in tracking CSV: {missing}")

    df = df.rename(columns={HATCH_COL:"Hatch Date"})
    return df


def load_call_table(path: Path) -> pd.DataFrame:
    """Load a Female or Nestling call CSV with columns: year, month, day, validated.
    Add a 'date' column as a proper datetime.
    Returns an empty DataFrame if file does not exist.
    """
    if not path.exists():
        print(f"[WARN] Call file not found: {path}")
        return pd.DataFrame(columns=["year", "month", "day", "validated", "date"])

    df = pd.read_csv(path, dtype={"year": int, "month": int, "day": int, "validated": str})

    for col in ["year", "month", "day", "validated"]:
        if col not in df.columns:
            raise ValueError(f"File {path} is missing required column '{col}'.")

    # Build a date string column and parse it
    date_str = (
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str)
        + "-"
        + df["day"].astype(str)
    )

    df["date"] = pd.to_datetime(date_str, errors="coerce").dt.normalize()

    bad_dates = df["date"].isna().sum()
    if bad_dates:
        print(f"[WARN] {bad_dates} rows in {path} have invalid year/month/day and were ignored.")
        df = df.dropna(subset=["date"])

    df = df.sort_values("date").reset_index(drop=True)
    return df


def count_valid_calls(df: pd.DataFrame) -> int:
    """Count rows where validated == 'present' (case-insensitive, trimmed)."""
    if df.empty:
        return 0
    vals = df["validated"].astype(str).str.strip().str.lower()
    return (vals == "present").sum()


def summarize_for_hatch_date(
    hatch_date: pd.Timestamp,
    female_df: pd.DataFrame,
    nestling_df: pd.DataFrame,
    fledgling_df: pd.DataFrame,
    site_info:pd.Series,
) -> dict:
    comment = ""
    """Given a site name, a hatch_date, and already-loaded female/nestling dataframes
    (with a 'date' column), compute all summary metrics and return a dict for results.
    """
    #TODO Each window must be > 3 days long. If not enough data, output 'n/a' for averages and ratio. 
    # Add a column indicating this.

    # Female window: from hatch_date - (DAYS_TO_COUNT - 1) to hatch_date inclusive
    female_start = hatch_date - timedelta(days=DAYS_TO_COUNT)
    female_end = hatch_date - timedelta(days=1)
    potential_female = female_df[
        (female_df["date"] >= female_start) & (female_df["date"] <= female_end)
    ]

    # Nestling window: start NESTLING_OFFSET_DAYS after hatch_date, for DAYS_TO_COUNT days
    nestling_start = hatch_date + timedelta(days=NESTLING_OFFSET_DAYS)
    nestling_end = nestling_start + timedelta(days=DAYS_TO_COUNT - 1)
    potential_nestling = nestling_df[
        (nestling_df["date"] >= nestling_start) & (nestling_df["date"] <= nestling_end)
    ]

    # Fledgling window
    #count of days from the hatch_date + 11 until fledge_start + 9 or until latest fledge rec
    fledgling_start = hatch_date + timedelta(days=FLEDGLING_OFFSET_DAYS)
    fledgling_end = fledgling_start + timedelta(days=FLEDGLING_LATEST_DAY_OFFSET - 1)
    potential_fledgling = fledgling_df[
        (fledgling_df["date"] >= fledgling_start) & (fledgling_df["date"] <= fledgling_end)
    ]
    
    # Female summaries (dates based on any recordings in window, not only 'present')
    if not potential_female.empty:
        earliest_rec = potential_female["date"].min()
        incubation_days = (hatch_date - earliest_rec).days
        total_female_calls = count_valid_calls(potential_female)
        avg_female_per_day = (
            total_female_calls / incubation_days if incubation_days > 0 else None
        )
    else:
        earliest_rec = None
        incubation_days = 0
        total_female_calls = 0
        avg_female_per_day = None

    # Nestling summaries
    if not potential_nestling.empty:
        latest_rec = potential_nestling["date"].max()
        nestling_days = (latest_rec - nestling_start).days + 1
        total_nestling_calls = count_valid_calls(potential_nestling)
        avg_nestling_per_day = (
            total_nestling_calls / nestling_days if nestling_days > 0 else None
        )
    else:
        latest_rec = None
        nestling_days = 0
        total_nestling_calls = 0
        avg_nestling_per_day = None

    #Fledgling summaries
    if not potential_fledgling.empty:
        latest_fledge_rec = potential_fledgling["date"].max()
        fledgling_days = (latest_fledge_rec - fledgling_start).days + 1
        total_fledgling_calls = count_valid_calls(potential_fledgling)
        avg_fledgling_calls_per_day = (
            total_fledgling_calls / fledgling_days if fledgling_days > 0 else None
        )
    else:
        latest_fledge_rec = None
        fledgling_days = 0
        total_fledgling_calls = 0
        avg_fledgling_calls_per_day = None

    # Calculate ARI Ratio
    if not (avg_female_per_day is None) and \
       incubation_days >= 4 and \
       nestling_days >= 4 and \
       site_info[OUTCOME_COL] in (VALID_OUTCOMES) and \
       site_info["Breeding Type"] in VALID_BT:
        if avg_female_per_day > 0 :
            ratio = avg_nestling_per_day / avg_female_per_day 
        else:
            comment = "Female calls = 0"
            ratio = None
    else:
        ratio = None
        if total_female_calls == 0:
            comment += "Female calls = 0 |"
        if incubation_days < 4:
            comment += "Incubation days less than 4 |"
        if nestling_days < 4:
            comment += "Nestling days less than 4 |"
        if not (site_info[OUTCOME_COL] in (VALID_OUTCOMES)):
            comment += f"Outcome is not valid: {site_info[OUTCOME_COL]} |"
        if not (site_info["Breeding Type"] in VALID_BT):
            comment += f"Breeding Type is not valid: {site_info["Breeding Type"]}"

    def fmt_date(d: pd.Timestamp | None) -> str:
        if d is None or pd.isna(d):
            return "n/a"
        return d.strftime("%Y-%m-%d")

    def fmt_number(x):
        if x is None:
            return "n/a"
        return x

    return {
        "Hatch_Date": fmt_date(hatch_date),
        "Earliest_Rec": fmt_date(earliest_rec),
        "Incubation_Days": incubation_days,
        "Total_Female_Calls": total_female_calls,
        "Avg_Female_Calls_Day": fmt_number(
            round(avg_female_per_day, 3) if avg_female_per_day is not None else None
        ),
        "Latest_Rec": fmt_date(latest_rec),
        "Nestling_Days": nestling_days,
        "Total_Nestling_Calls": total_nestling_calls,
        "Avg_Nestling_Calls_Day": fmt_number(
            round(avg_nestling_per_day, 3) if avg_nestling_per_day is not None else None
        ),
        "ARI": fmt_number(round(ratio, 3) if ratio is not None else None),

        "Latest_Fledgling_Rec" : fmt_date(latest_fledge_rec), 
        "Fledgling_Days" : fledgling_days, 
        "Total_Fledgling_Calls" : total_fledgling_calls, 
        "Avg_Fledgling_Calls_Day" : fmt_number(
            round(avg_fledgling_calls_per_day, 3) if avg_fledgling_calls_per_day is not None else None
        ), 
        "Fledglings_Present" : True if total_fledgling_calls else False,
        "Comment" : comment,
    }


def save_csv_with_retry(df, path, retry_delay=1.0):
    """
    Attempt to save a CSV. If the file is open (e.g., in Excel),
    prompt the user and retry until successful.
    """
    while True:
        try:
            df.to_csv(path, index=False)
            print(f"Saved results to {path}")
            break

        except PermissionError:
            print(
                f"\n⚠️  Cannot write to {path}."
                "\nIt is likely open in Excel."
                "\nPlease close the file, then press Enter to retry "
                "(Ctrl+C to abort)."
            )
            input()
            time.sleep(retry_delay)



def find_ari_cutoff_option1(
    filtered_df: pd.DataFrame,
    ari_col_num: str = "ARI_num",
    upper_bound: float = 0.8,
    top_n: int = 10,
    rounding: float = 0.01,
):
    """
    Option 1 (recommended): find the *largest adjacent gap within the lower ARI regime*
    and use it to define a cutoff separating suppressed vs normal productivity.

    Key idea
    --------
    Instead of taking the largest gap anywhere (which often lands in the high-success tail),
    restrict the gap search to ARI values <= upper_bound (default 0.8). This focuses the
    cutoff on the transition relevant to partial abandonment.

    Inputs
    ------
    filtered_df[ari_col_num] should be numeric with NaNs for non-numeric values.
    Example:
        filtered_df["ARI_num"] = pd.to_numeric(filtered_df["ARI"], errors="coerce")

    Returns
    -------
    cutoff_info : dict
        Bracketing values, gap size, midpoint cutoff, and rounded cutoff.
    gaps_df_used : pd.DataFrame
        Adjacent gaps within the restricted range, sorted by gap size (desc).
    """

    # 1) Clean + sort numeric ARI
    ari_sorted = (
        filtered_df[ari_col_num]
        .dropna()
        .astype(float)
        .sort_values()
        .reset_index(drop=True)
    )

    if len(ari_sorted) < 3:
        raise ValueError("Not enough numeric ARI values to compute gaps.")

    # 2) Build adjacent-gap table
    lower = ari_sorted.iloc[:-1].to_numpy()
    upper = ari_sorted.iloc[1:].to_numpy()
    gaps = upper - lower

    gaps_df = pd.DataFrame(
        {"ARI_lower": lower, "ARI_upper": upper, "gap": gaps}
    )

    # 3) Restrict to the lower regime (Option 1)
    gaps_df_used = gaps_df[gaps_df["ARI_upper"] <= upper_bound].copy()
    if gaps_df_used.empty:
        raise ValueError(
            f"No adjacent pairs found with ARI_upper <= {upper_bound}. "
            "Increase upper_bound or check your data."
        )

    # 4) Identify the largest gap within the restricted range
    gaps_df_used.sort_values("gap", ascending=False, inplace=True)
    best = gaps_df_used.iloc[0]

    ari_lower = float(best["ARI_lower"])
    ari_upper = float(best["ARI_upper"])
    best_gap = float(best["gap"])

    # Midpoint is a natural boundary; you can also choose ari_upper or a rounded value.
    cutoff_midpoint = (ari_lower + ari_upper) / 2.0

    # 5) Round to a "nice" cutoff if desired
    if rounding and rounding > 0:
        cutoff_rounded = round(cutoff_midpoint / rounding) * rounding
    else:
        cutoff_rounded = cutoff_midpoint

    cutoff_info = {
        "n_values_total": int(len(ari_sorted)),
        "upper_bound_used": float(upper_bound),
        "ari_lower": ari_lower,
        "ari_upper": ari_upper,
        "gap": best_gap,
        "cutoff_midpoint": float(f"{cutoff_midpoint:.6g}"),
        "cutoff_rounded": float(f"{cutoff_rounded:.6g}"),
        "interpretation": (
            f"Option 1: largest adjacent gap with ARI <= {upper_bound} is {best_gap:.6g} "
            f"between {ari_lower:.6g} and {ari_upper:.6g}. "
            f"Suggested cutoff is midpoint {cutoff_midpoint:.6g} "
            f"(rounded to {cutoff_rounded:.6g})."
        ),
    }

    # Return a transparent list of the top gaps within the restricted range
    top_gaps = gaps_df_used.head(top_n).reset_index(drop=True)

    return cutoff_info, top_gaps






def make_ratios():
    print("Loading breeding dates table...")
    breeding_dates_df = load_breeding_dates_table()

    results = []

    for _, row in breeding_dates_df.iterrows():
        name = str(row["Name"]).strip() if pd.notna(row["Name"]) else ""
        hatch_cell = row["Hatch Date"]

        if not name:
            continue

        if pd.isna(hatch_cell) or not str(hatch_cell).strip():
            continue

        # Split into lines for multiple hatch dates
        lines = str(hatch_cell).splitlines()

        # Load call data once per site
        female_path = PMJ_DIR / name / f"{name} Female.csv"
        nestling_path = PMJ_DIR / name / f"{name} Nestling.csv"
        fledgling_path = PMJ_DIR / name / f"{name} Fledgling.csv"

        female_df = load_call_table(female_path)
        nestling_df = load_call_table(nestling_path)
        fledgling_df = load_call_table(fledgling_path)

        if female_df.empty and nestling_df.empty:
            print(f"[INFO] No female or nestling data for '{name}'; still outputting rows with 'n/a' where needed.")

        for raw_line in lines:
            base = make_base_result_row(row=row)
            hatch_dt = parse_hatch_date_line(name, raw_line)
            if hatch_dt is None:
                # Output a row indicating this line existed but could not be parsed
                base["Hatch_Date"] = str(raw_line).strip() if raw_line != "ND" else "NHD"
                base["Comment"] = "No valid hatch date"
            else:
                patch = summarize_for_hatch_date(hatch_dt, 
                                               female_df, 
                                               nestling_df,
                                               fledgling_df,
                                               row)
                base.update(patch)
            results.append(base)

    if results:
        results_df = pd.DataFrame(results)

        save_csv_with_retry(results_df, OUTPUT_CSV)
        save_csv_with_retry(results_df, SHARING_OUTPUT_CSV)
        print(f"[DONE] Wrote full results to: {OUTPUT_CSV} and {SHARING_OUTPUT_CSV}")

        filtered_outcomes =  {"Successful","Partially Abandoned",}
        mask_keep = (
            results_df[OUTCOME_COL].isin(filtered_outcomes) &
            results_df["Breeding_Type"].isin(VALID_BT) &
            results_df["Incubation_Days"].ge(4) &  #.ge == "greater than or equal to"
            results_df["Nestling_Days"].ge(4) &
            results_df["Total_Female_Calls"].ge(1) &
            results_df["Fledglings_Present"]
        )
        filtered_df = results_df.loc[mask_keep].reset_index(drop=True)  
        save_csv_with_retry(filtered_df, FILTERED_OUTPUT_CSV)
        print(f"[DONE] Wrote filtered results to: {FILTERED_OUTPUT_CSV}")

        # ------------------------------------------------------------------
        # 1) Create numeric ARI once (shared logic for entire pipeline), gets rid of "n/a"
        # ------------------------------------------------------------------
        results_df["ARI_num"] = pd.to_numeric(results_df["ARI"], errors="coerce")
        filtered_df["ARI_num"] = pd.to_numeric(filtered_df["ARI"], errors="coerce")

        # ------------------------------------------------------------------
        # 2_old) Compute percentiles from the filtered (valid) dataset
        # ------------------------------------------------------------------
        quantiles = filtered_df["ARI_num"].quantile(
            [0.05, 0.10, 0.15],
            interpolation="linear"
        )

        msg = ""
        msg += "ARI percentiles (from filtered_df):\n"
        msg += f"{quantiles}"

        # cutoff = quantiles.loc[0.10]
        # print(f"\nUsing 10th percentile cutoff = {cutoff:.3f}")

        # ------------------------------------------------------------------
        # 2_new) Compute ARI cutoff from the filtered (valid) dataset
        # ------------------------------------------------------------------

        cutoff_info, top_gaps = find_ari_cutoff_option1(
            filtered_df,
            ari_col_num="ARI_num",
            upper_bound=0.8,   # tune: 0.8 or 1.0 are common; pick one and keep it consistent
            top_n=10,
            rounding=0.01
        )

        msg += f"{cutoff_info['interpretation']}"
        msg += "\nTop gaps within lower regime:\n"
        msg += f"{top_gaps.to_string(index=False)}"
        msg += f"\n{cutoff_info}\n"

        cutoff = cutoff_info["cutoff_midpoint"]

        # ------------------------------------------------------------------
        # 3) Initialize Calculated_Outcome
        # ------------------------------------------------------------------
        results_df["Calculated_Outcome"] = "Unknown"

        # ------------------------------------------------------------------
        # 4) Apply outcome logic (priority order matters)
        # ------------------------------------------------------------------
        # TODO: If there are fledglings, then we're calling it either PA or S despite the ARI score. 
        # May want to adjust the output later to reflect this. 

        # ARI == 0 → Abandoned
        mask_abandoned = results_df["ARI_num"].eq(0)
        results_df.loc[mask_abandoned, "Calculated_Outcome"] = "Abandoned"

        # 0 < ARI ≤ cutoff → Partially Abandoned
        mask_partial = (
            results_df["ARI_num"].gt(0) &
            results_df["ARI_num"].le(cutoff)
        )
        results_df.loc[mask_partial, "Calculated_Outcome"] = "Partially Abandoned"

        # ARI > cutoff  → Successful
        mask_success = results_df["ARI_num"].gt(cutoff)
        results_df.loc[mask_success, "Calculated_Outcome"] = "Successful"

        # ------------------------------------------------------------------
        # 5) Optional sanity check
        # ------------------------------------------------------------------
        msg += "\nCalculated_Outcome counts:"
        msg += f"{results_df["Calculated_Outcome"].value_counts()}"


        # ------------------------------------------------------------------
        # 6) Handle cases that can't be calc'd from ARI
        # ------------------------------------------------------------------
        
        #TBD

        with open("ratios_results.txt", "w") as file:
            file.write(msg)

        # For comparison with the old one
        bad = {"NHD", "post", "pre"}
        bad2={"Unknown"}
        bad3={"Simple", "Sequential"}
        mask = (
            ~results_df["Hatch_Date"].isin(bad) &
            ~results_df["Outcome"].isin(bad2) &
            results_df["Breeding_Type"].isin(bad3)
        )
        for_comp = results_df.loc[mask].reset_index(drop=True) 
        for_comp_csv = DATA_DIR / "nestling-to-female-ratios-for-comparison.csv"
        save_csv_with_retry(for_comp, for_comp_csv)


    else:
        print("Empty file, no results generated")


if __name__ == "__main__":
    make_ratios()

