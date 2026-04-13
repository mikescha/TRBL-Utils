'''
Docstring for validate_dates

Get breeding attempts from breeding dates.csv

For each breeding attempt, validate dates as follows
	if hatch is a date,
		Load the year data file for that site
		get the dates hatch +/- N
		if this is pulse 1, 
			get all days with P1N = 1
				Add up altsong per day
				Report the first day with altsong >= 4 if none of the days have at least 4, then report the highest day with a (*)

		if this is pulse 2-4 do the same thing but for ync-p whatever like in the graphing

	if the date is fledgestart
		load the PM for fledging for that site
		get the dates fledgestart +/- N
		count the number of recordings per day where validated == "present". 
		Report the first day with count of recordings >= 4 if none of the days have at least 4 then report the highest day with a (*)

	if the date is fledgedisp
		same as above
'''


from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from enum import Enum, auto
from typing import Optional, Union
import time
import math
DAYS=5 #The number of days on either side of the target day to check
threshold=4  #The target number of recordings per day that must match. We're starting with this and will update it.





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


BASE_DIR = Path(".")
DATA_DIR = Path("C:\\Users\\mikes\\OneDrive\\Documents\\GitHub\\TRBLSummarizer\\TRBLSummarizer\\")
BREEDING_DATES_CSV = BASE_DIR / "breeding dates.csv"
PMJ_DIR = DATA_DIR / "PMJ Data"
EDGE_DIR = DATA_DIR / "Data"
SHARING_OUTPUT_DIR = Path("G:\\My Drive\\TRBL for Wendy GDrive\\")
VALIDATIONS_FILENAME = "validations.csv"
OUTPUT_CSV = BASE_DIR / VALIDATIONS_FILENAME
SHARING_OUTPUT_CSV = SHARING_OUTPUT_DIR / VALIDATIONS_FILENAME


NAME_COL = "Name"
OUTCOME_COL = "Outcome"
PULSE_COL = "Pulse Name"
HATCH_COL = "hatch"
FLEDGE_START_COL = "fledgestart"
FLEDGE_DISP_COL = "fledgedisp"
INC_START_COL = "incstart"
MC_START_COL = "mcstart"

RESULTS_HATCH = "Hatch Date"
RESULTS_FS = "Fledge Start"
RESULTS_DISP = "Dispersal"
RESULT = "result"
COMMENT = "Comment"

TAG_MAP = {"p1": ["tag<p1n>", "val<Agelaius tricolor/Alternative Song>"],
           "p2": ["tag<p2n>", "tag<YNC-p2>"],
           "p3": ["tag<p3n>", "tag<YNC-p3>"],
           "p4": ["tag<p4n>", "tag<YNC-p4>"],
}

def make_output_file_row(row: pd.Series)->dict:
    d = {
        "Site ID" : row["Site ID"],
        PULSE_COL : row[PULSE_COL],
        "Breeding_type" : row["Breeding Type"],
        "Complex_type" : row["Complex Types"],
        "Twiddle" : "",
        "Duplicate":row["Comment"],
        "Original_MC_Date": format_date(row[MC_START_COL]),
        "Calc_MC_Date":"",
        "Calc_MC_Delta":"",
        "MC_msg":"",
        "Original_Inc_Date": format_date(row[INC_START_COL]),
        "Original_Hatch_Date": format_date(row[HATCH_COL]),
        "Calc_Hatch_Date": "",
        "Hatch_Comment": "",
        "Hatch_Delta":"",
#        "Hatch_Window_Values": "",
        "Threshold" : threshold,
        "Original_Fledge_Date": format_date(row[FLEDGE_START_COL]),
        "Calc_Fledge_Date": "",
        "Fledge_Comment": "",
        "Fledge_Delta":"",
#        "Fledge_Window_Values": "",
        "Original_Dispersal_Date": format_date(row[FLEDGE_DISP_COL]),
        "Calc_Dispersal_Date": "",
        "Dispersal_Comment": "",
        "Dispersal_Delta":"",
#        "Dispersal_Window_Values": "",
    }
    return d


def load_breeding_dates_table() -> pd.DataFrame:
    """Load the tracking CSV, using row 2 as headers (skip row 1),
    and keep only Name + Hatch Date (from column B).
    """
    df = pd.read_csv(
        BREEDING_DATES_CSV,
        dtype=str,
    )
    expected_cols = [NAME_COL, PULSE_COL, HATCH_COL, FLEDGE_START_COL, FLEDGE_DISP_COL]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in tracking CSV: {missing}")
    return df


def clean_tag_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Normalize tag columns so that:
      - '---' -> NaN
      - NaN stays NaN
      - 0 -> 0
      - any positive value -> 1
    """
    df = df.copy()

    for col in cols:
        if not col == "dt":
            s = pd.to_numeric(df[col], errors="coerce")
            s = np.where(s > 0, 1, s)

            if (s < 0).any():
                bad = df.loc[s < 0, col].unique()
                raise ValueError(f"Negative values found in {col}: {bad}")

            df[col] = s

    return df


def load_data_for_site(site:str) -> pd.DataFrame:
    year = site[0:4]

    pfile_name = EDGE_DIR / f"data {year}.parquet"
    usecols = ["site", "dt",]
    all_cols = [v for values in TAG_MAP.values() for v in values]
    usecols.extend(all_cols)

    df = pd.read_parquet(pfile_name, columns=usecols)
    df = df[df["site"] == site]
    assert not df.empty, "Got no data"

    df = clean_tag_columns(df, usecols)
    df = df.set_index("dt")
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df.rename_axis("date")
    return df


class PMType(Enum):
    FLEDGLING = "Fledgling"
    HATCHLING = "Nestling"
    FEMALE = "Female"
    MC = "Male Chorus"
    MS = "MS"

    def csv_path(self, site_name: str) -> Path:
        return PMJ_DIR / site_name / f"{site_name} {self.value}.csv"


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


def load_pm_data_for_site(site_name:str, pm_type:PMType) -> pd.DataFrame:
    path = pm_type.csv_path(site_name)
    df = load_call_table(path)
    return df


def parse_date_for_index(value: str) -> pd.Timestamp | None:
    """
    Parse a string that should represent a date.
    Returns a normalized pandas Timestamp suitable for lookup
    against a normalized DatetimeIndex.

    Raises ValueError if the string is not a valid date.
    """
    value = format_date(value) # Try to convert to a DT. This includes some cleanup, like stripping extra chars
    try:
        ts = pd.to_datetime(value, errors="raise")
        return ts.normalize() 
    except Exception:
        return None


def count_paired_hits_per_day(df: pd.DataFrame, key_col: str, value_col: str) -> pd.Series:
    """
    Returns a Series indexed by date (normalized), values = count of rows where
    df[key_col] >= 1 AND df[value_col] >= 1.
    Missing/NaN values behave as non-hits.
    """
    # (Optional) assert values are only NaN/0/1
    allowed = {0, 1}
    for col in [key_col, value_col]:
        bad = df[col].dropna().loc[~df[col].dropna().isin(allowed)]
        assert bad.empty, f"Unexpected values in {col}: {bad.unique()}"

    mask = df[key_col].ge(1) & df[value_col].ge(1)

    # Group by calendar day (works whether index has times or not)
    day_index = pd.to_datetime(df.index, errors="coerce").normalize()
    daily_counts = mask.groupby(day_index).sum().astype(int)

    # Ensure DatetimeIndex
    daily_counts.index = pd.DatetimeIndex(daily_counts.index).normalize()
    return daily_counts


def count_present_per_day(df:pd.DataFrame) -> pd.Series:
    daily_counts = (
        df.loc[df["validated"] == "present"]
        .groupby("date")
        .size()
        .astype(int)
    )
    return daily_counts


def window_series(daily_counts: pd.Series, center_date, date_is_first=False) -> pd.Series:
    """
    Returns a strict N-day window as a Series indexed by daily dates.
    Missing days filled with 0.
    """
    if date_is_first:
        #The date we have is the leftmost edge of our window instead of the middle
        left = pd.to_datetime(center_date).normalize()
        idx = pd.date_range(left,
                            left + pd.Timedelta(days=21),
                            freq="D")
    else:
        center = pd.to_datetime(center_date).normalize()
        idx = pd.date_range(center - pd.Timedelta(days=DAYS),
                            center + pd.Timedelta(days=DAYS),
                            freq="D")
    return daily_counts.reindex(idx, fill_value=0)


def window_to_csvcell(window: pd.Series, floats:bool = False) -> str:
    """Convert the N-day window Series to '0,2,4,5,4,3,0'."""
    if floats:
        return ",".join(f"{x:.3f}" for x in window)
    else:
        return ",".join(str(int(x)) for x in window)


def first_date_meeting_threshold(window: pd.Series, threshold: int = threshold):
    """
    Given a N-day window Series (already reindexed with 0 fill),
    return the first date (chronological) with value >= threshold, plus that value.
    """
    hits = window[window.ge(threshold)]
    if hits.empty:
        return None, None
    return hits.index[0], int(hits.iloc[0])



def first_day_before_gap_at_least(window: pd.Series, threshold: int, gap: int):
    """
    Return the first date where value >= threshold and is followed by
    AT LEAST `gap` consecutive days with value < threshold (immediately after).

    Returns (date, value) or (None, None).
    """
    if gap < 1:
        raise ValueError("gap must be >= 1")

    w = window.sort_index()
    vals = pd.to_numeric(w, errors="coerce").fillna(0)

    n = len(vals)
    if n <= gap:
        return None, None

    for i in range(0, n - gap):
        if vals.iat[i] >= threshold and (vals.iloc[i+1:i+gap+1] < threshold).all():
            return w.index[i], int(vals.iat[i])

    return None, None


def format_date(value) -> str:
    """
    Format a value as MM/DD/YYYY if it looks like a date.
    Otherwise return 'n/a'.
    """
    if value is None or pd.isna(value):
        return value

    potential_date = value
    if type(value) == str:
        # Strip leading '~' and trailing '*' markers around dates
        # e.g. "~8/24/2024" -> "8/24/2024", "7/14/2024*" -> "7/14/2024"
        potential_date = potential_date.lstrip("~").rstrip("*").strip()

    try:
        ts = pd.to_datetime(potential_date, errors="raise")
        return ts.strftime("%m/%d/%Y")
    
    except Exception:
        return value


def date_str_to_date(value:str) -> pd.Timestamp | None:
    potential_date = value
    # Strip leading '~' and trailing '*' markers around dates
    # e.g. "~8/24/2024" -> "8/24/2024", "7/14/2024*" -> "7/14/2024"
    potential_date = potential_date.lstrip("~").rstrip("*").strip()

    try:
        ts = pd.to_datetime(potential_date, errors="raise")
        return ts
    except Exception:
        return None


def get_recording_vectors(
        window: pd.Series,
        recs_per_day: pd.DataFrame,
) -> dict:
    rec_date_col = "date"
    rec_count_col = "n_recordings"
    max_days = 365 #this is larger than any dataset, want to return everything available but keep the logic for now

    # ---- Prepare window ----
    w = window.sort_index()
    w_raw = pd.to_numeric(w, errors="coerce").fillna(0.0)

    # Ensure that we have at least N days of data but not past the end of our window
    start = w_raw.index.min()
    end_detections = start + pd.Timedelta(days=max_days - 1)

    recs_idx_by_date = recs_per_day.assign(date=pd.to_datetime(recs_per_day["date"]).dt.normalize()) \
                               .set_index("date")
    end_recordings = recs_idx_by_date.index.max()
    end = min(end_detections, end_recordings)
    
    if start is pd.NaT or end is pd.NaT:
        vectors = {
            "detections_per_day":None,
            "recordings_per_day":None,
            "normalized":None,
            "errors":None,
        }
    else:    
        full_idx = pd.date_range(start, end, freq="D")

        # Pad / fill missing days in the window span with 0 detections
        w_raw = w_raw.reindex(full_idx, fill_value=0.0)
        
        # ---- Build denominators keyed by date ----
        denom_by_day = (
            recs_per_day.assign(date=pd.to_datetime(recs_per_day[rec_date_col]).dt.normalize())
                    .groupby("date")[rec_count_col]
                    .sum()
                    .astype(float)
                    .sort_index()
        )
        denom = denom_by_day.reindex(w_raw.index)

        # ---- Data quality checks ----
        n_missing = denom.isna().sum()
        n_zero = (denom == 0).sum()

        # Avoid divide-by-zero; keep NaN for bad denom so it doesn't silently become 0
        denom_safe = denom.mask(denom <= 0, np.nan)

        # Do the normalization
        window_norm = (w_raw / denom_safe).replace([np.inf, -np.inf], np.nan)

        # For the gap algorithm, treat NaN normalized values as 0 *after* tracking issues
        window_norm = window_norm.fillna(0.0).sort_index()

        error_text = ""
        if n_missing or n_zero:
            error_text = f"Missing: {int(n_missing)}, Zero: {int(n_zero)}"

        vectors = {
            "detections_per_day":w_raw,
            "recordings_per_day":denom_safe,
            "normalized":window_norm,
            "errors":error_text,
        }
    
    return vectors


def is_small_colony(v:dict)-> dict:
    """
    Determine if a colony is "small" based on the ratio of detections to recordings in the window, using a log scale and 
    quartiles derived from the dataset.
    """
    # Get number of recordings made during the window
    recs_per_day = v["recordings_per_day"]
    clean_recs = recs_per_day.mask(pd.isna(recs_per_day), 0)
    recordings_count = sum(clean_recs)

    # Get number of presences in the window
    detects_per_day = v["detections_per_day"]
    detections_count = sum(detects_per_day)

    # Use log 
    log_value = math.log10((detections_count/recordings_count) + 1)

    # Data calculated externally
    quartile_edges = [
        (float("inf"), 0.052011628, "Q3"),
        (0.052011628, 0.026146981, "Q2"),
        (0.026146981, 0.005783594, "Q1"),
        (0.005783594, 0, "Q0"),
    ]

    # Determine which quartile this particular site is in
    quartile = None
    for high, low, label in quartile_edges:
        if high >= log_value > low:
            quartile = label
            break

    summary = {
        "log" : log_value,
        "quartile" : quartile,
        "detections": detections_count,
        "recordings": recordings_count,
        "is_small": True if quartile == "Q0" else False
    }
    return summary


def date_before_gap_by_normalized_baseline(
    date_vectors: dict,
    *,
    median_window: int = 3,   # N
    gap_days: int = 2,         # K
    drop_reqd: float = 0.25,
    max_days = 28,
):

    """
    Normalize `window` by recordings/day. Baseline = mean of first `median_window` days
    non-zero normalized values.

    Gap definition (updated):
      Find the FIRST date i where window_norm[i] >= baseline AND the next `gap_days`
      consecutive days are all < baseline. Return that date i (the "date_before_gap").

    Returns a dict with:
      - date_before_gap
      - raw_before_gap
      - norm_before_gap
      - baseline
      - window_norm (Series, useful for debugging)
      - msg
    """
    result = {
        "date_before_gap": None,
        "raw_before_gap": None,
        "norm_before_gap": None,
        "baseline": 0,
        "window_norm": None,
        "msg": "None",
    }

    window_norm = date_vectors["normalized"]
    recs_per_day = date_vectors["recordings_per_day"]
    w_raw = date_vectors["detections_per_day"]

    # Ensure that we have at least N days of data but not past the end of our window
    start = w_raw.index.min()
    end_detections = start + pd.Timedelta(days=max_days - 1)

    end_recordings = recs_per_day.index.max()
    end = min(end_detections, end_recordings)
    
    full_idx = pd.date_range(start, end, freq="D")

    # Pad / fill missing days in the window span with 0 detections
    window_norm = window_norm.reindex(full_idx, fill_value=0.0)

    # ---- Baseline: mean of first N non-zero normalized days ----
    nonzero = window_norm[window_norm > 0]
    if len(nonzero) < median_window:
        result["msg"] = "Not enough non-zero days in the dataset to calculate a mean"
        return result

    baseline = float(nonzero.iloc[:median_window].mean()) * drop_reqd

    # ---- Find FIRST "drop" day: >= baseline followed by gap_days consecutive < baseline ----
    vals = window_norm.to_numpy()
    idx = window_norm.index

    pos_before_gap = None
    for i in range(0, len(vals) - gap_days):
        if (vals[i] >= baseline) and (vals[i + 1 : i + gap_days + 1] < baseline).all():
            pos_before_gap = i
            break

    result["baseline"] = baseline

    if pos_before_gap is None:
        result["msg"] = "No gap found"
    else:
        # ---- Populate results ----
        result["date_before_gap"] = idx[pos_before_gap] + pd.Timedelta(days=1)
        result["raw_before_gap"] = float(w_raw.iloc[pos_before_gap])
        result["norm_before_gap"] = float(window_norm.iloc[pos_before_gap])
        result["msg"] = "OK"
    return result


def check_if_dispersal_dates_match(date_vectors:dict, 
                         orig_dispersal_date, 
                         median_window:int = 3,
                         drop_reqd:float = 0.25,
                         gap_days:int = 3,
                         days_to_check:int = 28,
    )->dict:

    max_hit = ""
    #In this case, we need the window to start on target_date and extend N days, currently using 21 days
    data = date_before_gap_by_normalized_baseline(
        date_vectors,
        median_window = median_window,     #How many days at the start we use to calculate the median?
        drop_reqd = drop_reqd,              #What drop below the median is required to consider dispersal has started?
        gap_days = gap_days,                #How many days below the threshold are required to consider dispersal has occurred?
        max_days = days_to_check,           #How many days of recordings will we check, at most?
    )
    calc_dt = data["date_before_gap"]
    #todo i took out the code that looked for missing recording dates, may want to put it back
    if data["msg"] == "No gap found":
        max_hit = "; Date at or after window end!"

    comment = (
        "equal"
        if pd.to_datetime(orig_dispersal_date).normalize() == calc_dt
        else "not equal"
    ) + max_hit

    results = {
        "calc_date":format_date(calc_dt),
        "comment":comment,
    }
    results["Disp_Threshold"] = f"{data["baseline"]:.3f}"
    return results


def check_dispersal_model2(
    detections_per_day_norm: pd.Series,
    detections_per_day_raw: pd.Series,
    orig_dispersal_date: Union[str, pd.Timestamp, None],
    median_window: int = 3,
    moving_median_window: int = 3,
    drop_reqd: float = 0.1,
    days_absent_per_week: int = 7,
    max_window_days: int = 30,
) -> dict:
    """
    Model 2 (forward-median + sliding 7-day absence blocks)

    - Onset date = first day in the provided series (after sorting/normalizing index).
    - Baseline B = median of the top `median_window` daily rates within the first 14 days of onset.
    - For each candidate date (Day 0..max_window_days from onset):
        * compute forward moving median over `moving_median_window` days
        * evaluate the *following 7 days* block (candidate..candidate+6)
        * count days in that 7-day block where forward-median value <= (B * drop_reqd)
        * if count >= days_absent_per_week: dispersal_date = candidate (first date of the block)

    Notes:
    - The series is reindexed to a complete daily range so "following 7 days" means 7 calendar days.
    - NaNs in the 7-day block do NOT count as absent (conservative); adjust if you prefer otherwise.
    """

    # ---------- basic validation ----------
    if detections_per_day_norm is None or len(detections_per_day_norm) == 0:
        return {
            "status": "No data",
            "message": "detections_per_day_norm is empty",
            "dispersal_date": None,
            "delta_days_vs_orig": None,
            "baseline_B": None,
            "threshold": None,
        }

    if median_window <= 0:
        raise ValueError("median_window must be >= 1")
    if moving_median_window <= 0:
        raise ValueError("moving_median_window must be >= 1")
    if drop_reqd <= 0:
        raise ValueError("drop_reqd must be > 0")
    if not (1 <= days_absent_per_week <= 7):
        raise ValueError("days_absent_per_week must be in [1, 7]")
    if max_window_days <= 0:
        raise ValueError("max_window_days must be >= 1")

    s = detections_per_day_norm.copy()

    # ---------- normalize index to daily DatetimeIndex ----------
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception as e:
            return {
                "status": "Bad index",
                "message": f"Could not convert index to datetime: {e}",
                "dispersal_date": None,
                "delta_days_vs_orig": None,
                "baseline_B": None,
                "threshold": None,
            }

    s = s.sort_index()
    s.index = s.index.normalize()

    # aggregate duplicates (if any) by mean
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()

    # Reindex to a complete daily range so 7-day blocks are calendar-true
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)

    # ---------- onset definition ----------
    # As requested: onset is the first day in the series (post-sorting/normalizing).
    onset_date = s.index.min()
    s_from_onset = s.loc[onset_date:]

    # ---------- compute baseline B from first 14 days ----------
    first14 = s_from_onset.iloc[:14].dropna()
    if len(first14) == 0:
        return {
            "status": "No baseline",
            "message": "First 14 days from onset contain only NaNs; cannot compute baseline.",
            "dispersal_date": None,
            "delta_days_vs_orig": _delta_days_safe(orig_dispersal_date, None),
            "baseline_B": None,
            "threshold": None,
            "onset_date": onset_date,
        }

    k = min(median_window, len(first14))
    B = float(first14.nlargest(k).median())
    threshold = B * float(drop_reqd)

    if not np.isfinite(B) or B <= 0:
        return {
            "status": "No baseline",
            "message": f"Computed baseline B is not usable (B={B}).",
            "dispersal_date": None,
            "delta_days_vs_orig": _delta_days_safe(orig_dispersal_date, None),
            "baseline_B": B,
            "threshold": threshold,
            "onset_date": onset_date,
        }

    # ---------- moving median ----------
    # Implement via trailing rolling median + negative shift.
    trailing_med = s_from_onset.rolling(window=moving_median_window, min_periods=1).median()
    forward_med = trailing_med.shift(-(moving_median_window - 1))

    # ---------- scan candidates (sliding 7-day blocks) ----------
    max_scan = min(max_window_days, len(forward_med) - 1)
    best_date: Optional[pd.Timestamp] = None
    debug_rows = []

    for day_offset in range(0, max_scan + 1):
        candidate_date = onset_date + pd.Timedelta(days=day_offset)
        if candidate_date not in forward_med.index:
            continue

        # 7-day block candidate..candidate+6 (June 1-7, then June 2-8, etc.)
        block_idx = pd.date_range(candidate_date, candidate_date + pd.Timedelta(days=6), freq="D")
        block = forward_med.reindex(block_idx)

        absent_mask = block <= threshold
        absent_count = int(absent_mask.fillna(False).sum())

        debug_rows.append(
            {
                "candidate_date": candidate_date,
                "absent_count_in_block": absent_count,
                "non_nan_days_in_block": int(block.notna().sum()),
                "block": block,
            }
        )

        if absent_count >= days_absent_per_week:
            block_raw = detections_per_day_raw.reindex(block_idx)
            #Must be at least one detection on a day count; if not then pick the first day
            mask = block_raw > 0
            dispersal_end = (
                block_raw.loc[mask].index.max()
                if mask.any()
                else block_raw.index.min()
            )
            best_date = dispersal_end + pd.Timedelta(days=1) #pick the day after this
            break

    # ---------- outcomes ----------
    if best_date is not None:
        return {
            "status": "Dispersal detected",
            "message": "Threshold met for a 7-day sliding block starting at dispersal_date.",
            "dispersal_date": best_date.strftime("%Y-%m-%d"),
            "delta_days_vs_orig": _delta_days_safe(orig_dispersal_date, best_date),
            "baseline_B": B,
            "threshold": threshold,
            "onset_date": onset_date,
            "params": {
                "median_window": median_window,
                "moving_median_window": moving_median_window,
                "drop_reqd": drop_reqd,
                "days_absent_per_week": days_absent_per_week,
                "max_window_days": max_window_days,
            },
            "debug_scan": pd.DataFrame(debug_rows),
        }

    # If no trigger, decide why (end of data vs window limit)
    reached_window = (onset_date + pd.Timedelta(days=max_window_days)) <= forward_med.index.max()
    if reached_window:
        return {
            "status": "Reached window",
            "message": "No dispersal detected within max_window_days starting at onset.",
            "dispersal_date": None,
            "delta_days_vs_orig": _delta_days_safe(orig_dispersal_date, None),
            "baseline_B": B,
            "threshold": threshold,
            "onset_date": onset_date,
            "params": {
                "median_window": median_window,
                "moving_median_window": moving_median_window,
                "drop_reqd": drop_reqd,
                "days_absent_per_week": days_absent_per_week,
                "max_window_days": max_window_days,
            },
            "debug_scan": pd.DataFrame(debug_rows),
        }

    return {
        "status": "No dispersal in dataset",
        "message": "Reached end of dataset before dispersal threshold was met.",
        "dispersal_date": None,
        "delta_days_vs_orig": _delta_days_safe(orig_dispersal_date, None),
        "baseline_B": B,
        "threshold": threshold,
        "onset_date": onset_date,
        "params": {
            "median_window": median_window,
            "moving_median_window": moving_median_window,
            "drop_reqd": drop_reqd,
            "days_absent_per_week": days_absent_per_week,
            "max_window_days": max_window_days,
        },
        "debug_scan": pd.DataFrame(debug_rows),
    }


def _delta_days_safe(
    orig_dispersal_date: Union[str, pd.Timestamp, None],
    computed_date: Optional[pd.Timestamp],
) -> Optional[int]:
    if orig_dispersal_date is None or computed_date is None:
        return None
    try:
        od = pd.to_datetime(orig_dispersal_date).normalize()
        cd = pd.to_datetime(computed_date).normalize()
        return int((cd - od).days)
    except Exception:
        return None


def check_if_fledge_dates_match(daily_counts: pd.Series, target_date)->dict:
    # Get the first date that has at least 4 recordings and convert the data to a format we can save to CSV
    window = window_series(daily_counts, target_date)
    calc_dt, calc_hatch_val = first_date_meeting_threshold(window, threshold=threshold)
    window_str = window_to_csvcell(window)

    comment = (
        "equal"
        if pd.to_datetime(target_date).normalize() == calc_dt
        else "not equal"
    )

    results = {
        "calc_date":format_date(calc_dt),
        "values":window_str,
        "comment":comment,
    }
    return results


def check_if_hatch_dates_match(daily_counts: pd.Series, target_date)->dict:
    # Get the first date that has at least 4 recordings and convert the data to a format we can save to CSV
    window = window_series(daily_counts, target_date)
    # We're using a fixed threshold of 4 because wendy manually reviewed 8 files for each day per site
    # So, this shouldn't vary by number of recordings per day
    calc_dt, calc_hatch_val = first_date_meeting_threshold(window, threshold=4)
    window_str = window_to_csvcell(window)

    comment = (
        "equal"
        if pd.to_datetime(target_date).normalize() == calc_dt
        else "not equal"
    )

    results = {
        "calc_date":format_date(calc_dt),
        "values":window_str,
        "comment":comment,
    }
    return results


def count_valid_calls(df: pd.DataFrame) -> int:
    """Count rows where validated == 'present' (case-insensitive, trimmed)."""
    if df.empty:
        return 0
    vals = df["validated"].astype(str).str.strip().str.lower()
    return (vals == "present").sum()


def fill_date_delta(orig_date, calc_date, comment) -> str:
    result = ""
    if comment:
        orig_dt = pd.to_datetime(orig_date, errors="coerce", format="%m/%d/%Y",)
        calc_dt = pd.to_datetime(calc_date, errors="coerce", format="%m/%d/%Y",)

        if pd.isna(calc_dt):
            result = "Not in range"
        elif orig_dt != calc_dt:
            result = str((orig_dt - calc_dt).days)
        else:
            result = "equal"  # dates are the same, no comment

    return result



def validate_hatch(row:pd.Series)->dict:
    # given a hatch date and a site, check the 7 days around it for first day with count of tags >= 4    
    results_dict = {}

    # Validate that the hatch date matches
    hatch_date = row[HATCH_COL]
    orig_hatch_date = parse_date_for_index(hatch_date)
    if orig_hatch_date is None:
        results_dict["Hatch_Comment"] = f"Hatch date was not a date"

    else:
        # Get the appropriate site data
        site_name = row["Name"]
        df = load_data_for_site(site_name)

        # Get the tags we need to filter on
        pulse = row[PULSE_COL][-2:]
        p_tag, song_tag = TAG_MAP[pulse]

        # Filter the data to ones that have both tags as >= 1 and then count the number of recordings per day
        daily_counts = count_paired_hits_per_day(df, p_tag, song_tag)

        results = check_if_hatch_dates_match(daily_counts, orig_hatch_date)

        #Make a dict of what we learned
        results_dict={
            "Calc_Hatch_Date": results["calc_date"],
            "Hatch_Comment": results["comment"],
            #"Hatch_Window_Values": results["values"],
            "Hatch_Delta" : fill_date_delta(orig_hatch_date, results["calc_date"], results["comment"])
        }
    return results_dict



def validate_male_chorus(row:pd.Series, recs_per_day:pd.DataFrame)->dict:
    def get_male_chorus_per_day() -> pd.Series:
        site_name = row["Name"]
        df = load_pm_data_for_site(site_name, PMType.MC)

        # Count the number of recordings/day that have "validated"="present"
        daily_counts = count_present_per_day(df)

        # Scale this by recording effort
        v = get_recording_vectors(daily_counts, recs_per_day)
        if v["normalized"] is None:
            return pd.Series() # No MC found
        return v["normalized"]

    def is_valid_bout(group):
        # Requirement: >= 3 active days within any 5-day span
        # We apply a rolling count on the original mask within this specific bout
        original_active_days = active_mask.loc[group.index]
        
        if len(original_active_days) < 3:
            return False
            
        # Check if there is ANY 5-day window in this bout with >= 3 active days
        # (If the bout is shorter than 5 days, it just sums the whole thing)
        density_check = original_active_days.rolling(window=5, min_periods=1).sum()
        return (density_check >= 3).any()

    ''' 
    Identify start of prospecting 
    '''
    results = {
        "Calc_MC_Date":"",
        "MC_msg":""

    }
    mc_per_day = get_male_chorus_per_day()
    if len(mc_per_day) == 0:
        results["MC_msg"] = "No male chorus"
        return results 
    
    # Step 1: Define the search window. It will end at hatch-1, and begin at hatch-w. 
    # Limit it to only days with male chorus in that window
    inc_start_date = row[INC_START_COL]
    if inc_start_date == "ND":
        results["MC_msg"] = "No inc start date"
        return results 
    elif inc_start_date == "pre":
        results["Calc_MC_Date"] = "pre"
        results["MC_msg"] = "inc_start was pre"
        return results
    
    window = 60 - 1 #for inclusive
    inc_start_date_date = date_str_to_date(inc_start_date)
    if inc_start_date_date is None:
        results["MC_msg"] = "No inc start date"
        return results 

    max_date = inc_start_date_date - pd.Timedelta(days=1)
    # Pick the latest date that's in our data set: so if max_date = Jan 31 but mc_per_day ends on Jan 28, give Jan 28
    max_mc = min(mc_per_day.index.max(), max_date)
    min_date = max_date - pd.Timedelta(days=window) 
    # Pick the earliest date that's in our data set: so if min_date = Jan 1 but mc_per_day starts on Feb 1, give Feb 1
    min_mc = max(mc_per_day.index.min(), min_date)

    # Reindex to a complete daily range
    full_idx = pd.date_range(min_mc, max_mc, freq="D")    
    if len(full_idx) == 0:
        results["MC_msg"] = "No MC in range"
        return results

    #Calculate the threshold that would be "local to" the breeding stage
    threshold_window = 14-1
    min_date_for_threshold = max_date - pd.Timedelta(days=threshold_window)
    threshold_earliest = max(mc_per_day.index.min(), min_date_for_threshold) 
    threshold_date_idx = pd.date_range(threshold_earliest, max_mc, freq="D")

    # Step 2: Determine the threshold, want the median of the 7 noisiest days
    median_window = 7 #Looking for the 7 noisiest days to set the threshold
    threshold_ratio = 0.2
    mc_per_day_in_threshold_window = mc_per_day.reindex(threshold_date_idx)
    k = min(median_window, len(mc_per_day_in_threshold_window))
    median_noisy = mc_per_day_in_threshold_window.nlargest(median_window).median()
    threshold = median_noisy * threshold_ratio

    # Step 3: Determine "bouts" of active days
    # Mask to find any active days
    mc_per_day_in_window = mc_per_day.reindex(full_idx)
    active_mask = mc_per_day_in_window >= threshold
    # Find windows of minimum length
    min_bout_length = 3

    # 1. Use a rolling window to find gaps, but only fill if surrounded
    # We look for a 3-day window where the sum is >= 2 (meaning 2 out of 3 days are True)
    # This naturally ignores a "False True False" because the sum is only 1.
    # bridged_mask = active_mask.rolling(window=min_bout_length, center=True, min_periods=1).sum() >= 1

    # 2. TO FIX THE "SMEAR": Only allow a False to become True if it was 
    # specifically a gap of 1 day between two Trues.
    is_gap_of_1 = (~active_mask) & active_mask.shift(1) & active_mask.shift(-1)
    refined_mask = active_mask | is_gap_of_1

    # # If you want to allow gaps of up to 2 days:
    # is_gap_of_2 = (~active_mask) & active_mask.shift(2) & active_mask.shift(-1) # e.g., T F F T
    # is_gap_of_2_alt = (~active_mask) & active_mask.shift(1) & active_mask.shift(-2)

    # refined_mask_2 = active_mask | is_gap_of_1 | is_gap_of_2 | is_gap_of_2_alt

    # Identify the start of each new bout
    bout_ids = (refined_mask != refined_mask.shift()).cumsum()
    # Filter to only keep the True regions
    bouts = bout_ids[refined_mask]
    def density_filter(group_idx):
        # Pull the original un-bridged data for these indices
        data_chunk = active_mask.loc[group_idx]
        if len(data_chunk) < 3: return False
        
        # Requirement: >= 2 active days within ANY 3-day span in this bout
        return (data_chunk.rolling(min_bout_length, min_periods=1).sum() >= min_bout_length-1).any()
    
    # Get the final valid bouts
    final_bouts = bouts.groupby(bouts).filter(lambda x: density_filter(x.index))
    
    # # Group by the bout IDs and filter
    # valid_bout_ids = bouts.groupby(bouts).filter(is_valid_bout).unique()

    if not final_bouts.empty:
        last_bout_id = final_bouts.iloc[-1]
        last_bout_indices = bouts[bouts == last_bout_id].index        
        last_bout_start = last_bout_indices[0]
        last_bout_end = last_bout_indices[-1]

        #Calculate the earliest possible date for data and see if this is the same
        if last_bout_start == mc_per_day.index.min():
            results[f"Calc_MC_Date"] = "pre"
            results[f"MC_msg"] = "Last bout started on day 1"
        else:
            results[f"Calc_MC_Date"] = (last_bout_start + pd.Timedelta(days=0)).strftime("%Y-%m-%d")
            results[f"MC_msg"] = "OK"
            orig_mc = row[MC_START_COL]
            orig_mc_date = date_str_to_date(orig_mc)
            if not orig_mc_date is None:
                results["Calc_MC_Delta"] = str((orig_mc_date - last_bout_start).days)
    else:
        results["MC_msg"] = "No qualifying bouts found"

    return results


def validate_fledge(row:pd.Series, recs_per_day:pd.DataFrame)->dict:
    #given a fledge date and a site, check the N days around it for the first day with a count of recs >=4

    results_dict = {}
    check_fledge = True
    check_dispersal = True

    # check whether we have original dates to work with
    fledge_date = row[FLEDGE_START_COL]
    orig_fledge_date = parse_date_for_index(fledge_date)
    if orig_fledge_date is None:
        results_dict["Fledge_Comment"] = f"Fledge date was not a date"
        check_fledge = False

    dispersal_date = row[FLEDGE_DISP_COL]
    orig_dispersal_date = parse_date_for_index(dispersal_date)
    if orig_dispersal_date is None:
        results_dict["Dispersal_Comment"] = f"Dispersal date was not a date"
        check_dispersal = False

    site_name = row["Name"]
    # Count the number of recordings/day that have "validated"="present"
    df = load_pm_data_for_site(site_name, PMType.FLEDGLING)
    daily_counts = count_present_per_day(df)
    v = get_recording_vectors(daily_counts, recs_per_day)
    sc = {"is_small": False}
    if not v["detections_per_day"] is None and not v["recordings_per_day"] is None:
        sc = is_small_colony(v)
        results_dict["log"] = sc["log"]
        results_dict["rec_count"] = sc["recordings"]
        results_dict["detections"] = sc["detections"]

    if check_fledge or check_dispersal:
        if check_fledge:
            #Check and save the results
            results = check_if_fledge_dates_match(daily_counts, orig_fledge_date)
            results_dict["Calc_Fledge_Date"] = results["calc_date"]
            results_dict["Fledge_Comment"] = results["comment"]
            #results_dict["Fledge_Window_Values"] = results["values"]
            results_dict["Fledge_Delta"] = fill_date_delta(orig_fledge_date, results["calc_date"], results["comment"])

        if check_dispersal:
            type_list = []
            if not orig_fledge_date is None:
                target_date = orig_fledge_date
                window = daily_counts.loc[daily_counts.index >= target_date] 
                v = get_recording_vectors(window, recs_per_day)
                
                #If a colony is Small, try different strategies
                breeding_type = f'{row["Breeding Type"]}, {row["Complex Types"]}'
                if sc["is_small"]:
                    type_list.append("small")
                    #If a colony is "small"/0-25% percentile of log(detections/recording), the window should be 
                    #   a) Last date with a recording after fledgestart

                    s = v["detections_per_day"]
                    if "Asynchronous" in breeding_type:
                        max_window = 89
                    else:
                        max_window = 29 #+29 to get a 30-day window
                    start = s.index.min()
                    end_cap = min(start + pd.Timedelta(days=max_window), s.index.max()) 

                    last_date = s.loc[start:end_cap].loc[s != 0].index.max()
                    last_date = last_date + pd.Timedelta(days=1)  #Need the first unoccupied day, that will be the one after the last one with a detection
                    results_dict[f"Calc_Dispersal_Date"] = last_date.strftime("%Y-%m-%d")
                    delta = fill_date_delta(orig_dispersal_date, last_date, "small")
                    results_dict[f"Dispersal_Delta"] = "0" if delta == "equal" else delta

                else:
                    if "Asynchronous" in breeding_type:
                        #If a colony is asynchronous the window to check should be FledgeStart to the end of the dataset
                        type_list.append("async")
                        drop = 0.1
                        window = 89
                    else:
                        drop = 0.1
                        window = 29
                    results = check_dispersal_model2(v["normalized"],  v["detections_per_day"],
                                                        orig_dispersal_date,
                                                        median_window=7,
                                                        moving_median_window=3,
                                                        drop_reqd=drop,
                                                        days_absent_per_week=6,
                                                        max_window_days=window)

                    results_dict[f"Calc_Dispersal_Date"] = results["dispersal_date"]
                    results_dict[f"Dispersal_Comment"] = results["message"]
                    results_dict[f"Dispersal_Delta"] = results["delta_days_vs_orig"]           
            else:
                results_dict["Dispersal_Comment"] = f"Couldn't check, no fledge start date"

            results_dict["type"] = ";".join(type_list)
    return results_dict


def get_recs_per_day(row:pd.Series):
    DAILY_COUNTS_PARQUET = BASE_DIR / "recordings_per_day.parquet"
    df = pd.read_parquet(DAILY_COUNTS_PARQUET)    
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df["n_recordings"] = df["n_recordings"].astype("int64")
    df = df[df["site"]==row["Name"]]
    return df


def get_threshold(df: pd.DataFrame):
    mode =  df["n_recordings"].mode()
    mode_val = mode[0]
    mode_val = int(np.round(mode_val * 4/72))   #Scale it to match our current value
    return mode_val


def validate():
    global threshold
    print("Loading breeding dates table...")
    breeding_dates_df = load_breeding_dates_table()

    all_results = []

    for _, row in breeding_dates_df.iterrows():   
        print(f"Working on {row[PULSE_COL]}")
        results_row = make_output_file_row(row)

        if row["Site ID"] == "65481":
            pass #for debugging, set breakpoint here.

        recs_per_day = get_recs_per_day(row)
        threshold = get_threshold(recs_per_day)
        results_row["Threshold"] = threshold

        has_tilde = row.str.contains("~", regex=False, na=False).any() #TODO log which columns have the tilde
        results_row["Twiddle"] = "Twiddle" if has_tilde else ""

        validation_results = validate_male_chorus(row, recs_per_day)
        results_row.update(validation_results)

        validation_results = validate_hatch(row) 
        results_row.update(validation_results)

        # Validate that the fledging date matches
        validation_results = validate_fledge(row, recs_per_day)
        results_row.update(validation_results)

        all_results.append(results_row)
    
    results_df = pd.DataFrame(all_results)
    save_csv_with_retry(results_df, OUTPUT_CSV)
    save_csv_with_retry(results_df, SHARING_OUTPUT_CSV)

if __name__ == "__main__":
    validate()



