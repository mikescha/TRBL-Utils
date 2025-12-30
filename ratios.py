import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import re

DAYS_TO_COUNT = 10
NESTLING_OFFSET_DAYS = 2

BASE_DIR = Path(".")
DATA_DIR = Path("C:\\Users\\mikes\\OneDrive\\Documents\\GitHub\\TRBLSummarizer\\TRBLSummarizer\\")
BREEDING_DATES_CSV = BASE_DIR / "breeding dates.csv"
PMJ_DIR = DATA_DIR / "PMJ Data"
OUTPUT_CSV = DATA_DIR / "nestling-to-female-ratios.csv"

# Source column name for site in the tracking CSV (row-2 headers)
NAME_SOURCE_COL = "Name"
HATCH_COL = "hatch"
OUTCOME_COL = "Outcome"
PULSE_COL = "Pulse Name"


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

    df = df[expected_cols]
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
    name: str,
    hatch_date: pd.Timestamp,
    female_df: pd.DataFrame,
    nestling_df: pd.DataFrame,
    pulse: str, 
    outcome: str
) -> dict:
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
        # NOTE: per your original spec, this is still days from hatch_date,
        # not from nestling_start, even though the window starts +2 days.
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

    # Ratio
    if avg_female_per_day is None or avg_female_per_day == 0 or avg_nestling_per_day is None:
        ratio = None
    else:
        ratio = avg_nestling_per_day / avg_female_per_day 

    def fmt_date(d: pd.Timestamp | None) -> str:
        if d is None or pd.isna(d):
            return "n/a"
        return d.strftime("%Y-%m-%d")

    def fmt_number(x):
        if x is None:
            return "n/a"
        return x

    return {
        "Site Name": name,
        PULSE_COL : pulse,
        OUTCOME_COL : outcome,
        "Hatch Date": fmt_date(hatch_date),
        "Earliest Rec": fmt_date(earliest_rec),
        "Incubation Days": incubation_days,
        "Total Female Calls": total_female_calls,
        "Avg Female Calls/Day": fmt_number(
            round(avg_female_per_day, 3) if avg_female_per_day is not None else None
        ),
        "Latest Rec": fmt_date(latest_rec),
        "Nestling Days": nestling_days,
        "Total Nestling Calls": total_nestling_calls,
        "Avg Nestling Calls/Day": fmt_number(
            round(avg_nestling_per_day, 3) if avg_nestling_per_day is not None else None
        ),
        "Ratio": fmt_number(round(ratio, 3) if ratio is not None else None),
    }


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

        female_df = load_call_table(female_path)
        nestling_df = load_call_table(nestling_path)

        if female_df.empty and nestling_df.empty:
            print(f"[INFO] No female or nestling data for '{name}'; still outputting rows with 'n/a' where needed.")

        for raw_line in lines:
            hatch_dt = parse_hatch_date_line(name, raw_line)
            if hatch_dt is None:
                # Output a row indicating this line existed but could not be parsed
                results.append({
                    "Site Name": name,
                    PULSE_COL : row[PULSE_COL],
                    OUTCOME_COL : row[OUTCOME_COL],
                    "Hatch Date": str(raw_line).strip() if raw_line != "ND" else "NHD", 
#                    "Hatch Date": str(raw_line).strip(),
                    "Earliest Rec": "n/a",
                    "Incubation Days": 0,
                    "Total Female Calls": 0,
                    "Avg Female Calls/Day": "n/a",
                    "Latest Rec": "n/a",
                    "Nestling Days": 0,
                    "Total Nestling Calls": 0,
                    "Avg Nestling Calls/Day": "n/a",
                    "Ratio": "n/a",
                })
                continue

            summary = summarize_for_hatch_date(name, hatch_dt, female_df, nestling_df, row[PULSE_COL], row[OUTCOME_COL])
            results.append(summary)

    if results:
        results_df = pd.DataFrame(results)
    else:
        columns = [
            "Site Name",
            PULSE_COL,
            OUTCOME_COL,
            "Hatch Date",
            "Earliest Rec",
            "Incubation Days",
            "Total Female Calls",
            "Avg Female Calls/Day",
            "Latest Rec",
            "Nestling Days",
            "Total Nestling Calls",
            "Avg Nestling Calls/Day",
            "Ratio",
        ]
        results_df = pd.DataFrame(columns=columns)

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Wrote results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    make_ratios()

