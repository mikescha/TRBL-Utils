from __future__ import annotations

from pathlib import Path
from datetime import timedelta
import re

import pandas as pd


# -----------------
# Paths / inputs
# -----------------

BASE_DIR = Path(".")
TRACKING_CSV = BASE_DIR / "breeding dates.csv"

# Root of TRBLSummarizer repo (used for denominators + PMJ Data)
# Use forward slashes to avoid escape issues on Windows.
DATA_DIR = Path("C:/Users/mikes/OneDrive/Documents/GitHub/TRBLSummarizer/TRBLSummarizer")
RAW_DATA_DIR = DATA_DIR / "Data"  # contains data 2017.csv ... data 2024.csv
PMJ_DIR = DATA_DIR / "PMJ Data"

# Optional: precomputed daily recording-count tables (recommended).
# Prefer Parquet if present, fall back to CSV.
# Expected columns: site, date, n_recordings
DAILY_COUNTS_PARQUET = BASE_DIR / "recordings_per_day.parquet"
DAILY_COUNTS_CSV = RAW_DATA_DIR / "recordings_per_day.csv"

INSECT_OUTPUT_CSV = BASE_DIR / "insects.csv"
PTF_OUTPUT_CSV = BASE_DIR / "ptfs.csv"


# -----------------
# Tracking CSV columns
# -----------------

NAME_SOURCE_COL = "Name"          # site name
PULSE_NAME_COL = "Pulse Name"     # pulse name
HATCH_DATE_COL = "hatch"          # renamed internally to "Hatch Date"
FLEDGE_START_COL = "fledgestart"  # used when hatch line == 'pre'

HATCH_DATE_PRE_TOKEN = "pre"


# -----------------
# Organisms
# -----------------

INSECT_TYPES: tuple[str, ...] = ("Insect 30", "Insect 31", "Insect 32", "Insect 33")
FROG_TYPES: tuple[str, ...] = ("Pacific Tree Frog",)

ORGANISMS: dict[str, dict] = {
    "PTF": {"types": FROG_TYPES, "output_csv": PTF_OUTPUT_CSV},
    "Insect": {"types": INSECT_TYPES, "output_csv": INSECT_OUTPUT_CSV},
}

WINDOW_DAYS = 11  # inclusive; window_end = start + WINDOW_DAYS


# -----------------
# Parsing helpers
# -----------------

def parse_hatch_date_line(site_name: str, raw_line: str) -> pd.Timestamp | None:
    """Parse a single hatch-date line into a normalized Timestamp.

    Cleanup rules:
      - Ignore blank / ND
      - Strip leading '~' and trailing '*'
      - Strip trailing parenthetical codes like ' (r)', ' (m1)' (1–2 alnum)

    Returns normalized Timestamp or None.
    """
    if raw_line is None:
        return None

    s = str(raw_line).strip()
    if not s or s.upper() == "ND":
        return None

    s = s.lstrip("~").rstrip("*").strip()
    s = re.sub("\\s*\\([A-Za-z0-9]{1,2}\\)$", "", s).strip()

    try:
        dt = pd.to_datetime(s, errors="raise", dayfirst=False)
    except Exception:
        valid = ["post"]
        if raw_line in valid:
            print(f"[ERROR] Bad hatch date for '{site_name}': {raw_line!r}")
        return None

    return dt.normalize()


def fmt_date(d: pd.Timestamp | None) -> str:
    if d is None or pd.isna(d):
        return "n/a"
    return d.strftime("%Y-%m-%d")


def ratio(present: int, total: int) -> str | float:
    if total <= 0:
        return "n/a"
    return round(present / total, 4)


def count_present(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    v = df["validated"].astype(str).str.strip().str.lower()
    return int((v == "present").sum())


# -----------------
# Tracking table
# -----------------

def load_tracking_table() -> pd.DataFrame:
    """Load tracking CSV and return standardized columns.

    Required columns:
      - NAME_SOURCE_COL
      - HATCH_DATE_COL

    Optional (used if present):
      - PULSE_NAME_COL
      - FLEDGE_START_COL
    """
    print(f"Loading breeding dates: {TRACKING_CSV}")
    df = pd.read_csv(TRACKING_CSV, dtype=str)

    required = [NAME_SOURCE_COL, HATCH_DATE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in tracking CSV: {missing}")

    df = df.rename(columns={HATCH_DATE_COL: "Hatch Date"})

    for optional in (PULSE_NAME_COL, FLEDGE_START_COL):
        if optional not in df.columns:
            df[optional] = ""

    return df


# -----------------
# Call table loaders
# -----------------

def load_call_table(path: Path) -> pd.DataFrame:
    """Load a call CSV with columns: year, month, day, validated, recording; add normalized 'date'.

    Returns empty DF with correct columns if file missing.
    """
    if not path.exists():
        print(f"[WARN] Call file not found: {path}")
        return pd.DataFrame(columns=["year", "month", "day", "validated", "recording", "date"])

    print("Loading call file:", path)
    df = pd.read_csv(
        path,
        dtype={"year": int, "month": int, "day": int, "validated": str, "recording": str},
    )

    for col in ("year", "month", "day", "validated", "recording"):
        if col not in df.columns:
            raise ValueError(f"File {path} is missing required column '{col}'.")

    date_str = df["year"].astype(str) + "-" + df["month"].astype(str) + "-" + df["day"].astype(str)
    df["date"] = pd.to_datetime(date_str, errors="coerce").dt.normalize()

    bad = int(df["date"].isna().sum())
    if bad:
        print(f"[WARN] {bad} rows in {path} have invalid year/month/day and were ignored.")
        df = df.dropna(subset=["date"]).copy()

    return df.sort_values("date").reset_index(drop=True)


# -----------------
# Denominator helpers (total recordings per window)
# -----------------

def site_year(site_name: str) -> int:
    """Extract leading 4-digit year from site name like '2017 Rush Ranch'."""
    m = re.match("^(\\d{4})\\b", site_name.strip())
    if not m:
        raise ValueError(f"Site name does not start with a 4-digit year: {site_name!r}")
    return int(m.group(1))


_daily_counts_df: pd.DataFrame | None = None


def load_daily_counts() -> pd.DataFrame | None:
    """Load precomputed recordings_per_day table.

    Preference order:
      1) recordings_per_day.parquet (fastest)
      2) recordings_per_day.csv

    Expected columns: site, date, n_recordings
    """
    global _daily_counts_df
    if _daily_counts_df is not None:
        return _daily_counts_df

    if DAILY_COUNTS_PARQUET.exists():
        df = pd.read_parquet(DAILY_COUNTS_PARQUET)
    elif DAILY_COUNTS_CSV.exists():
        df = pd.read_csv(DAILY_COUNTS_CSV, dtype={"site": str, "n_recordings": int})
    else:
        return None

    if not {"site", "date", "n_recordings"}.issubset(df.columns):
        raise ValueError("Daily counts table must contain columns: site, date, n_recordings")

    df = df[["site", "date", "n_recordings"]].copy()
    df["site"] = df["site"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df["n_recordings"] = df["n_recordings"].astype("int64")

    _daily_counts_df = df
    return _daily_counts_df


def count_total_recordings(site_name: str, window_start: pd.Timestamp, window_end: pd.Timestamp) -> int:
    """Count total recordings for a site in [window_start, window_end] (inclusive).

    Uses recordings_per_day.(parquet|csv).
    """
    daily = load_daily_counts()
    total = 0
    if daily is not None:
        df_site = daily.loc[
            (daily["site"] == site_name)
            & (daily["date"] >= window_start)
            & (daily["date"] <= window_end)
        ]
        total = int(df_site["n_recordings"].sum())

#None of this should be neccesary as I'll make sure we have the parquet file
    # yr = site_year(site_name)
    # raw_path = RAW_DATA_DIR / f"data {yr}.csv"

    # total = 0
    # usecols = ["site", "day", "month", "year"]

    # for chunk in pd.read_csv(
    #     raw_path,
    #     usecols=usecols,
    #     dtype={"site": str, "day": int, "month": int, "year": int},
    #     chunksize=2_000_000,
    # ):
    #     chunk = chunk[chunk["site"] == site_name]
    #     if chunk.empty:
    #         continue

    #     date_str = chunk["year"].astype(str) + "-" + chunk["month"].astype(str) + "-" + chunk["day"].astype(str)
    #     chunk_dates = pd.to_datetime(date_str, errors="coerce").dt.normalize()
    #     mask = (chunk_dates >= window_start) & (chunk_dates <= window_end)
    #     total += int(mask.sum())

    return total


# -----------------
# Summarization
# -----------------

def summarize_organisms_for_window(
    site_name: str,
    pulse_name: str,
    date_type_used: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    org_key: str,
    organism_cfg: dict,
) -> dict:
    """Summarize detections for a site + time window.

    Denominator (TOTAL) is total recordings in the window from TRBLSummarizer data.
    Numerators (PRESENT) come from validated=='present' in the organism call files.
    """

    org_types: tuple[str, ...] = organism_cfg["types"]

    total_recordings_window = count_total_recordings(site_name, window_start, window_end)

    per_type_total: dict[str, int] = {}
    per_type_present: dict[str, int] = {}

    earliest_any: pd.Timestamp | None = None

    # For present_all_types (dedupe by recording)
    window_rows_for_present: list[pd.DataFrame] = []

    for t in org_types:
        print(f"Loading organism calls for {t}...")
        path = PMJ_DIR / site_name / f"{site_name} {t}.csv"

        df = load_call_table(path)
        mask = (df["date"] >= window_start) & (df["date"] <= window_end)
        df2 = df.loc[mask, ["recording", "validated", "date"]].copy().assign(organism_type=t)

        pres = count_present(df2)

        per_type_total[t] = int(total_recordings_window)
        per_type_present[t] = int(pres)

        if not df2.empty:
            dmin = df2["date"].min()
            if earliest_any is None or dmin < earliest_any:
                earliest_any = dmin
            window_rows_for_present.append(df2[["recording", "validated"]])

    out: dict[str, object] = {
        "Site Name": site_name,
        "Pulse Name": pulse_name,
        "Date Used": date_type_used,
        "Window Start": fmt_date(window_start),
        "Window End": fmt_date(window_end),
        "Earliest Rec": fmt_date(earliest_any),
    }

    for t in org_types:
        out[f"Total Recordings in Window"] = per_type_total.get(t, 0)
        out[f"Total {t} Calls Present"] = per_type_present.get(t, 0)
        out[f"Ratio {t}"] = ratio(per_type_present.get(t, 0), per_type_total.get(t, 0))

    if window_rows_for_present:
        all_df = pd.concat(window_rows_for_present, ignore_index=True)
        v = all_df["validated"].astype(str).str.strip().str.lower()
        present_unique = int(all_df.loc[v == "present", "recording"].nunique())
    else:
        present_unique = 0

    if len(org_types) > 1:
        out[f"Total Recordings All Types"] = int(total_recordings_window)
        out[f"Total {org_key} Calls Present All Types"] = int(present_unique)
        out["Ratio All Types"] = ratio(present_unique, int(total_recordings_window))

    return out


def _audit_row(site_name: str, pulse_name: str, org_key: str, org_types: tuple[str, ...], window_start_raw: str) -> dict:
    row = {
        "Site Name": site_name,
        "Pulse Name": pulse_name,
        "Date Used": "n/a",
        "Window Start": window_start_raw,
        "Window End": "n/a",
        "Earliest Rec": "n/a",
        **{f"Total Recordings in Window": 0 for t in org_types},
        **{f"Total {t} Calls Present": 0 for t in org_types},
        **{f"Ratio {t}": "n/a" for t in org_types},
        f"Total Recordings All Types": 0,
        f"Total {org_key} Calls Present All Types": 0,
        "Ratio All Types": "n/a",
    }
    if len(org_types) == 1:
        del row["Total Recordings All Types"]
        del row[f"Total {org_key} Calls Present All Types"]
        del row["Ratio All Types"]
        
    return row


# -----------------
# Main
# -----------------

def make_critter_ratios_file() -> None:
    tracking_df = load_tracking_table()

    for org_key, cfg in ORGANISMS.items():
        org_types: tuple[str, ...] = cfg["types"]
        results: list[dict] = []

        for _, row in tracking_df.iterrows():
            site_name = str(row.get(NAME_SOURCE_COL, "")).strip()
            pulse_name = str(row.get(PULSE_NAME_COL, "")).strip()
            hatch_cell = row.get("Hatch Date", "")

            if not site_name:
                continue

            if pd.isna(hatch_cell) or not str(hatch_cell).strip():
                continue

            for raw_line in str(hatch_cell).splitlines():
                raw_s = str(raw_line).strip()

                if raw_s.lower() == HATCH_DATE_PRE_TOKEN:
                    fledge_cell = row.get(FLEDGE_START_COL, "")
                    if pd.isna(fledge_cell) or not str(fledge_cell).strip() or str(fledge_cell).strip().upper() == "ND":
                        print(f"[ERROR] Hatch Date line is 'pre' but Fledge Start is missing/ND for '{site_name}'.")
                        results.append(_audit_row(site_name, pulse_name, org_key, org_types, raw_s))
                        continue

                    try:
                        fledge_dt = pd.to_datetime(str(fledge_cell).strip(), errors="raise", dayfirst=False).normalize()
                    except Exception:
                        valid = ["pre"]
                        if fledge_cell in valid:
                            print(f"[ERROR] Bad Fledge Start for '{site_name}': {fledge_cell!r}")
                            results.append(_audit_row(site_name, pulse_name, org_key, org_types, raw_s))
                        continue

                    window_end = fledge_dt
                    window_start = fledge_dt - timedelta(days=WINDOW_DAYS)
                    date_type_used = "Fledge"

                    results.append(
                        summarize_organisms_for_window(
                            site_name=site_name,
                            pulse_name=pulse_name,
                            date_type_used=date_type_used,
                            window_start=window_start,
                            window_end=window_end,
                            org_key=org_key,
                            organism_cfg=cfg,
                        )
                    )
                    continue

                hatch_dt = parse_hatch_date_line(site_name, raw_line)
                if hatch_dt is None:
                    results.append(_audit_row(site_name, pulse_name, org_key, org_types, raw_s))
                    continue

                window_start = hatch_dt
                window_end = hatch_dt + timedelta(days=WINDOW_DAYS)
                date_type_used = "Hatch"

                results.append(
                    summarize_organisms_for_window(
                        site_name=site_name,
                        pulse_name=pulse_name,
                        date_type_used=date_type_used,
                        window_start=window_start,
                        window_end=window_end,
                        org_key=org_key,
                        organism_cfg=cfg,
                    )
                )

        df_out = pd.DataFrame(results)

        if not df_out.empty:
            int_cols = [c for c in df_out.columns if c.startswith("Total ")]
            df_out[int_cols] = df_out[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")

        output_file: Path = cfg["output_csv"]
        df_out.to_csv(output_file, index=False)
        print(f"[DONE] Wrote results to: {output_file}")


if __name__ == "__main__":
    make_critter_ratios_file()
