"""build_recordings_per_day_hour.py

One-time builder for an intermediate table of recording counts per site per day per hour.

Input:
  - Raw TRBLSummarizer exports: "data YYYY.csv" (e.g., data 2017.csv ... data 2024.csv)
    Must include columns: site, day, month, year, hour

Output:
  - recordings_per_day_hour.parquet (recommended) and/or recordings_per_day_hour.csv

Schema:
  - site (str)
  - date (datetime64[ns] normalized to midnight)
  - hour (int 0–23)
  - n_recordings (int)

Usage examples:
  python build_recordings_per_day_hour.py \
    --raw-dir "C:\\Users\\mikes\\OneDrive\\Documents\\GitHub\\TRBLSummarizer\\TRBLSummarizer\\Data" \
    --out-parquet "recordings_per_day_hour.parquet"

  python build_recordings_per_day_hour.py --raw-dir "...\\Data" --out-csv "recordings_per_day_hour.csv"

Dependencies:
  - pandas
  - pyarrow (only if writing parquet): pip install pyarrow

Notes:
  - This script reads very large CSVs. It processes in chunks, aggregates within each chunk,
    then merges into a Python dict keyed by (site, date, hour).
  - If you have sufficient RAM and want higher speed, you can increase chunksize.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Dict, Tuple

import pandas as pd


def discover_year_files(raw_dir: Path) -> list[Path]:
    """Return data YYYY.csv files sorted by year."""
    files = sorted(raw_dir.glob("data *.csv"))

    def year_key(p: Path) -> int:
        m = re.search(r"data\s+(\d{4})\.csv$", p.name)
        if not m:
            return 0
        return int(m.group(1))

    files = sorted([p for p in files if re.search(r"data\s+\d{4}\.csv$", p.name)], key=year_key)
    if not files:
        raise FileNotFoundError(f"No files matching 'data YYYY.csv' found in: {raw_dir}")
    return files


def build_counts(raw_dir: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Scan all 'data YYYY.csv' files and build a (site, date, hour) -> n_recordings table."""

    year_files = discover_year_files(raw_dir)

    # Aggregate counts across all years: (site, date, hour) -> int
    counts: Dict[Tuple[str, pd.Timestamp, int], int] = {}

    usecols = ["site", "date", "hour"]
    dtypes = {"site": "string", "date": "string", "hour": "string"}

    for fpath in year_files:
        print(f"[READ] {fpath}")

        for chunk in pd.read_csv(
            fpath,
            usecols=usecols,
            dtype=dtypes,
            chunksize=chunksize,
        ):
            chunk = chunk.dropna(subset=["site", "date", "hour"])
            # hour values look like '20:40:00' or '9:32:00' -> extract the hour (20, 9)
            h = chunk["hour"].astype("string").str.strip()

            # Take everything before the first ':' (works for '9:32:00', '10:00:00', etc.)
            h = h.str.split(":", n=1).str[0]

            chunk["hour"] = pd.to_numeric(h, errors="coerce")

            # Drop bad hour parses and keep only 0..23
            chunk = chunk.dropna(subset=["hour"])
            chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= 23)]

            chunk["hour"] = chunk["hour"].astype("int16")
            if chunk.empty:
                continue

            # Basic hour sanity filter (keeps only 0..23)
            chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= 23)]
            if chunk.empty:
                continue

            # Construct a normalized date.
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.normalize()
            chunk = chunk.dropna(subset=["date"])
            if chunk.empty:
                continue

            # Chunk-level groupby to reduce size drastically.
            g = chunk.groupby(["site", "date", "hour"], dropna=False).size()

            for (site, date, hour), n in g.items():
                key = (str(site), pd.Timestamp(date), int(hour))
                counts[key] = counts.get(key, 0) + int(n)

    out = pd.DataFrame(
        [(site, date, hour, n) for (site, date, hour), n in counts.items()],
        columns=["site", "date", "hour", "n_recordings"],
    )

    out = out.sort_values(["site", "date", "hour"]).reset_index(drop=True)

    out["site"] = out["site"].astype("string")
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    out["hour"] = pd.to_numeric(out["hour"], errors="raise").astype("int16")
    out["n_recordings"] = pd.to_numeric(out["n_recordings"], errors="raise").astype("int64")

    return out


def build_recordings_per_day_per_hour_file() -> None:
    ap = argparse.ArgumentParser(
        description="Build per-site per-day per-hour recording counts from data YYYY.csv files."
    )

    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(
            "C:/Users/mikes/OneDrive/Documents/GitHub/TRBLSummarizer/TRBLSummarizer/Data"
        ),
        help="Directory containing 'data YYYY.csv' files.",
    )

    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("C:/Users/mikes/OneDrive/Documents/GitHub/TRBLSummarizer/TRBLSummarizer/Data/recordings_per_day_hour.parquet"),
        help="Write output to this Parquet file (recommended).",
    )

    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write output to this CSV file (optional).",
    )

    ap.add_argument(
        "--chunksize",
        type=int,
        default=2_000_000,
        help="Rows per chunk when reading raw CSVs.",
    )

    args = ap.parse_args()

    raw_dir: Path = args.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw-dir does not exist: {raw_dir}")

    if args.out_parquet is None and args.out_csv is None:
        raise ValueError("Specify at least one of --out-parquet or --out-csv")

    df = build_counts(raw_dir=raw_dir, chunksize=args.chunksize)

    if args.out_parquet is not None:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out_parquet, index=False)
        print(f"[WRITE] Parquet: {args.out_parquet}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[WRITE] CSV: {args.out_csv}")

    print(f"[DONE] rows={len(df):,}")


if __name__ == "__main__":
    build_recordings_per_day_per_hour_file()
