"""build_recordings_per_day.py

One-time builder for an intermediate table of recording counts per site per day.

Input:
  - Raw TRBLSummarizer exports: "data YYYY.csv" (e.g., data 2017.csv ... data 2024.csv)
    Located under: <RAW_DATA_DIR>

Output:
  - recordings_per_day.parquet (recommended) and/or recordings_per_day.csv

Schema:
  - site (str)
  - date (datetime64[ns] normalized to midnight)
  - n_recordings (int)

Why this exists:
  The raw data CSVs are very large. Downstream scripts only need the denominator
  (total recordings) per site per day, so precomputing this makes later filters fast.

Usage examples:
  python build_recordings_per_day.py \
    --raw-dir "C:\\Users\\mikes\\OneDrive\\Documents\\GitHub\\TRBLSummarizer\\TRBLSummarizer\\Data" \
    --out-parquet "recordings_per_day.parquet"

  python build_recordings_per_day.py --raw-dir "...\\Data" --out-csv "recordings_per_day.csv"

Dependencies:
  - pandas
  - pyarrow (only if writing parquet): pip install pyarrow
"""

from __future__ import annotations

import argparse
import builtins
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from constants import DATA_DIR


def discover_year_files(raw_dir: Path) -> list[Path]:
    """Return data YYYY.parquet files sorted by year."""
    files = sorted(raw_dir.glob("data *.parquet"))

    def year_key(p: Path) -> int:
        m = re.search(r"data\s+(\d{4})\.parquet$", p.name)
        if not m:
            return 0
        return int(m.group(1))

    files = sorted([p for p in files if re.search(r"data\s+\d{4}\.parquet$", p.name)], key=year_key)
    if not files:
        raise FileNotFoundError(f"No files matching 'data YYYY.parquet' found in: {raw_dir}")
    return files


def build_counts(raw_dir: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Scan all 'data YYYY.parquet' files and build a (site, date) -> n_recordings table."""

    year_files = discover_year_files(raw_dir)

    # Aggregate counts across all years: (site, date) -> int
    counts: builtins.dict[builtins.tuple[str, pd.Timestamp], int] = {}

    usecols = ["site", "date"]

    for fpath in year_files:
        print(f"[READ] {fpath}")

        pf = pq.ParquetFile(fpath)
        # Read in chunks to keep memory bounded.
        for batch in pf.iter_batches(batch_size=chunksize, columns=usecols):
            chunk = batch.to_pandas()
            
            # Drop obvious NA sites (shouldn't happen, but cheap).
            chunk = chunk.dropna(subset=["site", "date"])
            if chunk.empty:
                continue

            # Construct a normalized date.
            # Using string concatenation avoids type-checker complaints about dict[Series].
            date_str = chunk["date"]
            chunk["date"] = pd.to_datetime(date_str, errors="coerce").dt.normalize()
            chunk = chunk.dropna(subset=["date"])
            if chunk.empty:
                continue

            # Chunk-level groupby reduces rows massively before we touch the Python dict.
            g = chunk.groupby(["site", "date"], dropna=False).size()

            # Accumulate into Python dict.
            # Converting to items avoids constructing a huge intermediate DataFrame.
            for (site, date), n in g.items():
                # site is a pandas scalar; cast to str for stable keys.
                key = (str(site), pd.Timestamp(date))
                counts[key] = counts.get(key, 0) + int(n)

    # Convert dict to DataFrame.
    out = pd.DataFrame(
        [(site, date, n) for (site, date), n in counts.items()],
        columns=["site", "date", "n_recordings"],
    )

    # Sort for readability and stable downstream behavior.
    out = out.sort_values(["site", "date"]).reset_index(drop=True)

    # Ensure types are sane.
    out["site"] = out["site"].astype("string")
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    out["n_recordings"] = out["n_recordings"].astype("int64")

    return out


def build_recordings_per_day_file() -> None:
    ap = argparse.ArgumentParser(description="Build per-site per-day recording counts from data YYYY.csv files.")
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing 'data YYYY.parquet' files.",
    )
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("recordings_per_day.parquet"),
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
    build_recordings_per_day_file()
