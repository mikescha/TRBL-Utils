import numpy as np
import pandas as pd

df1_path = r"C:\Users\mikes\GitHub\TRBL-Utils\TRBL Analysis Tracking - All(new sheet converted to old format).csv"
df2_path = r"C:\Users\mikes\GitHub\TRBL-Utils\TRBL Analysis Tracking - All.csv"
df2_path = r"C:\Users\mikes\GitHub\TRBLSummarizer\TRBLSummarizer\Data\TRBL Analysis tracking - All.csv"

# Load the data, setting "Name" as the index to align rows regardless of order
# (If these files have the 2 blank lines from your previous question, add `skiprows=2` here)
df1 = pd.read_csv(df1_path, index_col="Name", skiprows=2, keep_default_na=False, encoding="utf-8")
df2 = pd.read_csv(df2_path, index_col="Name", skiprows=2, keep_default_na=False, encoding="utf-8")

#map old column names to new column names in df1
col_map = {
    "Id":"Site ID",
    "p1outcome":"p1Outcome",
    "p2outcome":"p2Outcome",
    "p3outcome":"p3Outcome",
    "p4outcome":"p4Outcome",
    "Skip":"Skip Site"
}
df1.rename(columns=col_map, inplace=True)

# Find only the rows and columns that exist in BOTH dataframes
common_rows = df1.index.intersection(df2.index)
common_cols = df1.columns.intersection(df2.columns)

# Slice BOTH dataframes down to this exact shared grid
df1_aligned = df1.loc[common_rows, common_cols]
df2_aligned = df2.loc[common_rows, common_cols]

# --- ADD NORMALIZATION STEPS HERE ---

# 1. Allow 'ND' == 'n/a' (and 'n/a' == 'ND')
# We just force both DataFrames to use "n/a" uniformly
#df1_aligned = df1_aligned.replace("ND", "n/a")
df2_aligned = df2_aligned.replace("n/a", "ND")
df2_aligned = df2_aligned.map(lambda x: x.strip() if isinstance(x, str) else x)

# 2. Allow '-122.014' == '-122.0140'
# Convert columns to numbers where possible. 
# Pandas will evaluate trailing zeros as mathematically equal (e.g., -122.014 == -122.0140).
# The 'ignore' error flag means if a column is purely text (like 'Site Notes'), it safely skips it.
for col in common_cols:
    # Convert what can be converted to numbers, turning text/strings into NaN
    num1 = pd.to_numeric(df1_aligned[col], errors='coerce')
    num2 = pd.to_numeric(df2_aligned[col], errors='coerce')
    
    # Fill the NaNs back in with the original string data
    df1_aligned[col] = num1.fillna(df1_aligned[col])
    df2_aligned[col] = num2.fillna(df2_aligned[col])

# Create a boolean mask of differences (ignoring NaN == NaN)
differences = (df1_aligned != df2_aligned) & ~(df1_aligned.isna() & df2_aligned.isna())

# Get the row and column integer coordinates of discrepancies
diff_rows, diff_cols = np.where(differences)

if len(diff_rows) == 0:
    print("No differences found in the overlapping data!")
else:
    print(f"Found {len(diff_rows)} differences:\n")

    # Create an empty list to store the rows for our new CSV
    diff_records = []

    for r, c in zip(diff_rows, diff_cols, strict=True):
        r, c = int(r), int(c)  #To satisfy the linter
        row_name = df1_aligned.index[r]
        col_name = df1_aligned.columns[c]
        
        val1 = df1_aligned.iloc[r, c]
        val2 = df2_aligned.iloc[r, c]
        
        print(f"Name: '{row_name}' | Column: '{col_name}'")
        print(f"  -> df1 value : {val1}")
        print(f"  -> df2 value : {val2}\n")
        diff_records.append({
            "Name": row_name,
            "Column": col_name,
            "df1_value": val1,
            "df2_value": val2
        })

    # Save the differences to a CSV for further inspection
    if diff_records:
        diff_df = pd.DataFrame(diff_records)
        diff_df.to_csv("differences.csv", index=False, encoding="utf-8")
        print("Differences saved to 'differences.csv'")