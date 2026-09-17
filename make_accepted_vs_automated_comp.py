import datetime as dt
import re

import openpyxl
import pandas as pd
import requests
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


def date_difference(value1, value2) -> dt.timedelta | None:
    """Strips non-date characters from both values and returns the difference
    between the resulting dates. Returns None if either value is not a valid date.
    """
    def to_date(value):
        cleaned = re.sub(r"[^0-9-]", "", str(value))
        try:
            return dt.datetime.strptime(cleaned, "%Y-%m-%d").date()
        except ValueError:
            return None

    date1 = to_date(value1)
    date2 = to_date(value2)
    if date1 is None or date2 is None:
        return None
    return date1 - date2


def export_comparison_to_csv(df_baseline: pd.DataFrame, df_results: pd.DataFrame, output_file="dataframe_diff.xlsx"):
    def get_diff_table(df_baseline, df_results, key_col="site"):
        # 1. Outer merge baseline and results on the key column
        merged = pd.merge(
            df_baseline,
            df_results,
            on=key_col,
            how="outer",
            suffixes=("_baseline", "_results"),
        )
        value_cols = [c for c in df_baseline.columns if c != key_col]
        # Build output column list (interleaved side-by-side for easy comparison)
        output_cols = [key_col]
        has_change = pd.Series(False, index=merged.index)
        for col in value_cols:
            b_col = f"baseline_{col}"
            r_col = f"results_{col}"
            merged.rename(
                columns={f"{col}_baseline": b_col, f"{col}_results": r_col},
                inplace=True,
            )
            output_cols.extend([b_col, r_col])
            # Identify cells where baseline equals results (including dual NaNs)
            is_same = (merged[b_col] == merged[r_col]) | (
                merged[b_col].isna() & merged[r_col].isna()
            )
            has_change |= ~is_same
            # Convert to object dtype and blank out matching values
            merged[b_col] = merged[b_col].astype(object)
            merged[r_col] = merged[r_col].astype(object)
            merged.loc[is_same, [b_col, r_col]] = ""
        # Return only rows with at least one difference
        return merged.loc[has_change, output_cols].reset_index(drop=True)
    def style_diff(df_diff):
        return df_diff.style.apply(
            lambda s: [
                "background-color: #EBF3F9"
                if s.name.startswith("baseline_")
                else (
                    "background-color: #E6F4EA"
                    if s.name.startswith("results_")
                    else "background-color: #F2F4F4"
                )
                for _ in s
            ],
            axis=0,
        )
    def diff_dataframes(df_baseline, df_results, key_col="site"):
        """Compares two DataFrames on a key column and returns Added, Removed,
        and Mismatched rows.
        """
        # 1. Outer merge with indicator column
        merged = pd.merge(
            df_baseline,
            df_results,
            on=key_col,
            how="outer",
            suffixes=("_baseline", "_results"),
            indicator=True,
        )
        # 2. Extract Added and Removed rows
        removed = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        added = merged[merged["_merge"] == "right_only"].drop(columns=["_merge"])
        # 3. Filter for rows present in both
        both = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
        # 4. Check for value mismatches across non-key columns
        value_cols = [c for c in df_baseline.columns if c != key_col]
        mismatch_mask = pd.Series(False, index=both.index)
        for col in value_cols:
            b_col, r_col = f"{col}_baseline", f"{col}_results"
            # Flag values that differ, treating dual NaNs as equal
            col_mismatch = (both[b_col] != both[r_col]) & ~(
                both[b_col].isna() & both[r_col].isna()
            )
            mismatch_mask |= col_mismatch
        mismatches = both[mismatch_mask]
        return {"Removed Rows": removed, "Added Rows": added, "Value Changes": mismatches}
    diff_output = get_diff_table(df_baseline, df_results, key_col="site")
    # Export directly to Excel with column colors
    styled_df = style_diff(diff_output)
    with pd.ExcelWriter("dataframe_diff.xlsx", engine="openpyxl") as writer:
        styled_df.to_excel(writer, sheet_name="Value Diff", index=False)


# Function to lowercase all string values in a DataFrame
def normalize_df(df):
    # 1. Replace "(H)" with "(C)" (escaped parens because regex=True)
    df = df.replace(r"\(H\)", "(C)", regex=True)
    # 2. Lowercase and strip whitespace from all string cells
    return df.map(lambda x: x.lower() if isinstance(x, str) else x)


def test_if_same(val1, val2):
    """Compares two values, treating dual NaNs as equal."""

    delta = date_difference(val1, val2) 
    if delta is not None:
        if abs(delta.days) > 3: 
            return "big"
        else:
            return "different"

    if val1 == val2:
        return "equal"
    else:
        return "different"
    #strip "(C)"
    if isinstance(val1, str):
        val1 = val1.replace("(C)", "")
    if isinstance(val2, str):
        val2 = val2.replace("(C)", "")

    #strip "~"
    if isinstance(val1, str):
        val1 = val1.replace("~", "")
    if isinstance(val2, str):
        val2 = val2.replace("~", "")

    if pd.isna(val1) and pd.isna(val2):
        return True
    return val1 == val2


def export_full_data_styled_excel(
    df_baseline,
    df_results,
    key_col="site",
    output_file="dataframe_diff.xlsx",
):
    # Pre-process both DataFrames before running your diff export
    df_baseline = normalize_df(df_baseline)
    df_results = normalize_df(df_results)

    # 1. Outer merge baseline and results
    merged = pd.merge(
        df_baseline,
        df_results,
        on=key_col,
        how="outer",
        suffixes=("_W", "_A"),
    )

    value_cols = [c for c in df_baseline.columns if c != key_col]

    output_cols = [key_col]
    for col in value_cols:
        b_col = f"W_{col}"
        r_col = f"A_{col}"
        merged.rename(
            columns={f"{col}_W": b_col, f"{col}_A": r_col},
            inplace=True,
        )
        output_cols.extend([b_col, r_col])

    full_df = merged[output_cols].fillna("")

    # 2. Build Excel Workbook
    wb = openpyxl.Workbook()
    ws = wb.active

    if ws is None:
        return
    
    ws.title = "Full Data Diff"
    ws.views.sheetView[0].showGridLines = True

    # Color Fills
    key_hdr_fill = PatternFill(
        start_color="34495E", end_color="34495E", fill_type="solid"
    )
    base_hdr_fill = PatternFill(
        start_color="ED4764", end_color="1F4E78", fill_type="solid"
    )
    res_hdr_fill = PatternFill(
        start_color="1E6B52", end_color="1E6B52", fill_type="solid"
    )

    key_cell_fill = PatternFill(
        start_color="F2F4F4", end_color="F2F4F4", fill_type="solid"
    )
    base_cell_fill = PatternFill(
        start_color="F9BEC8", end_color="EBF3F9", fill_type="solid"
    )
    res_cell_fill = PatternFill(
        start_color="E6F4EA", end_color="E6F4EA", fill_type="solid"
    )
    big_diff_cell_fill = PatternFill(
        start_color="F4FF78", end_color="FADBD8", fill_type="solid"
    )


    # All Fonts set to size=8
    hdr_font = Font(name="Segoe UI", size=8, bold=True, color="FFFFFF")
    key_font = Font(name="Segoe UI", size=8, bold=True, color="1C2833")

    diff_font = Font(
        name="Segoe UI", size=8, bold=True, color="1C2833"
    )  # Changed -> 8pt Bold
    same_font = Font(
        name="Segoe UI", size=8, bold=False, color="95A5A6"
    )  # Matching -> 8pt Light Gray

    # Grid borders
    border_side = Side(border_style="thin", color="D5D8DC")
    cell_border = Border(
        left=border_side, right=border_side, top=border_side, bottom=border_side
    )

    # Write Headers
    headers = list(full_df.columns)
    ws.append(headers)

    for col_idx, col_name in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = hdr_font
        cell.alignment = Alignment(
            horizontal="center", vertical="center", wrap_text=False
        )
        cell.border = cell_border

        if col_name == key_col:
            cell.fill = key_hdr_fill
        elif col_name.startswith("W_"):
            cell.fill = base_hdr_fill
        elif col_name.startswith("A_"):
            cell.fill = res_hdr_fill

    # Write Data & Apply Formatting
    for row_idx, row_data in enumerate(full_df.values, 2):
        key_val = row_data[0]
        key_cell = ws.cell(row=row_idx, column=1, value=key_val)
        key_cell.fill = key_cell_fill
        key_cell.font = key_font
        key_cell.alignment = Alignment(horizontal="left", vertical="center")
        key_cell.border = cell_border

        for c_idx, _ in enumerate(value_cols):
            b_idx = 1 + (c_idx * 2)
            r_idx = 1 + (c_idx * 2) + 1

            b_val = row_data[b_idx]
            r_val = row_data[r_idx]

            #is_same = b_val == r_val
            is_same = test_if_same(b_val, r_val)
            target_font = same_font if is_same == "equal" else diff_font
            b_cell = ws.cell(row=row_idx, column=b_idx + 1, value=b_val)
            b_cell.fill = big_diff_cell_fill if is_same == "big" else base_cell_fill
            b_cell.font = target_font
            b_cell.alignment = Alignment(
                horizontal="center", vertical="center"
            )
            b_cell.border = cell_border

            r_cell = ws.cell(row=row_idx, column=r_idx + 1, value=r_val)
            r_cell.fill = big_diff_cell_fill if is_same == "big" else res_cell_fill
            r_cell.font = target_font
            r_cell.alignment = Alignment(
                horizontal="center", vertical="center"
            )
            r_cell.border = cell_border

    # Excel-Style AutoFit Column Widths (Adjusted for 8pt font padding)
    for col in ws.columns:
        max_len = max(len(str(c.value or "")) for c in col)
        column_index = col[0].column
        if column_index is None:
            continue
        col_letter = get_column_letter(column_index)
        ws.column_dimensions[col_letter].width = max(max_len*0.75, 3) if col[0].value != "site" else 17

    wb.save(output_file)
    print(f"Wrote results to '{output_file}'")


DEFAULT_AUTOMATED_RESULTS_DIR = (
    "C:\\Users\\mikes\\GitHub\\TRBL-Breeding-Stages-Final\\data\\accepted_publication_outputs\\"
)

def load_automated_results_from_csv(dir:str = DEFAULT_AUTOMATED_RESULTS_DIR) -> pd.DataFrame:
    """
    Load automated results from a CSV file into a pandas DataFrame.
    
    Args:
        dir (str): Directory containing the CSV file. Defaults to DEFAULT_AUTOMATED_RESULTS_DIR.

    Returns:
        pd.DataFrame: DataFrame containing the loaded automated results.
    """
    file_path = dir + "trbl_breeding_chronology_summary.csv"
    df_all_data = pd.read_csv(file_path)

    #fix up the strings the way we like them
    df_all_data.replace(
        {"no_data": "ND", "inferred_pre_recording": "inf"}, inplace=True
    )

    cols_to_keep = [
        "site","pulse",
        "settlement_start","settlement_end",
        "incubation_onset","brooding_onset","fledging_onset","fledgling_dispersal"
    ]
    df_main_cols = df_all_data[cols_to_keep]

    # 1. Convert columns to string series
    site_str = df_main_cols['site'].astype(str)
    pulse_str = df_main_cols['pulse'].astype(str)

    # 2. Build full string, then replace with just 'site' if pulse is 'no_pulse'
    site_pulse_vals = (site_str + ' ' + pulse_str).mask(pulse_str == 'no_pulse', site_str)

    # 3. Drop the original cols
    df_main_cols = df_main_cols.drop(columns=['site', 'pulse'])

    # 4. Insert the combined column
    df_main_cols.insert(
        0, #position
        'site', #column name
        site_pulse_vals
    )

    return df_main_cols

def load_manual_results_from_csv() -> pd.DataFrame:
    file_path = "C:\\Users\\mikes\\GitHub\\TRBL-Utils\\breeding_dates.csv"
    df_manual_all = pd.read_csv(file_path)

    cols_to_keep = [
        "Pulse_Name", "Breeding_Type", "Complex_Types", "Outcome", 
        "mcstart", "mcend", 
        "incstart", "Hatch_Date", 
        "fledgestart", "fledgedisp", 
        "Abandoned_Date", "Partial_Abandon_Date"
    ]
    df_main_cols = df_manual_all[cols_to_keep]
    name_map = {
        "Pulse_Name": "site",
        "Breeding_Type": "breeding_type",
        "Complex_Types": "complex_types",
        "Outcome": "outcome",
        "mcstart": "settlement_start",
        "mcend": "settlement_end",
        "incstart": "incubation_onset",
        "Hatch_Date": "hatch",
        "fledgestart": "fledging_onset",
        "fledgedisp": "fledgling_dispersal",
        "Abandoned_Date": "abandon",
        "Partial_Abandon_Date": "partial_abandon"
    }
    df_manual = df_main_cols.rename(columns=name_map)

    return df_manual

def load_ARI_score_from_csv() -> pd.DataFrame:
    file_path = "C:\\Users\\mikes\\GitHub\\TRBL-Utils\\nestling_to_female_ratios.csv"
    df_ARI = pd.read_csv(file_path)
    
    return df_ARI



def sync_via_webhook(df: pd.DataFrame, webhook_url: str, sheet_name: str = ""):
    if len(sheet_name) == 0:
        sheet_name = f"automated_results_{dt.datetime.now().strftime('%Y%m%d')}"
    
    # Clean NaN values and convert all columns to string
    df_clean = df.fillna("").astype(str)
    
    # Standardize key column formatting
    if "nest_id" in df_clean.columns:
        df_clean["nest_id"] = df_clean["nest_id"].str.strip().str.upper()
        
    payload = {
        "sheet_name": sheet_name,
        "headers": df_clean.columns.tolist(),
        "rows": df_clean.values.tolist()
    }
    
    # POST JSON payload to Google Apps Script Web App
    response = requests.post(webhook_url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    if result.get("status") == "success":
        print(f"✅ Updated '{sheet_name}' with {result.get('rows_written')} rows via Webhook.")
    else:
        print(f"❌ Apps Script Error: {result.get('message')}")

# --- Usage Example ---
if __name__ == "__main__":
    WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxtzRiML6eQQCyERCQSywEvLZYCFglybWn5CQ_WJuHC6Mw77SbTIvkjulu6F16Ob4EWMg/exec"

    df_baseline = pd.read_csv("TRBL_dates - accepted_results.csv")

    latest_results_dir = "C:\\Users\\mikes\\GitHub\\TRBL-Breeding-Stages-Final\\outputs\\check09-17\\"
    df_results = load_automated_results_from_csv(dir=latest_results_dir)

    export_full_data_styled_excel(
        df_baseline, df_results, key_col="site", output_file="dataframe_diff.xlsx"
    )
    
    #sync_via_webhook(df_results, WEBHOOK_URL)

    #df_manual_results = load_manual_results_from_csv()
    #sync_via_webhook(df_manual_results, WEBHOOK_URL, sheet_name="manual_results")

    # df_ARI = load_ARI_score_from_csv()
    # sync_via_webhook(df_ARI, WEBHOOK_URL, sheet_name="ARI_scores")