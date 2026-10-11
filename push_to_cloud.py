import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from common import (
    ARI_SCORE_FILE,
    DEFAULT_AUTOMATED_RESULTS_DIR,
    FEMALE_CHATTER,
    FLEDGLING_BEGGING_CALL,
    HATCHLING_BEGGING_CALL,
    MALE_CHORUS,
    NESTLING_BEGGING_CALL,
    WEBHOOK_URL,
)

TYPE_MAP = {
    FEMALE_CHATTER: "fc_block_id",
    FLEDGLING_BEGGING_CALL: "fbc_block_id",
    HATCHLING_BEGGING_CALL: "hbc_block_id",
    MALE_CHORUS: "mc_block_id",
    NESTLING_BEGGING_CALL: "nbc_block_id"
}


def load_automated_results_from_csv(dir:Path = DEFAULT_AUTOMATED_RESULTS_DIR) -> pd.DataFrame:
    """
    Load automated results from a CSV file into a pandas DataFrame.
    
    Args:
        dir (Path): Directory containing the CSV file. Defaults to DEFAULT_AUTOMATED_RESULTS_DIR.

    Returns:
        pd.DataFrame: DataFrame containing the loaded automated results.
    """
    file_path = dir / "trbl_breeding_chronology_summary.csv"
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
    df_main_cols = df_all_data[cols_to_keep].copy()

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


def load_ARI_score_from_csv() -> pd.DataFrame:
    df_ARI = pd.read_csv(ARI_SCORE_FILE)
    
    return df_ARI


def load_activity_blocks_from_csv(dir:Path = DEFAULT_AUTOMATED_RESULTS_DIR, ab_type: str = "") -> pd.DataFrame:
    """
    Load activity blocks from a CSV file into a pandas DataFrame. Filter it to only calls of the 
    requested type if specified.
    
    Args:
        dir (Path): Directory containing the CSV file. Defaults to DEFAULT_AUTOMATED_RESULTS_DIR.

    Returns:
        pd.DataFrame: DataFrame containing the loaded activity blocks, filtered by type if specified.
    """
    file_path = dir / "trbl_activity_blocks.csv"
    df_activity_blocks = pd.read_csv(file_path)

    if ab_type:
        df_activity_blocks = df_activity_blocks[df_activity_blocks["song_type"] == ab_type]

    return df_activity_blocks


def load_used_activity_blocks_from_csv(dir:Path = DEFAULT_AUTOMATED_RESULTS_DIR, ab_type: str = "") -> pd.DataFrame:
    """
    Load used activity blocks from a CSV file into a pandas DataFrame.
    
    Args:
        dir (Path): Directory containing the CSV file. Defaults to DEFAULT_AUTOMATED_RESULTS_DIR.

    Returns:
        pd.DataFrame: DataFrame containing the loaded used activity blocks.
    """
    file_path = dir / "trbl_breeding_chronology_summary.csv"
    df_chronology = pd.read_csv(file_path)

    #Filter the chronology to only include non-blank rows of the requested activity block type
    df_used_activity_blocks = df_chronology.dropna(subset=[TYPE_MAP[ab_type]]) if ab_type else df_chronology.copy()

    return df_used_activity_blocks



def clip_activity_blocks_by_recording_date(
        df_activity_blocks: pd.DataFrame, 
        start_col: str = "start_date", 
        end_col: str = "end_date"
    ) -> pd.DataFrame:
    """
    Clip activity blocks to fit within the recording start and end dates.

    Parameters:
        df_activity_blocks (pd.DataFrame): DataFrame containing activity blocks with start and end date columns.
        start_col (str): Name of the column representing the start date of the activity block. Defaults to "start_date".
        end_col (str): Name of the column representing the end date of the activity block. Defaults to "end_date".

    Returns:
        pd.DataFrame: DataFrame with clipped activity blocks.
    """
    df_abs = df_activity_blocks.copy()

    #Get recording dates
    SITE_METADATA = Path(".\\output\\trbl_site_metadata.csv")
    df_site_metadata = pd.read_csv(SITE_METADATA)

    # site_metadata looks like this:
    #   id,site,evidence_site,first_recording,last_recording,distance_to_colony,approx_colony_size,altitude,latitude,longitude,substrate  # noqa: E501
    #   2059,2017 Rush Ranch,2017 Rush Ranch,2017-03-02,2017-07-24,3,800,6,38.208,-122.014,"Bulrush, cattails"

    # df_activity_blocks has a column "site" which corresponds to the "site" column in df_site_metadata
    # for each row in df_activity_blocks that has a matching site in df_site_metadata, retrieve the 
    # first_recording and last_recording dates, then do the truncation.
    df_merged = df_abs.merge(
        df_site_metadata[['evidence_site', 'first_recording', 'last_recording']],
        left_on="site_name", right_on="evidence_site",
        how='left',
        suffixes=('', '_site_metadata')
    )

    df_merged['first_recording'] = pd.to_datetime(df_merged['first_recording'])
    df_merged['last_recording'] = pd.to_datetime(df_merged['last_recording'])

    df_merged[start_col] = pd.to_datetime(df_merged[start_col])
    df_merged[end_col] = pd.to_datetime(df_merged[end_col])

    df_truncated = df_merged.copy()
    one_day = pd.Timedelta(days=1)
    # Start date can't be earlier than first recording date, instead this should 
    # check for start_col on or 1 day after the first recording date
    df_truncated.loc[
        df_truncated[start_col] <= df_truncated['first_recording'] + one_day, start_col
    ] = pd.NaT

    # End date can't be later than the last recording date, instead this should 
    # check for end_col on or 1 day before the last recording date
    df_truncated.loc[
        df_truncated[end_col] >= df_truncated['last_recording'] - one_day, end_col
    ] = pd.NaT

    df_final = df_truncated[["site", start_col, end_col]]

    return df_final


def get_activity_block_dates(
        df_activity_blocks: pd.DataFrame, 
        df_used_activity_blocks: pd.DataFrame, 
        ab_type: str = ""
    ) -> pd.DataFrame:
    """
    df_activity_blocks contains all the activity blocks. example:
    site	song_type	block_id	start_date	end_date	confident_present_days	bridge_days	total_detection_hours	low_effort_bridge_days
    2017 Rush Ranch	fbc	2017_Rush_Ranch__fbc__E01	2017-06-12	2017-06-21	10	2	76	0
    2017 Rush Ranch	fc	2017_Rush_Ranch__female_chatter__E06	2017-05-14	2017-06-02	17	1	157	0

    df_used_activity_blocks contains only the used activity blocks.
    site	pulse	settlement_start	settlement_end	incubation_onset	brooding_onset	fledging_onset	expected_fledging_onset	observed_fledging_onset	fledgling_dispersal	needs_review	mc_block_id	fc_block_id	hbc_block_id	fbc_block_id
    2017 Rush Ranch	P1	2017-04-10	2017-04-12	2017-04-19(C)	2017-05-01	no_data	2017-05-14	no_data	no_data		2017_Rush_Ranch__male_chorus__E03		2017_Rush_Ranch__hbc__E01	
    2017 Rush Ranch	P2	2017-05-08	2017-05-15	2017-05-18(C)	2017-05-30	2017-06-12	2017-06-12	2017-06-12	2017-06-22(C)		2017_Rush_Ranch__male_chorus__E04	2017_Rush_Ranch__female_chatter__E06	2017_Rush_Ranch__hbc__E02	2017_Rush_Ranch__fbc__E01

    The function returns a DataFrame with 3 columns: site, activity block start, activity block end, 
    with one row for each site+pulse combination, if there was a used activity block of the matching type.
    site should be "{site} {pulse}", e.g. "2017 Rush Ranch P2"
    Example:
        site    fc_block_id_start    fc_block_id_end
        2017 Rush Ranch P2  2017-05-14	2017-06-02
    
    Finally, adjust the dates so that if a start date is prior to recording start, it is replaced with "ND", 
    and if an end date is after the recording end, it is replaced with "ND".
    """  # noqa: E501
    ab_column = TYPE_MAP[ab_type] if ab_type else None
    
    df_used_abs = df_used_activity_blocks.copy()
    if ab_column:
        df_used_abs = df_used_abs.dropna(subset=[ab_column])

    df_merged = df_used_abs.merge(
        df_activity_blocks,
        left_on=ab_column,
        right_on="block_id",
        how="left",
        suffixes=("", "_activity_blocks"),
    )

    # Add a new column that's the merge of the site and pulse columns to create a unique site identifier
    df_merged = df_merged.assign(
        site=df_merged["site"] + " " + df_merged["pulse"]
    )

    # Keep only the relevant columns and rename them for clarity
    df_merged = df_merged[["site", "site_activity_blocks", "start_date", "end_date"]]
    start_col = f"{ab_type}_start" if ab_type else "activity_block_start"
    end_col = f"{ab_type}_end" if ab_type else "activity_block_end"
    df_merged.columns = [
        "site",
        "site_name",
        start_col,
        end_col
    ]

    df_result_clipped = clip_activity_blocks_by_recording_date(
        df_merged, 
        start_col=start_col, 
        end_col=end_col
    )
    return df_result_clipped



def push_via_webhook(df: pd.DataFrame, sheet_name: str = ""):
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
    response = requests.post(WEBHOOK_URL, json=payload)
    response.raise_for_status()
    
    result = response.json()
    if result.get("status") == "success":
        print(f"✅ Updated '{sheet_name}' with {result.get('rows_written')} rows via Webhook.")
    else:
        print(f"❌ Apps Script Error: {result.get('message')}")


if __name__ == "__main__":
    latest_results_dir = Path(
        "C:\\Users\\mikes\\GitHub\\TRBL-Breeding-Stages-Final\\docs\\reviews\\20261001\\publication",
    )
    # df_results = load_automated_results_from_csv(dir=latest_results_dir)

    # # with no sheet name parameter, the script will automatically use today's date as the sheet name
    # push_via_webhook(df_results)

    # df_ARI = load_ARI_score_from_csv()
    # push_via_webhook(df_ARI, sheet_name="ARI_scores")

    df_activity_blocks = load_activity_blocks_from_csv(dir=latest_results_dir, ab_type=FEMALE_CHATTER)
    df_used_activity_blocks = load_used_activity_blocks_from_csv(ab_type=FEMALE_CHATTER)
    df_filtered_activity_blocks = get_activity_block_dates(
        df_activity_blocks, 
        df_used_activity_blocks, 
        ab_type=FEMALE_CHATTER)

    date_cols = [
        f"{FEMALE_CHATTER}_start",
        f"{FEMALE_CHATTER}_end",
    ]
    df_for_webhook = df_filtered_activity_blocks.copy()
    for col in date_cols:
        df_for_webhook[col] = (
            df_for_webhook[col].dt.strftime("%Y-%m-%d").fillna("ND")
        )
    push_via_webhook(df_for_webhook, sheet_name="fc_activity_blocks")