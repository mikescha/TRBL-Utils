from pathlib import Path

COL_SITE_ID = "Site_ID"
COL_SITE_NAME = "Site_Name"
COL_DEPLOYMENT_START = "Deployment_Start"
COL_DEPLOYMENT_END = "Deployment_End"
COL_BREEDING_TYPE = "Breeding_Type"
COL_COMPLEX_TYPES = "Complex_Types"
COL_APPROX_COLONY_SIZE = "Approx_Colony_Size"
COL_COLONY_SIZE = "Colony_Size"
COL_SUBSTRATE = "Substrate"
COL_GROUP = "Group"
COL_PRETTY_SITE_NAME = "Pretty_Site_Name"
COL_SKIP_SITE = "Skip_Site"
COL_COMMENT = "Comment"
COL_PULSE_NAME = "Pulse_Name"
COL_OUTCOME = "Outcome"
COL_HATCH_DATE = "Hatch_Date"
COL_ABANDON_DATE = "Abandoned_Date"
COL_PARTIAL_ABANDON_DATE = "Partial_Abandon_Date"
STATUS_ND = "ND"

# Manual outcome labels
OUTCOME_ABANDONED = "Abandoned"
OUTCOME_PARTIALLY_ABANDONED = "Partially Abandoned"
OUTCOME_SUCCESSFUL = "Successful"
OUTCOME_UNKNOWN = "Unknown"
OUTCOME_NO_COLONY = "No Colony"
OUTCOME_NO_TRBL = "No TRBL"

# File locations
INPUT_CSV = Path(
    r"C:\Users\mikes\OneDrive\Documents\GitHub\TRBLSummarizer\TRBLSummarizer\Data\TRBL Analysis tracking - All.csv"
)
DATA_ROOT = Path(r"C:\Users\mikes\OneDrive\Documents\GitHub\TRBLSummarizer\TRBLSummarizer")
PMJ_DIR = DATA_ROOT / "PMJ Data"
DATA_DIR = DATA_ROOT / "Data"
HOURLY_PARQUET_FILES = DATA_DIR / Path("recordings_per_day_hour.parquet")