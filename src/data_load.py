"""
Data loading script for the Bank Campaign Targeting System.

This script downloads the UCI Bank Marketing ZIP file, extracts the preferred
bank-additional-full.csv file, saves a raw local copy, and creates metadata
files for documentation and reproducibility.
"""

from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

DOWNLOAD_ZIP_PATH = RAW_DIR / "bank_marketing_uci_original.zip"
EXTRACT_DIR = RAW_DIR / "uci_bank_marketing_extracted"

PREFERRED_RAW_FILE_NAME = "bank-additional-full.csv"

RAW_DATA_PATH = RAW_DIR / "bank_marketing_uci_bank_additional_full_raw.csv"
ATTRIBUTION_PATH = RAW_DIR / "dataset_attribution.md"
DATA_ACCESS_SUMMARY_PATH = REPORTS_DIR / "data_access_summary.csv"


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP file into a target folder."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI Bank Marketing dataset ZIP...")
    urlretrieve(UCI_ZIP_URL, DOWNLOAD_ZIP_PATH)

    print("Extracting main ZIP file...")
    extract_zip(DOWNLOAD_ZIP_PATH, EXTRACT_DIR)

    nested_zip_files = list(EXTRACT_DIR.rglob("*.zip"))

    for nested_zip in nested_zip_files:
        print(f"Extracting nested ZIP file: {nested_zip.name}")
        extract_zip(nested_zip, nested_zip.parent / nested_zip.stem)

    preferred_files = list(EXTRACT_DIR.rglob(PREFERRED_RAW_FILE_NAME))

    if not preferred_files:
        raise FileNotFoundError(
            f"Could not find {PREFERRED_RAW_FILE_NAME} after extracting the UCI ZIP file."
        )

    preferred_file_path = preferred_files[0]

    print(f"Loading preferred raw file: {preferred_file_path.name}")

    raw_df = pd.read_csv(preferred_file_path, sep=";")

    raw_df.to_csv(RAW_DATA_PATH, index=False)

    target_col = "y"

    if target_col not in raw_df.columns:
        raise ValueError(f"Expected target column '{target_col}' was not found.")

    target_counts = raw_df[target_col].value_counts(dropna=False).reset_index()
    target_counts.columns = [target_col, "count"]

    summary_df = pd.DataFrame(
        {
            "item": [
                "dataset_name",
                "source",
                "uci_dataset_id",
                "preferred_source_file",
                "raw_rows",
                "raw_columns",
                "target_column",
                "raw_data_path",
            ],
            "value": [
                "Bank Marketing",
                "UCI Machine Learning Repository",
                "222",
                PREFERRED_RAW_FILE_NAME,
                raw_df.shape[0],
                raw_df.shape[1],
                target_col,
                str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)),
            ],
        }
    )

    summary_df.to_csv(DATA_ACCESS_SUMMARY_PATH, index=False)

    attribution_text = """# Dataset Attribution

Dataset: Bank Marketing

Source: UCI Machine Learning Repository

UCI Dataset ID: 222

Preferred file used in this project: bank-additional-full.csv

Dataset URL: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

Citation:
Moro, S., Rita, P., & Cortez, P. (2012). Bank Marketing [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Related paper:
Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict
the success of bank telemarketing. Decision Support Systems, 62, 22-31.
https://doi.org/10.1016/j.dss.2014.03.001

Project note:
This dataset is used as a public proxy for a bank campaign targeting problem.
It is not Rakuten Bank data and should not be interpreted as real Rakuten
customer behavior.
"""

    ATTRIBUTION_PATH.write_text(attribution_text, encoding="utf-8")

    print("\nData access completed.")
    print("=" * 70)
    print(f"Preferred file used: {PREFERRED_RAW_FILE_NAME}")
    print(f"Raw dataset saved to: {RAW_DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Attribution saved to: {ATTRIBUTION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary saved to: {DATA_ACCESS_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 70)
    print(f"Rows: {raw_df.shape[0]:,}")
    print(f"Columns: {raw_df.shape[1]:,}")
    print(f"Target column: {target_col}")
    print("\nTarget distribution:")
    print(target_counts.to_string(index=False))


if __name__ == "__main__":
    main()
