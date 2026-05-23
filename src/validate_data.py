"""
Data validation and leakage check script for the Bank Campaign Targeting System.

This script checks the raw UCI Bank Marketing dataset and creates a deployable
modeling dataset where exact duplicate rows are removed and the leakage-prone
duration column is excluded.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing_uci_bank_additional_full_raw.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_READY_PATH = PROCESSED_DIR / "bank_marketing_model_ready.csv"
VALIDATION_SUMMARY_PATH = REPORTS_DIR / "data_validation_summary.csv"
COLUMN_PROFILE_PATH = REPORTS_DIR / "column_profile.csv"
TARGET_DISTRIBUTION_PATH = REPORTS_DIR / "target_distribution.csv"
LEAKAGE_REPORT_PATH = REPORTS_DIR / "leakage_check_report.md"

EXPECTED_ROWS = 41188
EXPECTED_COLUMNS = 21
TARGET_COLUMN = "y"
LEAKAGE_COLUMNS = ["duration"]


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Create a simple column-level profile for validation."""
    profile_rows = []

    for column in df.columns:
        profile_rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_rate": round(float(df[column].isna().mean()), 6),
                "unique_count": int(df[column].nunique(dropna=False)),
                "sample_values": ", ".join(
                    map(str, df[column].dropna().astype(str).unique()[:5])
                ),
            }
        )

    return pd.DataFrame(profile_rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file was not found: {RAW_DATA_PATH}. "
            "Run src/data_load.py first."
        )

    print("Loading raw dataset...")
    raw_df = pd.read_csv(RAW_DATA_PATH)

    print("Running validation checks...")

    raw_rows = raw_df.shape[0]
    raw_columns = raw_df.shape[1]
    duplicate_rows = int(raw_df.duplicated().sum())
    total_missing = int(raw_df.isna().sum().sum())

    target_exists = TARGET_COLUMN in raw_df.columns
    duration_exists = "duration" in raw_df.columns

    if not target_exists:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    target_distribution = (
        raw_df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
    )
    target_distribution["percentage"] = (
        target_distribution["count"] / len(raw_df) * 100
    ).round(4)

    valid_target_values = set(raw_df[TARGET_COLUMN].dropna().unique())
    expected_target_values = {"yes", "no"}
    unexpected_target_values = sorted(valid_target_values - expected_target_values)

    model_df = raw_df.copy()

    # Remove exact duplicate rows before modeling to reduce avoidable leakage risk.
    model_df = model_df.drop_duplicates().reset_index(drop=True)

    for leakage_col in LEAKAGE_COLUMNS:
        if leakage_col in model_df.columns:
            model_df = model_df.drop(columns=[leakage_col])

    model_df["y_binary"] = model_df[TARGET_COLUMN].map({"yes": 1, "no": 0})

    if model_df["y_binary"].isna().any():
        raise ValueError("Target conversion produced missing values. Check target labels.")

    model_df.to_csv(MODEL_READY_PATH, index=False)

    column_profile = build_column_profile(raw_df)
    column_profile.to_csv(COLUMN_PROFILE_PATH, index=False)

    target_distribution.to_csv(TARGET_DISTRIBUTION_PATH, index=False)

    validation_summary = pd.DataFrame(
        {
            "item": [
                "raw_rows",
                "expected_rows",
                "raw_columns",
                "expected_columns",
                "duplicate_rows_removed",
                "total_missing_values",
                "target_column_exists",
                "target_column",
                "unexpected_target_values",
                "duration_column_exists",
                "duration_removed_from_model_ready",
                "model_ready_rows",
                "model_ready_columns",
                "model_ready_path",
            ],
            "value": [
                raw_rows,
                EXPECTED_ROWS,
                raw_columns,
                EXPECTED_COLUMNS,
                duplicate_rows,
                total_missing,
                target_exists,
                TARGET_COLUMN,
                unexpected_target_values,
                duration_exists,
                "duration" not in model_df.columns,
                model_df.shape[0],
                model_df.shape[1],
                str(MODEL_READY_PATH.relative_to(PROJECT_ROOT)),
            ],
        }
    )

    validation_summary.to_csv(VALIDATION_SUMMARY_PATH, index=False)

    leakage_report = f"""# Leakage Check Report

## Main leakage decision

The `duration` column is excluded from the deployable modeling dataset.

## Reason

`duration` represents the last contact duration in seconds. In a real pre-call campaign targeting workflow, this value would only be known after the customer has already been contacted.

Because this project is designed to support customer prioritization before contact, using `duration` would create target leakage.

## Duplicate handling

Exact duplicate rows are removed from the model-ready dataset before modeling.

- Raw rows: {raw_rows}
- Exact duplicate rows removed: {duplicate_rows}
- Model-ready rows: {model_df.shape[0]}

## Implementation

- Raw dataset columns: {raw_columns}
- Model-ready columns: {model_df.shape[1]}
- `duration` exists in raw data: {duration_exists}
- `duration` removed from model-ready data: {"duration" not in model_df.columns}

## Output

The model-ready dataset is saved locally at:

`{MODEL_READY_PATH.relative_to(PROJECT_ROOT)}`

This processed CSV is not committed to GitHub because it is generated from the raw public dataset.
"""

    LEAKAGE_REPORT_PATH.write_text(leakage_report, encoding="utf-8")

    print("\nData validation completed.")
    print("=" * 70)
    print(f"Raw rows: {raw_rows:,}")
    print(f"Raw columns: {raw_columns:,}")
    print(f"Duplicate rows removed: {duplicate_rows:,}")
    print(f"Total missing values: {total_missing:,}")
    print(f"Target column exists: {target_exists}")
    print(f"Unexpected target values: {unexpected_target_values}")
    print(f"Duration column exists in raw data: {duration_exists}")
    print(f"Duration removed from model-ready data: {'duration' not in model_df.columns}")
    print(f"Model-ready rows: {model_df.shape[0]:,}")
    print(f"Model-ready columns: {model_df.shape[1]:,}")
    print("=" * 70)
    print("\nTarget distribution:")
    print(target_distribution.to_string(index=False))
    print("\nGenerated files:")
    print(f"- {MODEL_READY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {VALIDATION_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {COLUMN_PROFILE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {TARGET_DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {LEAKAGE_REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
