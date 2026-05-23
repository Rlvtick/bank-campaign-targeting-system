
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_READY_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing_model_ready.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VALIDATION_PATH = PROCESSED_DIR / "validation.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"

PREPROCESSING_SUMMARY_PATH = REPORTS_DIR / "preprocessing_summary.csv"
SPLIT_DISTRIBUTION_PATH = REPORTS_DIR / "split_distribution.csv"
FEATURE_LIST_PATH = REPORTS_DIR / "feature_lists.json"

TARGET_COLUMN = "y_binary"
ORIGINAL_TARGET_COLUMN = "y"
RANDOM_STATE = 42


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Create OneHotEncoder with compatibility for different scikit-learn versions.
    Newer versions use sparse_output, older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple, defensible feature engineering.

    pdays uses 999 as a sentinel for customers who were not previously contacted.
    Treating 999 as a normal day value would be misleading, especially for
    linear models. We create:
    - was_previously_contacted: 1 if pdays is not 999, else 0
    - pdays_clean: real pdays value if contacted before, else 0
    Then we drop the original pdays column.
    """
    df = df.copy()

    if "pdays" in df.columns:
        df["was_previously_contacted"] = (df["pdays"] != 999).astype(int)
        df["pdays_clean"] = np.where(df["pdays"] == 999, 0, df["pdays"])
        df = df.drop(columns=["pdays"])

    return df


def summarize_split(name: str, data: pd.DataFrame) -> dict:
    """Create target distribution summary for one split."""
    positive_count = int(data[TARGET_COLUMN].sum())
    total_count = int(len(data))
    positive_rate = positive_count / total_count if total_count > 0 else 0

    return {
        "split": name,
        "rows": total_count,
        "positive_count": positive_count,
        "negative_count": total_count - positive_count,
        "positive_rate": round(positive_rate, 6),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_READY_PATH.exists():
        raise FileNotFoundError(
            f"Model-ready dataset not found: {MODEL_READY_PATH}. "
            "Run src/validate_data.py first."
        )

    print("Loading model-ready dataset...")
    df = pd.read_csv(MODEL_READY_PATH)

    initial_rows = len(df)
    duplicate_rows_after_leakage_removal = int(df.duplicated().sum())

    print("Applying post-leakage duplicate handling...")
    df = df.drop_duplicates().reset_index(drop=True)
    rows_after_duplicate_removal = len(df)

    print("Applying feature engineering...")
    df = engineer_features(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    if ORIGINAL_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Original target column '{ORIGINAL_TARGET_COLUMN}' was not found.")

    # Drop the original string target from the feature set, but keep it in split files for readability.
    feature_columns = [
        col for col in df.columns
        if col not in [TARGET_COLUMN, ORIGINAL_TARGET_COLUMN]
    ]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    print("Creating stratified train, validation, and test split...")

    # 80/10/10 split:
    # 80% train, 10% validation, 10% test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values

    validation_df = X_validation.copy()
    validation_df[TARGET_COLUMN] = y_validation.values

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    train_df.to_csv(TRAIN_PATH, index=False)
    validation_df.to_csv(VALIDATION_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    print("Building ColumnTransformer preprocessing pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
        ],
        remainder="drop",
    )

    # Fit only on training data to avoid validation/test leakage.
    preprocessor.fit(X_train)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    split_distribution = pd.DataFrame(
        [
            summarize_split("train", train_df),
            summarize_split("validation", validation_df),
            summarize_split("test", test_df),
        ]
    )
    split_distribution.to_csv(SPLIT_DISTRIBUTION_PATH, index=False)

    preprocessing_summary = pd.DataFrame(
        {
            "item": [
                "input_rows_before_post_leakage_deduplication",
                "duplicate_rows_removed_after_duration_removal",
                "rows_after_post_leakage_deduplication",
                "train_rows",
                "validation_rows",
                "test_rows",
                "categorical_feature_count",
                "numerical_feature_count",
                "preprocessor_path",
                "random_state",
            ],
            "value": [
                initial_rows,
                duplicate_rows_after_leakage_removal,
                rows_after_duplicate_removal,
                len(train_df),
                len(validation_df),
                len(test_df),
                len(categorical_features),
                len(numerical_features),
                str(PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)),
                RANDOM_STATE,
            ],
        }
    )
    preprocessing_summary.to_csv(PREPROCESSING_SUMMARY_PATH, index=False)

    feature_lists = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "all_features": feature_columns,
        "target": TARGET_COLUMN,
        "dropped_from_features": [ORIGINAL_TARGET_COLUMN],
        "engineered_features": [
            "was_previously_contacted",
            "pdays_clean",
        ] if "pdays_clean" in df.columns else [],
    }

    FEATURE_LIST_PATH.write_text(json.dumps(feature_lists, indent=2), encoding="utf-8")

    print("\nPreprocessing completed.")
    print("=" * 70)
    print(f"Rows before post-leakage duplicate removal: {initial_rows:,}")
    print(f"Duplicate rows removed after duration removal: {duplicate_rows_after_leakage_removal:,}")
    print(f"Rows after duplicate removal: {rows_after_duplicate_removal:,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    print("=" * 70)
    print("\nSplit distribution:")
    print(split_distribution.to_string(index=False))
    print("\nGenerated files:")
    print(f"- {TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {VALIDATION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {TEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {PREPROCESSING_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {SPLIT_DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {FEATURE_LIST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
