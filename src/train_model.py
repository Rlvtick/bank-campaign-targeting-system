from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VALIDATION_PATH = PROCESSED_DIR / "validation.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"

FEATURE_LIST_PATH = REPORTS_DIR / "feature_lists.json"

VALIDATION_COMPARISON_PATH = REPORTS_DIR / "validation_model_comparison.csv"
FINAL_TEST_METRICS_PATH = REPORTS_DIR / "final_model_test_metrics.csv"
MODEL_RESULTS_SUMMARY_PATH = REPORTS_DIR / "model_results_summary.csv"
SELECTED_MODEL_SUMMARY_PATH = REPORTS_DIR / "selected_model_summary.csv"

VALIDATION_PREDICTIONS_PATH = REPORTS_DIR / "final_model_validation_predictions.csv"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "final_model_test_predictions.csv"

VALIDATION_CONFUSION_MATRIX_PATH = REPORTS_DIR / "validation_confusion_matrix.csv"
TEST_CONFUSION_MATRIX_PATH = REPORTS_DIR / "test_confusion_matrix.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"

TARGET_COLUMN = "y_binary"
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = [col for col in feature_columns if col not in df.columns]

    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def make_models(y_train: pd.Series) -> dict:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("Training data has no positive class.")

    scale_pos_weight = negative_count / positive_count

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=1,
            subsample=0.80,
            colsample_bytree=0.80,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.80,
            colsample_bytree=0.80,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def evaluate_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "threshold": threshold,
    }


def make_prediction_frame(
    split_name: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    y_pred = (y_proba >= threshold).astype(int)

    return pd.DataFrame(
        {
            "split": split_name,
            "y_true": y_true.values,
            "predicted_probability": y_proba,
            "predicted_label_0_5": y_pred,
        }
    )


def save_confusion_matrix(
    y_true: pd.Series,
    y_proba: np.ndarray,
    output_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["actual_no", "actual_yes"],
        columns=["predicted_no", "predicted_yes"],
    )

    cm_df.to_csv(output_path)


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        return preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        return [f"feature_{i}" for i in range(len(preprocessor.transformers_))]


def extract_feature_importance(
    pipeline: Pipeline,
    model_name: str,
    output_path: Path,
) -> None:
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
        importance_type = "absolute_coefficient"
    else:
        return

    importance_df = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": feature_names,
            "importance": values,
            "importance_type": importance_type,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.to_csv(output_path, index=False)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train, validation, and test splits...")
    train_df = load_split(TRAIN_PATH)
    validation_df = load_split(VALIDATION_PATH)
    test_df = load_split(TEST_PATH)

    print("Loading feature list and preprocessing pipeline...")
    feature_lists = json.loads(FEATURE_LIST_PATH.read_text(encoding="utf-8"))
    feature_columns = feature_lists["all_features"]

    base_preprocessor = joblib.load(PREPROCESSOR_PATH)

    X_train, y_train = split_features_target(train_df, feature_columns)
    X_validation, y_validation = split_features_target(validation_df, feature_columns)
    X_test, y_test = split_features_target(test_df, feature_columns)

    print("Preparing candidate models...")
    candidate_models = make_models(y_train)

    validation_results = []
    trained_pipelines = {}

    for model_name, model in candidate_models.items():
        print(f"Training model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(base_preprocessor)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        validation_proba = pipeline.predict_proba(X_validation)[:, 1]
        validation_metrics = evaluate_predictions(y_validation, validation_proba)

        validation_results.append(
            {
                "model_name": model_name,
                "split": "validation",
                **validation_metrics,
            }
        )

        trained_pipelines[model_name] = pipeline

    validation_results_df = pd.DataFrame(validation_results)
    validation_results_df = validation_results_df.sort_values(
        by="pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    selected_model_name = validation_results_df.loc[0, "model_name"]
    selected_pipeline = trained_pipelines[selected_model_name]

    print(f"Selected model based on validation PR-AUC: {selected_model_name}")

    validation_proba = selected_pipeline.predict_proba(X_validation)[:, 1]
    test_proba = selected_pipeline.predict_proba(X_test)[:, 1]

    selected_validation_metrics = evaluate_predictions(y_validation, validation_proba)
    selected_test_metrics = evaluate_predictions(y_test, test_proba)

    final_test_metrics_df = pd.DataFrame(
        [
            {
                "model_name": selected_model_name,
                "split": "test",
                **selected_test_metrics,
            }
        ]
    )

    selected_model_summary = pd.DataFrame(
        [
            {
                "selected_model": selected_model_name,
                "selection_metric": "validation_pr_auc",
                "validation_pr_auc": selected_validation_metrics["pr_auc"],
                "test_pr_auc": selected_test_metrics["pr_auc"],
                "test_roc_auc": selected_test_metrics["roc_auc"],
                "default_threshold": DEFAULT_THRESHOLD,
            }
        ]
    )

    model_results_summary = pd.concat(
        [
            validation_results_df,
            final_test_metrics_df,
        ],
        ignore_index=True,
    )

    validation_predictions = make_prediction_frame(
        "validation",
        y_validation,
        validation_proba,
    )

    test_predictions = make_prediction_frame(
        "test",
        y_test,
        test_proba,
    )

    validation_results_df.to_csv(VALIDATION_COMPARISON_PATH, index=False)
    final_test_metrics_df.to_csv(FINAL_TEST_METRICS_PATH, index=False)
    selected_model_summary.to_csv(SELECTED_MODEL_SUMMARY_PATH, index=False)
    model_results_summary.to_csv(MODEL_RESULTS_SUMMARY_PATH, index=False)

    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    save_confusion_matrix(
        y_validation,
        validation_proba,
        VALIDATION_CONFUSION_MATRIX_PATH,
    )

    save_confusion_matrix(
        y_test,
        test_proba,
        TEST_CONFUSION_MATRIX_PATH,
    )

    extract_feature_importance(
        selected_pipeline,
        selected_model_name,
        FEATURE_IMPORTANCE_PATH,
    )

    joblib.dump(selected_pipeline, FINAL_MODEL_PATH)

    print("\nModel training completed.")
    print("=" * 70)
    print("Validation model comparison:")
    print(validation_results_df.to_string(index=False))
    print("=" * 70)
    print("\nSelected model:")
    print(selected_model_summary.to_string(index=False))
    print("=" * 70)
    print("\nFinal test metrics:")
    print(final_test_metrics_df.to_string(index=False))
    print("=" * 70)
    print("\nGenerated files:")
    print(f"- {VALIDATION_COMPARISON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {FINAL_TEST_METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {MODEL_RESULTS_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {SELECTED_MODEL_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {VALIDATION_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {TEST_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {VALIDATION_CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {TEST_CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {FEATURE_IMPORTANCE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {FINAL_MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
