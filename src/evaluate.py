from pathlib import Path
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"

VALIDATION_DATA_PATH = PROCESSED_DIR / "validation.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv"

VALIDATION_PREDICTIONS_PATH = REPORTS_DIR / "final_model_validation_predictions.csv"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "final_model_test_predictions.csv"

VALIDATION_TOPK_PATH = REPORTS_DIR / "threshold_simulation_validation.csv"
TEST_TOPK_PATH = REPORTS_DIR / "threshold_simulation_test.csv"

VALIDATION_FIXED_THRESHOLD_PATH = REPORTS_DIR / "fixed_threshold_simulation_validation.csv"
TEST_FIXED_THRESHOLD_PATH = REPORTS_DIR / "fixed_threshold_simulation_test.csv"

SELECTED_POLICY_PATH = REPORTS_DIR / "selected_threshold_policy.csv"
CUSTOMER_SCORES_PATH = PREDICTIONS_DIR / "customer_scores.csv"

TOP_K_CONTACT_RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
FIXED_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

MIN_CAPTURE_RATE_FOR_SELECTION = 0.60


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file was not found: {path}")

    predictions = pd.read_csv(path)

    required_columns = {"split", "y_true", "predicted_probability"}

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(f"Missing required prediction columns: {missing_columns}")

    return predictions


def simulate_top_k(
    predictions: pd.DataFrame,
    contact_rates: list[float],
    split_name: str,
) -> pd.DataFrame:
    data = predictions.copy()
    data = data.sort_values("predicted_probability", ascending=False).reset_index(drop=True)

    total_customers = len(data)
    total_subscribers = int(data["y_true"].sum())
    baseline_rate = total_subscribers / total_customers if total_customers > 0 else 0

    rows = []

    for contact_rate in contact_rates:
        targeted_customers = max(1, math.ceil(total_customers * contact_rate))
        targeted = data.head(targeted_customers)

        captured_subscribers = int(targeted["y_true"].sum())
        false_positives = targeted_customers - captured_subscribers
        missed_subscribers = total_subscribers - captured_subscribers

        precision_at_k = captured_subscribers / targeted_customers if targeted_customers > 0 else 0
        capture_rate_at_k = captured_subscribers / total_subscribers if total_subscribers > 0 else 0
        lift_at_k = precision_at_k / baseline_rate if baseline_rate > 0 else 0

        score_cutoff = float(targeted["predicted_probability"].min())

        rows.append(
            {
                "split": split_name,
                "method": "top_k",
                "contact_rate": contact_rate,
                "targeted_customers": targeted_customers,
                "total_customers": total_customers,
                "captured_subscribers": captured_subscribers,
                "total_subscribers": total_subscribers,
                "false_positives": false_positives,
                "missed_subscribers": missed_subscribers,
                "precision_at_k": precision_at_k,
                "capture_rate_at_k": capture_rate_at_k,
                "lift_at_k": lift_at_k,
                "baseline_subscription_rate": baseline_rate,
                "score_cutoff": score_cutoff,
            }
        )

    return pd.DataFrame(rows)


def simulate_fixed_thresholds(
    predictions: pd.DataFrame,
    thresholds: list[float],
    split_name: str,
) -> pd.DataFrame:
    data = predictions.copy()

    total_customers = len(data)
    total_subscribers = int(data["y_true"].sum())
    baseline_rate = total_subscribers / total_customers if total_customers > 0 else 0

    rows = []

    for threshold in thresholds:
        targeted = data[data["predicted_probability"] >= threshold].copy()

        targeted_customers = len(targeted)
        captured_subscribers = int(targeted["y_true"].sum()) if targeted_customers > 0 else 0
        false_positives = targeted_customers - captured_subscribers
        missed_subscribers = total_subscribers - captured_subscribers

        contact_rate = targeted_customers / total_customers if total_customers > 0 else 0
        precision = captured_subscribers / targeted_customers if targeted_customers > 0 else 0
        capture_rate = captured_subscribers / total_subscribers if total_subscribers > 0 else 0
        lift = precision / baseline_rate if baseline_rate > 0 else 0

        rows.append(
            {
                "split": split_name,
                "method": "fixed_threshold",
                "threshold": threshold,
                "contact_rate": contact_rate,
                "targeted_customers": targeted_customers,
                "total_customers": total_customers,
                "captured_subscribers": captured_subscribers,
                "total_subscribers": total_subscribers,
                "false_positives": false_positives,
                "missed_subscribers": missed_subscribers,
                "precision": precision,
                "capture_rate": capture_rate,
                "lift": lift,
                "baseline_subscription_rate": baseline_rate,
            }
        )

    return pd.DataFrame(rows)


def select_policy(validation_topk: pd.DataFrame) -> pd.Series:
    candidates = validation_topk[
        validation_topk["capture_rate_at_k"] >= MIN_CAPTURE_RATE_FOR_SELECTION
    ].copy()

    if not candidates.empty:
        selected = candidates.sort_values(
            ["contact_rate", "lift_at_k"],
            ascending=[True, False],
        ).iloc[0].copy()
        selected["selection_rule"] = "smallest_contact_rate_with_capture_rate_at_least_0_60"
        return selected

    validation_topk = validation_topk.copy()
    validation_topk["business_score"] = (
        2
        * validation_topk["precision_at_k"]
        * validation_topk["capture_rate_at_k"]
        / (
            validation_topk["precision_at_k"]
            + validation_topk["capture_rate_at_k"]
        )
    )

    selected = validation_topk.sort_values("business_score", ascending=False).iloc[0].copy()
    selected["selection_rule"] = "highest_precision_capture_harmonic_score"
    return selected


def build_selected_policy_report(
    selected_validation_policy: pd.Series,
    test_topk: pd.DataFrame,
) -> pd.DataFrame:
    selected_contact_rate = float(selected_validation_policy["contact_rate"])

    selected_test_policy = test_topk[
        np.isclose(test_topk["contact_rate"], selected_contact_rate)
    ].iloc[0]

    report = pd.DataFrame(
        [
            {
                "selected_contact_rate": selected_contact_rate,
                "selection_rule": selected_validation_policy["selection_rule"],
                "validation_targeted_customers": int(selected_validation_policy["targeted_customers"]),
                "validation_captured_subscribers": int(selected_validation_policy["captured_subscribers"]),
                "validation_precision_at_k": selected_validation_policy["precision_at_k"],
                "validation_capture_rate_at_k": selected_validation_policy["capture_rate_at_k"],
                "validation_lift_at_k": selected_validation_policy["lift_at_k"],
                "validation_score_cutoff": selected_validation_policy["score_cutoff"],
                "test_targeted_customers": int(selected_test_policy["targeted_customers"]),
                "test_captured_subscribers": int(selected_test_policy["captured_subscribers"]),
                "test_precision_at_k": selected_test_policy["precision_at_k"],
                "test_capture_rate_at_k": selected_test_policy["capture_rate_at_k"],
                "test_lift_at_k": selected_test_policy["lift_at_k"],
                "test_score_cutoff": selected_test_policy["score_cutoff"],
            }
        ]
    )

    return report


def build_customer_scores(
    test_data: pd.DataFrame,
    test_predictions: pd.DataFrame,
    selected_contact_rate: float,
) -> pd.DataFrame:
    if len(test_data) != len(test_predictions):
        raise ValueError("Test data and test prediction row counts do not match.")

    scores = test_data.copy()
    scores["customer_id"] = [f"test_{i + 1:06d}" for i in range(len(scores))]
    scores["split"] = "test"
    scores["actual_subscription"] = test_predictions["y_true"].values
    scores["predicted_probability"] = test_predictions["predicted_probability"].values
    scores["predicted_label_0_5"] = test_predictions["predicted_label_0_5"].values

    scores = scores.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    scores["score_rank"] = np.arange(1, len(scores) + 1)
    scores["score_percentile"] = 1 - ((scores["score_rank"] - 1) / len(scores))

    for contact_rate in TOP_K_CONTACT_RATES:
        flag_name = f"recommended_top_{int(contact_rate * 100)}"
        cutoff_count = max(1, math.ceil(len(scores) * contact_rate))
        scores[flag_name] = scores["score_rank"] <= cutoff_count

    selected_flag_name = f"recommended_selected_top_{int(selected_contact_rate * 100)}"
    selected_count = max(1, math.ceil(len(scores) * selected_contact_rate))
    scores[selected_flag_name] = scores["score_rank"] <= selected_count

    first_cols = [
        "customer_id",
        "split",
        "actual_subscription",
        "predicted_probability",
        "predicted_label_0_5",
        "score_rank",
        "score_percentile",
        selected_flag_name,
    ]

    remaining_cols = [col for col in scores.columns if col not in first_cols]

    return scores[first_cols + remaining_cols]


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prediction files...")
    validation_predictions = load_predictions(VALIDATION_PREDICTIONS_PATH)
    test_predictions = load_predictions(TEST_PREDICTIONS_PATH)

    print("Running top-k targeting simulation...")
    validation_topk = simulate_top_k(
        validation_predictions,
        TOP_K_CONTACT_RATES,
        "validation",
    )
    test_topk = simulate_top_k(
        test_predictions,
        TOP_K_CONTACT_RATES,
        "test",
    )

    print("Running fixed threshold simulation...")
    validation_fixed_threshold = simulate_fixed_thresholds(
        validation_predictions,
        FIXED_THRESHOLDS,
        "validation",
    )
    test_fixed_threshold = simulate_fixed_thresholds(
        test_predictions,
        FIXED_THRESHOLDS,
        "test",
    )

    print("Selecting policy from validation results...")
    selected_validation_policy = select_policy(validation_topk)
    selected_policy_report = build_selected_policy_report(
        selected_validation_policy,
        test_topk,
    )

    print("Creating customer score table...")
    test_data = pd.read_csv(TEST_DATA_PATH)
    selected_contact_rate = float(selected_policy_report.loc[0, "selected_contact_rate"])

    customer_scores = build_customer_scores(
        test_data,
        test_predictions,
        selected_contact_rate,
    )

    validation_topk.to_csv(VALIDATION_TOPK_PATH, index=False)
    test_topk.to_csv(TEST_TOPK_PATH, index=False)
    validation_fixed_threshold.to_csv(VALIDATION_FIXED_THRESHOLD_PATH, index=False)
    test_fixed_threshold.to_csv(TEST_FIXED_THRESHOLD_PATH, index=False)
    selected_policy_report.to_csv(SELECTED_POLICY_PATH, index=False)
    customer_scores.to_csv(CUSTOMER_SCORES_PATH, index=False)

    print("\nEvaluation and threshold simulation completed.")
    print("=" * 70)
    print("Validation top-k simulation:")
    print(validation_topk.to_string(index=False))
    print("=" * 70)
    print("\nSelected policy:")
    print(selected_policy_report.to_string(index=False))
    print("=" * 70)
    print("\nTest top-k simulation:")
    print(test_topk.to_string(index=False))
    print("=" * 70)
    print("\nGenerated files:")
    print(f"- {VALIDATION_TOPK_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {TEST_TOPK_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {VALIDATION_FIXED_THRESHOLD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {TEST_FIXED_THRESHOLD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {SELECTED_POLICY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {CUSTOMER_SCORES_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
