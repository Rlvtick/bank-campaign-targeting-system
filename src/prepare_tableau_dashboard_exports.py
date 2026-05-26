from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
REPORTS_DIR = PROJECT_ROOT / "reports"

CUSTOMER_SCORES_PATH = DASHBOARD_DIR / "customer_scores.csv"
THRESHOLD_VALIDATION_PATH = REPORTS_DIR / "threshold_simulation_validation.csv"
THRESHOLD_TEST_PATH = REPORTS_DIR / "threshold_simulation_test.csv"
SELECTED_POLICY_SOURCE_PATH = REPORTS_DIR / "selected_threshold_policy.csv"

DASHBOARD_CUSTOMER_SCORES_PATH = DASHBOARD_DIR / "customer_scores.csv"
DASHBOARD_THRESHOLD_SIMULATION_PATH = DASHBOARD_DIR / "threshold_simulation.csv"
DASHBOARD_SELECTED_POLICY_PATH = DASHBOARD_DIR / "selected_policy.csv"


def read_required_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def format_lift(value: float) -> str:
    return f"{value:.2f}x"


def percent_to_x10000(value: float) -> int:
    return int(round(value * 10000))


def lift_to_x100(value: float) -> int:
    return int(round(value * 100))


def prepare_customer_scores(customer_scores: pd.DataFrame) -> pd.DataFrame:
    output = customer_scores.copy()

    if "actual_subscription_customer" in output.columns:
        output = output.drop(columns=["actual_subscription_customer"])

    if "selected_policy_flag" in output.columns:
        output["recommendation_group"] = output["selected_policy_flag"].map(
            {
                1: "Recommended: Top 20%",
                0: "Not Recommended",
            }
        )

    if "actual_subscription" in output.columns:
        output["subscription_label"] = output["actual_subscription"].map(
            {
                1: "Subscribed",
                0: "Did Not Subscribe",
            }
        )

    if "predicted_probability" in output.columns:
        output["predicted_probability_x10000"] = (
            output["predicted_probability"] * 10000
        ).round().astype(int)

        output["predicted_probability_display"] = output["predicted_probability"].map(
            format_percent
        )

        output["score_band_tableau"] = pd.cut(
            output["predicted_probability"],
            bins=[0, 0.20, 0.40, 0.60, 0.80, 1.00],
            labels=[
                "0% to 19%",
                "20% to 39%",
                "40% to 59%",
                "60% to 79%",
                "80% to 100%",
            ],
            include_lowest=True,
        ).astype(str)

    if "selected_contact_rate" in output.columns:
        output = output.drop(columns=["selected_contact_rate"])

    preferred_columns = [
        "customer_id",
        "split",
        "actual_subscription",
        "subscription_label",
        "predicted_probability",
        "predicted_probability_x10000",
        "predicted_probability_display",
        "score_rank",
        "score_percentile",
        "selected_policy_flag",
        "recommendation_group",
        "score_band",
        "score_band_tableau",
        "age",
        "job",
        "marital",
        "education",
        "contact",
        "month",
        "campaign",
        "previous",
        "poutcome",
        "recommended_top_5",
        "recommended_top_10",
        "recommended_top_15",
        "recommended_top_20",
        "recommended_top_25",
        "recommended_top_30",
    ]

    existing_preferred = [
        col for col in preferred_columns
        if col in output.columns
    ]

    remaining_columns = [
        col for col in output.columns
        if col not in existing_preferred
    ]

    return output[existing_preferred + remaining_columns]


def prepare_threshold_simulation(
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    source = pd.concat([validation, test], ignore_index=True)

    rows = []

    for _, row in source.iterrows():
        contact_rate_order = int(round(row["contact_rate"] * 100))
        contact_rate_label = f"Top {contact_rate_order}%"

        base = {
            "split": row["split"],
            "contact_rate_order": contact_rate_order,
            "contact_rate_label": contact_rate_label,
            "targeted_customers": int(row["targeted_customers"]),
            "captured_subscribers": int(row["captured_subscribers"]),
            "total_subscribers": int(row["total_subscribers"]),
            "false_positives": int(row["false_positives"]),
            "missed_subscribers": int(row["missed_subscribers"]),
            "score_cutoff_x10000": percent_to_x10000(row["score_cutoff"]),
            "score_cutoff_display": format_percent(row["score_cutoff"]),
        }

        rows.append(
            {
                **base,
                "metric": "Precision@k",
                "metric_type": "Percent",
                "metric_value_x10000": percent_to_x10000(row["precision_at_k"]),
                "metric_display": format_percent(row["precision_at_k"]),
            }
        )

        rows.append(
            {
                **base,
                "metric": "Capture Rate@k",
                "metric_type": "Percent",
                "metric_value_x10000": percent_to_x10000(row["capture_rate_at_k"]),
                "metric_display": format_percent(row["capture_rate_at_k"]),
            }
        )

        rows.append(
            {
                **base,
                "metric": "Lift@k",
                "metric_type": "Lift",
                "metric_value_x10000": lift_to_x100(row["lift_at_k"]),
                "metric_display": format_lift(row["lift_at_k"]),
            }
        )

    output = pd.DataFrame(rows)

    return output


def prepare_selected_policy(selected_policy: pd.DataFrame) -> pd.DataFrame:
    row = selected_policy.iloc[0]

    output = pd.DataFrame(
        [
            {
                "selected_policy_label": f"Top {int(row['selected_contact_rate'] * 100)}%",
                "selection_rule": row["selection_rule"],
                "recommended_customers": int(row["test_targeted_customers"]),
                "captured_subscribers": int(row["test_captured_subscribers"]),
                "precision_x10000": percent_to_x10000(row["test_precision_at_k"]),
                "precision_display": format_percent(row["test_precision_at_k"]),
                "capture_rate_x10000": percent_to_x10000(row["test_capture_rate_at_k"]),
                "capture_rate_display": format_percent(row["test_capture_rate_at_k"]),
                "lift_x100": lift_to_x100(row["test_lift_at_k"]),
                "lift_display": format_lift(row["test_lift_at_k"]),
                "score_cutoff_x10000": percent_to_x10000(row["test_score_cutoff"]),
                "score_cutoff_display": format_percent(row["test_score_cutoff"]),
                "validation_recommended_customers": int(row["validation_targeted_customers"]),
                "validation_captured_subscribers": int(row["validation_captured_subscribers"]),
                "validation_precision_display": format_percent(row["validation_precision_at_k"]),
                "validation_capture_rate_display": format_percent(row["validation_capture_rate_at_k"]),
                "validation_lift_display": format_lift(row["validation_lift_at_k"]),
            }
        ]
    )

    return output


def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading source files...")
    customer_scores = read_required_csv(CUSTOMER_SCORES_PATH, "Dashboard customer scores")
    threshold_validation = read_required_csv(
        THRESHOLD_VALIDATION_PATH,
        "Validation threshold simulation",
    )
    threshold_test = read_required_csv(
        THRESHOLD_TEST_PATH,
        "Test threshold simulation",
    )
    selected_policy = read_required_csv(
        SELECTED_POLICY_SOURCE_PATH,
        "Selected policy source",
    )

    print("Preparing Tableau-friendly dashboard files...")
    customer_scores_output = prepare_customer_scores(customer_scores)
    threshold_output = prepare_threshold_simulation(
        threshold_validation,
        threshold_test,
    )
    selected_policy_output = prepare_selected_policy(selected_policy)

    print("Replacing dashboard CSV files...")
    customer_scores_output.to_csv(DASHBOARD_CUSTOMER_SCORES_PATH, index=False)
    threshold_output.to_csv(DASHBOARD_THRESHOLD_SIMULATION_PATH, index=False)
    selected_policy_output.to_csv(DASHBOARD_SELECTED_POLICY_PATH, index=False)

    print("\nTableau dashboard exports completed.")
    print("=" * 80)
    print(f"{DASHBOARD_CUSTOMER_SCORES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(customer_scores_output):,}")
    print(f"Columns: {len(customer_scores_output.columns):,}")
    print(customer_scores_output.head(3).to_string(index=False))
    print("=" * 80)
    print(f"{DASHBOARD_THRESHOLD_SIMULATION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(threshold_output):,}")
    print(f"Columns: {len(threshold_output.columns):,}")
    print(threshold_output.head(9).to_string(index=False))
    print("=" * 80)
    print(f"{DASHBOARD_SELECTED_POLICY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(selected_policy_output):,}")
    print(f"Columns: {len(selected_policy_output.columns):,}")
    print(selected_policy_output.to_string(index=False))


if __name__ == "__main__":
    main()
