from pathlib import Path
import math
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
SELECTED_POLICY_PATH = PROJECT_ROOT / "reports" / "selected_threshold_policy.csv"

TARGET_COLUMNS = {"y", "y_binary"}
LEAKAGE_COLUMNS = {"duration"}

REQUIRED_MODEL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "was_previously_contacted",
    "pdays_clean",
]

MIN_RECOMMENDED_BATCH_SIZE = 10


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run src/train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def load_selected_policy() -> pd.DataFrame:
    if not SELECTED_POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Selected policy file not found: {SELECTED_POLICY_PATH}. Run src/evaluate.py first."
        )

    return pd.read_csv(SELECTED_POLICY_PATH)


def get_policy_details() -> dict[str, Any]:
    policy = load_selected_policy()
    row = policy.iloc[0].to_dict()

    selected_contact_rate = float(row["selected_contact_rate"])
    policy_cutoff = float(row.get("test_score_cutoff", row.get("validation_score_cutoff")))

    return {
        "selected_contact_rate": selected_contact_rate,
        "selected_contact_rate_label": f"Top {int(selected_contact_rate * 100)}%",
        "policy_cutoff": policy_cutoff,
        "test_precision_at_k": float(row["test_precision_at_k"]),
        "test_capture_rate_at_k": float(row["test_capture_rate_at_k"]),
        "test_lift_at_k": float(row["test_lift_at_k"]),
    }


def engineer_pdays(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()

    if "pdays" in output.columns:
        output["was_previously_contacted"] = (output["pdays"] != 999).astype(int)
        output["pdays_clean"] = output["pdays"].where(output["pdays"] != 999, 0)
        output = output.drop(columns=["pdays"])

    return output


def clean_input_columns(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()

    columns_to_drop = [
        col for col in output.columns
        if col in TARGET_COLUMNS or col in LEAKAGE_COLUMNS
    ]

    if columns_to_drop:
        output = output.drop(columns=columns_to_drop)

    return output


def prepare_model_input(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    output = engineer_pdays(output)
    output = clean_input_columns(output)

    missing_features = [
        col for col in REQUIRED_MODEL_FEATURES
        if col not in output.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required model features: {missing_features}")

    return output[REQUIRED_MODEL_FEATURES].copy()


def predict_probabilities(data: pd.DataFrame) -> pd.Series:
    model = load_model()
    model_input = prepare_model_input(data)
    probabilities = model.predict_proba(model_input)[:, 1]

    return pd.Series(probabilities, index=data.index, name="predicted_probability")


def score_single_customer(customer_data: dict[str, Any]) -> dict[str, Any]:
    policy_details = get_policy_details()
    input_df = pd.DataFrame([customer_data])

    probability = float(predict_probabilities(input_df).iloc[0])
    policy_cutoff = policy_details["policy_cutoff"]

    if probability >= policy_cutoff:
        recommendation = "High priority for contact"
    else:
        recommendation = "Lower priority for contact"

    return {
        "predicted_probability": probability,
        "recommendation": recommendation,
        "policy_used": policy_details["selected_contact_rate_label"],
        "policy_reference_cutoff": policy_cutoff,
        "note": "Single-customer scoring uses the selected policy cutoff as a reference. The top-k policy is strongest when ranking a batch of customers.",
    }


def prepare_output_frame(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()

    if "actual_subscription" not in output.columns:
        if "y_binary" in output.columns:
            output["actual_subscription"] = output["y_binary"]
        elif "y" in output.columns:
            output["actual_subscription"] = output["y"].map({"yes": 1, "no": 0})

    columns_to_drop = [
        col for col in ["duration", "y", "y_binary"]
        if col in output.columns
    ]

    if columns_to_drop:
        output = output.drop(columns=columns_to_drop)

    if "customer_id" not in output.columns:
        output.insert(
            0,
            "customer_id",
            [f"uploaded_{i + 1:06d}" for i in range(len(output))],
        )

    return output


def score_batch_customers(data: pd.DataFrame) -> pd.DataFrame:
    policy_details = get_policy_details()
    selected_contact_rate = policy_details["selected_contact_rate"]

    output = prepare_output_frame(data)
    probabilities = predict_probabilities(data)

    output["predicted_probability"] = probabilities.values
    output = output.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    output["score_rank"] = range(1, len(output) + 1)
    output["score_percentile"] = 1 - ((output["score_rank"] - 1) / len(output))

    selected_count = max(1, math.ceil(len(output) * selected_contact_rate))
    selected_column = f"recommended_selected_top_{int(selected_contact_rate * 100)}"
    output[selected_column] = output["score_rank"] <= selected_count

    output["recommendation"] = output[selected_column].map(
        {
            True: "Recommended for contact",
            False: "Not selected under current policy",
        }
    )

    output["policy_used"] = policy_details["selected_contact_rate_label"]
    output["batch_size"] = len(output)
    output["selected_customer_count"] = selected_count

    return output


def get_batch_size_note(batch_size: int) -> str:
    if batch_size < MIN_RECOMMENDED_BATCH_SIZE:
        return (
            f"This batch has only {batch_size} row(s). Top-k targeting is relative to the uploaded batch, "
            "so very small batches are useful for testing but less meaningful for campaign planning."
        )

    return ""


def get_required_feature_template() -> pd.DataFrame:
    rows = [
        {
            "age": 40,
            "job": "admin.",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "mon",
            "campaign": 2,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": 1.1,
            "cons.price.idx": 93.994,
            "cons.conf.idx": -36.4,
            "euribor3m": 4.857,
            "nr.employed": 5191.0,
        },
        {
            "age": 32,
            "job": "student",
            "marital": "single",
            "education": "university.degree",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "mar",
            "day_of_week": "tue",
            "campaign": 1,
            "pdays": 6,
            "previous": 2,
            "poutcome": "success",
            "emp.var.rate": -1.8,
            "cons.price.idx": 92.893,
            "cons.conf.idx": -46.2,
            "euribor3m": 1.313,
            "nr.employed": 5099.1,
        },
        {
            "age": 67,
            "job": "retired",
            "marital": "married",
            "education": "basic.4y",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "oct",
            "day_of_week": "wed",
            "campaign": 1,
            "pdays": 999,
            "previous": 1,
            "poutcome": "failure",
            "emp.var.rate": -3.4,
            "cons.price.idx": 92.431,
            "cons.conf.idx": -26.9,
            "euribor3m": 0.754,
            "nr.employed": 5017.5,
        },
        {
            "age": 45,
            "job": "blue-collar",
            "marital": "married",
            "education": "basic.9y",
            "default": "unknown",
            "housing": "yes",
            "loan": "yes",
            "contact": "telephone",
            "month": "may",
            "day_of_week": "fri",
            "campaign": 4,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": 1.1,
            "cons.price.idx": 93.994,
            "cons.conf.idx": -36.4,
            "euribor3m": 4.857,
            "nr.employed": 5191.0,
        },
        {
            "age": 29,
            "job": "technician",
            "marital": "single",
            "education": "professional.course",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "aug",
            "day_of_week": "thu",
            "campaign": 2,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": 1.4,
            "cons.price.idx": 93.444,
            "cons.conf.idx": -36.1,
            "euribor3m": 4.963,
            "nr.employed": 5228.1,
        },
        {
            "age": 53,
            "job": "management",
            "marital": "divorced",
            "education": "university.degree",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "sep",
            "day_of_week": "mon",
            "campaign": 1,
            "pdays": 3,
            "previous": 3,
            "poutcome": "success",
            "emp.var.rate": -3.4,
            "cons.price.idx": 92.379,
            "cons.conf.idx": -29.8,
            "euribor3m": 0.809,
            "nr.employed": 5017.5,
        },
        {
            "age": 38,
            "job": "services",
            "marital": "married",
            "education": "high.school",
            "default": "no",
            "housing": "yes",
            "loan": "yes",
            "contact": "telephone",
            "month": "jun",
            "day_of_week": "tue",
            "campaign": 5,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": 1.4,
            "cons.price.idx": 94.465,
            "cons.conf.idx": -41.8,
            "euribor3m": 4.864,
            "nr.employed": 5228.1,
        },
        {
            "age": 24,
            "job": "student",
            "marital": "single",
            "education": "high.school",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "dec",
            "day_of_week": "wed",
            "campaign": 1,
            "pdays": 999,
            "previous": 2,
            "poutcome": "failure",
            "emp.var.rate": -3.0,
            "cons.price.idx": 92.713,
            "cons.conf.idx": -33.0,
            "euribor3m": 0.715,
            "nr.employed": 5023.5,
        },
        {
            "age": 41,
            "job": "admin.",
            "marital": "single",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "jul",
            "day_of_week": "thu",
            "campaign": 2,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": 1.4,
            "cons.price.idx": 93.918,
            "cons.conf.idx": -42.7,
            "euribor3m": 4.962,
            "nr.employed": 5228.1,
        },
        {
            "age": 58,
            "job": "unemployed",
            "marital": "married",
            "education": "basic.6y",
            "default": "no",
            "housing": "unknown",
            "loan": "unknown",
            "contact": "cellular",
            "month": "apr",
            "day_of_week": "fri",
            "campaign": 1,
            "pdays": 12,
            "previous": 1,
            "poutcome": "success",
            "emp.var.rate": -1.8,
            "cons.price.idx": 93.075,
            "cons.conf.idx": -47.1,
            "euribor3m": 1.405,
            "nr.employed": 5099.1,
        },
    ]

    return pd.DataFrame(rows)
