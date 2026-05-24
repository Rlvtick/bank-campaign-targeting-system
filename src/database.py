from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
DATABASE_DIR = DATA_DIR / "database"

REPORTS_DIR = PROJECT_ROOT / "reports"
SQL_DIR = PROJECT_ROOT / "sql"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

DATABASE_PATH = DATABASE_DIR / "bank_campaign.db"

TEST_DATA_PATH = PROCESSED_DIR / "test.csv"
CUSTOMER_SCORES_PATH = PREDICTIONS_DIR / "customer_scores.csv"

THRESHOLD_VALIDATION_PATH = REPORTS_DIR / "threshold_simulation_validation.csv"
THRESHOLD_TEST_PATH = REPORTS_DIR / "threshold_simulation_test.csv"

FIXED_THRESHOLD_VALIDATION_PATH = REPORTS_DIR / "fixed_threshold_simulation_validation.csv"
FIXED_THRESHOLD_TEST_PATH = REPORTS_DIR / "fixed_threshold_simulation_test.csv"

SELECTED_POLICY_PATH = REPORTS_DIR / "selected_threshold_policy.csv"
CREATE_TABLES_SQL_PATH = SQL_DIR / "create_tables.sql"

DATABASE_SUMMARY_PATH = REPORTS_DIR / "database_summary.csv"
DASHBOARD_CUSTOMER_SCORES_PATH = DASHBOARD_DIR / "customer_scores.csv"
DASHBOARD_THRESHOLD_SIMULATION_PATH = DASHBOARD_DIR / "threshold_simulation.csv"
DASHBOARD_SELECTED_POLICY_PATH = DASHBOARD_DIR / "selected_policy.csv"


def read_csv_checked(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_csv(path)


def get_selected_flag_column(customer_scores: pd.DataFrame) -> str:
    selected_columns = [
        col for col in customer_scores.columns
        if col.startswith("recommended_selected_top_")
    ]

    if len(selected_columns) != 1:
        raise ValueError(
            f"Expected exactly one selected policy column, found: {selected_columns}"
        )

    return selected_columns[0]


def prepare_processed_customers(customer_scores: pd.DataFrame) -> pd.DataFrame:
    score_columns = {
        "predicted_probability",
        "predicted_label_0_5",
        "score_rank",
        "score_percentile",
        "y",
        "y_binary",
    }

    recommendation_columns = [
        col for col in customer_scores.columns
        if col.startswith("recommended_")
    ]

    excluded_columns = score_columns.union(recommendation_columns)

    feature_columns = [
        col for col in customer_scores.columns
        if col not in excluded_columns
    ]

    processed_customers = customer_scores[feature_columns].copy()

    return processed_customers


def prepare_model_scores(
    customer_scores: pd.DataFrame,
    selected_flag_column: str,
    selected_policy: pd.DataFrame,
) -> pd.DataFrame:
    recommendation_columns = [
        col for col in customer_scores.columns
        if col.startswith("recommended_top_")
    ]

    base_columns = [
        "customer_id",
        "split",
        "actual_subscription",
        "predicted_probability",
        "predicted_label_0_5",
        "score_rank",
        "score_percentile",
        selected_flag_column,
    ]

    score_columns = base_columns + recommendation_columns
    model_scores = customer_scores[score_columns].copy()

    model_scores = model_scores.rename(
        columns={selected_flag_column: "selected_policy_flag"}
    )

    selected_contact_rate = float(selected_policy.loc[0, "selected_contact_rate"])
    model_scores["selected_contact_rate"] = selected_contact_rate

    boolean_columns = [
        col for col in model_scores.columns
        if col.startswith("recommended_") or col == "selected_policy_flag"
    ]

    for col in boolean_columns:
        model_scores[col] = model_scores[col].astype(int)

    return model_scores


def prepare_dashboard_customer_scores(
    processed_customers: pd.DataFrame,
    model_scores: pd.DataFrame,
) -> pd.DataFrame:
    selected_customer_columns = [
        "customer_id",
        "age",
        "job",
        "marital",
        "education",
        "contact",
        "month",
        "campaign",
        "previous",
        "poutcome",
        "actual_subscription",
    ]

    available_customer_columns = [
        col for col in selected_customer_columns
        if col in processed_customers.columns
    ]

    dashboard_scores = model_scores.merge(
        processed_customers[available_customer_columns],
        on="customer_id",
        how="left",
        suffixes=("", "_customer"),
    )

    dashboard_scores["score_band"] = pd.cut(
        dashboard_scores["predicted_probability"],
        bins=[0, 0.20, 0.40, 0.60, 0.80, 1.00],
        labels=[
            "0.00 to 0.19",
            "0.20 to 0.39",
            "0.40 to 0.59",
            "0.60 to 0.79",
            "0.80 to 1.00",
        ],
        include_lowest=True,
    )

    dashboard_scores = dashboard_scores.sort_values(
        "score_rank",
        ascending=True,
    ).reset_index(drop=True)

    return dashboard_scores


def write_table(
    connection: sqlite3.Connection,
    dataframe: pd.DataFrame,
    table_name: str,
) -> None:
    dataframe.to_sql(
        table_name,
        connection,
        if_exists="replace",
        index=False,
    )


def create_views(connection: sqlite3.Connection) -> None:
    if not CREATE_TABLES_SQL_PATH.exists():
        raise FileNotFoundError(f"SQL file not found: {CREATE_TABLES_SQL_PATH}")

    sql_text = CREATE_TABLES_SQL_PATH.read_text()
    connection.executescript(sql_text)


def get_table_counts(connection: sqlite3.Connection) -> pd.DataFrame:
    tables = [
        "processed_customers",
        "model_scores",
        "threshold_simulation",
        "fixed_threshold_simulation",
        "selected_policy",
    ]

    rows = []

    for table in tables:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS row_count FROM {table}",
            connection,
        ).loc[0, "row_count"]

        rows.append(
            {
                "object_name": table,
                "object_type": "table",
                "row_count": int(count),
            }
        )

    views = [
        "top_recommended_customers_view",
        "selected_vs_non_selected_view",
        "score_distribution_view",
        "segment_performance_view",
    ]

    for view in views:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS row_count FROM {view}",
            connection,
        ).loc[0, "row_count"]

        rows.append(
            {
                "object_name": view,
                "object_type": "view",
                "row_count": int(count),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading source files...")
    customer_scores = read_csv_checked(CUSTOMER_SCORES_PATH, "Customer scores")
    selected_policy = read_csv_checked(SELECTED_POLICY_PATH, "Selected policy")

    threshold_validation = read_csv_checked(
        THRESHOLD_VALIDATION_PATH,
        "Validation threshold simulation",
    )
    threshold_test = read_csv_checked(
        THRESHOLD_TEST_PATH,
        "Test threshold simulation",
    )

    fixed_threshold_validation = read_csv_checked(
        FIXED_THRESHOLD_VALIDATION_PATH,
        "Validation fixed threshold simulation",
    )
    fixed_threshold_test = read_csv_checked(
        FIXED_THRESHOLD_TEST_PATH,
        "Test fixed threshold simulation",
    )

    selected_flag_column = get_selected_flag_column(customer_scores)

    print("Preparing database tables...")
    processed_customers = prepare_processed_customers(customer_scores)

    model_scores = prepare_model_scores(
        customer_scores,
        selected_flag_column,
        selected_policy,
    )

    threshold_simulation = pd.concat(
        [threshold_validation, threshold_test],
        ignore_index=True,
    )

    fixed_threshold_simulation = pd.concat(
        [fixed_threshold_validation, fixed_threshold_test],
        ignore_index=True,
    )

    dashboard_scores = prepare_dashboard_customer_scores(
        processed_customers,
        model_scores,
    )

    dashboard_threshold_simulation = threshold_simulation.copy()

    print("Writing SQLite database...")
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()

    with sqlite3.connect(DATABASE_PATH) as connection:
        write_table(connection, processed_customers, "processed_customers")
        write_table(connection, model_scores, "model_scores")
        write_table(connection, threshold_simulation, "threshold_simulation")
        write_table(connection, fixed_threshold_simulation, "fixed_threshold_simulation")
        write_table(connection, selected_policy, "selected_policy")

        create_views(connection)

        database_summary = get_table_counts(connection)

        selected_vs_non_selected = pd.read_sql_query(
            "SELECT * FROM selected_vs_non_selected_view",
            connection,
        )

        score_distribution = pd.read_sql_query(
            "SELECT * FROM score_distribution_view",
            connection,
        )

    print("Writing dashboard exports...")
    dashboard_scores.to_csv(DASHBOARD_CUSTOMER_SCORES_PATH, index=False)
    dashboard_threshold_simulation.to_csv(
        DASHBOARD_THRESHOLD_SIMULATION_PATH,
        index=False,
    )
    selected_policy.to_csv(DASHBOARD_SELECTED_POLICY_PATH, index=False)

    database_summary.to_csv(DATABASE_SUMMARY_PATH, index=False)

    selected_contact_rate = float(selected_policy.loc[0, "selected_contact_rate"])

    print("\nSQL and database layer completed.")
    print("=" * 70)
    print(f"SQLite database: {DATABASE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Selected contact rate: {selected_contact_rate:.0%}")
    print("=" * 70)
    print("\nDatabase summary:")
    print(database_summary.to_string(index=False))
    print("=" * 70)
    print("\nSelected versus non-selected view:")
    print(selected_vs_non_selected.to_string(index=False))
    print("=" * 70)
    print("\nScore distribution view:")
    print(score_distribution.to_string(index=False))
    print("=" * 70)
    print("\nGenerated files:")
    print(f"- {DATABASE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {DATABASE_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {DASHBOARD_CUSTOMER_SCORES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {DASHBOARD_THRESHOLD_SIMULATION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- {DASHBOARD_SELECTED_POLICY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
