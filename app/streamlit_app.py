from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    get_batch_size_note,
    get_policy_details,
    get_required_feature_template,
    score_batch_customers,
    score_single_customer,
)


REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
TEST_TOPK_PATH = REPORTS_DIR / "threshold_simulation_test.csv"
DASHBOARD_CUSTOMER_SCORES_PATH = DASHBOARD_DIR / "customer_scores.csv"


st.set_page_config(
    page_title="Bank Campaign Targeting System",
    page_icon="🏦",
    layout="wide",
)


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero-card {
    padding: 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #102033 0%, #162B46 55%, #19395F 100%);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #D2DAE6;
    max-width: 950px;
}

.metric-card {
    padding: 1.1rem 1.25rem;
    border-radius: 16px;
    background-color: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
}

.metric-label {
    font-size: 0.85rem;
    color: #AEB8C7;
    margin-bottom: 0.2rem;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 750;
    color: #FFFFFF;
}

.section-card {
    padding: 1.25rem;
    border-radius: 16px;
    background-color: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1rem;
}

.small-muted {
    color: #AEB8C7;
    font-size: 0.9rem;
}

.step-box {
    padding: 1rem;
    border-radius: 14px;
    background-color: rgba(45, 122, 255, 0.08);
    border: 1px solid rgba(45, 122, 255, 0.18);
    height: 100%;
}

.step-number {
    font-size: 0.8rem;
    color: #79AFFF;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.step-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.warning-soft {
    padding: 1rem;
    border-radius: 14px;
    background-color: rgba(255, 193, 7, 0.10);
    border: 1px solid rgba(255, 193, 7, 0.25);
}
</style>
"""


JOB_OPTIONS = {
    "Administrative staff": "admin.",
    "Blue-collar worker": "blue-collar",
    "Entrepreneur": "entrepreneur",
    "Housemaid": "housemaid",
    "Management": "management",
    "Retired": "retired",
    "Self-employed": "self-employed",
    "Services": "services",
    "Student": "student",
    "Technician": "technician",
    "Unemployed": "unemployed",
    "Unknown": "unknown",
}

MARITAL_OPTIONS = {
    "Married": "married",
    "Single": "single",
    "Divorced": "divorced",
    "Unknown": "unknown",
}

EDUCATION_OPTIONS = {
    "Basic education, 4 years": "basic.4y",
    "Basic education, 6 years": "basic.6y",
    "Basic education, 9 years": "basic.9y",
    "High school": "high.school",
    "Illiterate": "illiterate",
    "Professional course": "professional.course",
    "University degree": "university.degree",
    "Unknown": "unknown",
}

YES_NO_UNKNOWN_OPTIONS = {
    "No": "no",
    "Yes": "yes",
    "Unknown": "unknown",
}

CONTACT_OPTIONS = {
    "Cellular": "cellular",
    "Telephone": "telephone",
}

MONTH_OPTIONS = {
    "January": "jan",
    "February": "feb",
    "March": "mar",
    "April": "apr",
    "May": "may",
    "June": "jun",
    "July": "jul",
    "August": "aug",
    "September": "sep",
    "October": "oct",
    "November": "nov",
    "December": "dec",
}

DAY_OPTIONS = {
    "Monday": "mon",
    "Tuesday": "tue",
    "Wednesday": "wed",
    "Thursday": "thu",
    "Friday": "fri",
}

POUTCOME_OPTIONS = {
    "Previous campaign failed": "failure",
    "No previous campaign": "nonexistent",
    "Previous campaign succeeded": "success",
}


FEATURE_NAME_MAP = {
    "nr.employed": "Employment level",
    "emp.var.rate": "Employment variation rate",
    "was_previously_contacted": "Previously contacted",
    "cons.conf.idx": "Consumer confidence index",
    "cons.price.idx": "Consumer price index",
    "euribor3m": "Euribor 3 month rate",
    "pdays_clean": "Days since previous contact",
    "campaign": "Campaign contacts",
    "previous": "Previous campaign contacts",
    "poutcome_success": "Previous campaign success",
    "poutcome_failure": "Previous campaign failure",
    "contact_cellular": "Cellular contact",
    "contact_telephone": "Telephone contact",
    "default_no": "No credit default",
    "default_unknown": "Unknown credit default",
    "month_mar": "Contact month: March",
    "month_apr": "Contact month: April",
    "month_may": "Contact month: May",
    "month_sep": "Contact month: September",
    "month_oct": "Contact month: October",
    "month_nov": "Contact month: November",
    "day_of_week_fri": "Contact day: Friday",
}


@st.cache_data
def load_policy_details():
    return get_policy_details()


@st.cache_data
def load_feature_importance():
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame()

    data = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    data["display_feature"] = data["feature"].apply(prettify_feature_name)

    return data


@st.cache_data
def load_threshold_simulation():
    if not TEST_TOPK_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(TEST_TOPK_PATH)


@st.cache_data
def load_dashboard_scores():
    if not DASHBOARD_CUSTOMER_SCORES_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(DASHBOARD_CUSTOMER_SCORES_PATH)


def prettify_feature_name(feature_name):
    cleaned = (
        feature_name
        .replace("categorical__", "")
        .replace("numerical__", "")
    )

    if cleaned in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[cleaned]

    return cleaned.replace("_", " ").replace(".", " ").title()


def selectbox_from_mapping(label, options, default_value):
    labels = list(options.keys())
    values = list(options.values())

    if default_value in values:
        default_index = values.index(default_value)
    else:
        default_index = 0

    selected_label = st.selectbox(label, labels, index=default_index)

    return options[selected_label]


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_header(policy_details):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Bank Campaign Targeting System</div>
            <div class="hero-subtitle">
                A portfolio demo that ranks bank customers by predicted term-deposit subscription propensity, then recommends the highest-scored {policy_details["selected_contact_rate_label"]} for campaign outreach.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Selected policy", policy_details["selected_contact_rate_label"])

    with col2:
        metric_card("Test precision@k", f"{policy_details['test_precision_at_k']:.2%}")

    with col3:
        metric_card("Test capture rate@k", f"{policy_details['test_capture_rate_at_k']:.2%}")

    with col4:
        metric_card("Test lift@k", f"{policy_details['test_lift_at_k']:.2f}x")


def show_project_summary(policy_details):
    st.subheader("Project Summary")

    st.markdown(
        """
        <div class="section-card">
            This app simulates how a bank could prioritize campaign outreach using a machine learning propensity model.
            Instead of contacting every customer, the model ranks customers by predicted subscription likelihood and recommends the highest-scored customer group.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">STEP 1</div>
                <div class="step-title">Score customers</div>
                <div class="small-muted">The trained XGBoost model estimates each customer's subscription propensity.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">STEP 2</div>
                <div class="step-title">Rank by score</div>
                <div class="small-muted">Customers are sorted from highest to lowest predicted probability.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="step-box">
                <div class="step-number">STEP 3</div>
                <div class="step-title">Recommend {policy_details["selected_contact_rate_label"]}</div>
                <div class="small-muted">The highest-ranked group is recommended for campaign contact.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    left, right = st.columns([1.1, 1])

    with left:
        show_threshold_chart(policy_details)

    with right:
        show_selected_vs_non_selected()

    st.divider()

    show_top_model_drivers()

    st.markdown(
        """
        <div class="warning-soft">
            This is a public-data portfolio simulation built with the UCI Bank Marketing dataset. The app is intended to demonstrate an end-to-end campaign targeting workflow and should not be interpreted as a production-ready banking system or real customer decisioning tool.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_threshold_chart(policy_details):
    topk = load_threshold_simulation()

    st.markdown("#### Targeting Tradeoff")

    if topk.empty:
        st.info("Threshold simulation file was not found.")
        return

    plot_data = topk[["contact_rate", "precision_at_k", "capture_rate_at_k"]].copy()
    plot_data["contact_rate_label"] = (plot_data["contact_rate"] * 100).round(0).astype(int).astype(str) + "%"

    melted = plot_data.melt(
        id_vars=["contact_rate", "contact_rate_label"],
        value_vars=["precision_at_k", "capture_rate_at_k"],
        var_name="metric",
        value_name="value",
    )

    metric_labels = {
        "precision_at_k": "Precision@k",
        "capture_rate_at_k": "Capture Rate@k",
    }

    melted["metric"] = melted["metric"].map(metric_labels)

    fig = px.line(
        melted,
        x="contact_rate",
        y="value",
        color="metric",
        markers=True,
        labels={
            "contact_rate": "Contact Rate",
            "value": "Metric Value",
            "metric": "Metric",
        },
    )

    fig.add_vline(
        x=policy_details["selected_contact_rate"],
        line_dash="dash",
        line_color="gray",
    )

    fig.update_layout(
        height=390,
        yaxis_tickformat=".0%",
        xaxis_tickformat=".0%",
        legend_title_text="",
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The selected policy balances subscriber capture and campaign efficiency. Higher contact rates capture more subscribers but reduce precision."
    )


def show_selected_vs_non_selected():
    scores = load_dashboard_scores()

    st.markdown("#### Selected vs Non-selected")

    if scores.empty or "selected_policy_flag" not in scores.columns:
        st.info("Dashboard customer score file was not found.")
        return

    group_summary = (
        scores.assign(
            customer_group=scores["selected_policy_flag"].map(
                {
                    1: "Selected top-policy group",
                    0: "Not selected",
                }
            )
        )
        .groupby("customer_group")
        .agg(
            customers=("customer_id", "count"),
            observed_subscription_rate=("actual_subscription", "mean"),
            average_score=("predicted_probability", "mean"),
        )
        .reset_index()
    )

    display_summary = group_summary.copy()
    display_summary["observed_subscription_rate"] = display_summary["observed_subscription_rate"].map(lambda x: f"{x:.2%}")
    display_summary["average_score"] = display_summary["average_score"].map(lambda x: f"{x:.2%}")

    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    fig = px.bar(
        group_summary,
        x="customer_group",
        y="observed_subscription_rate",
        text=group_summary["observed_subscription_rate"].map(lambda x: f"{x:.2%}"),
        labels={
            "customer_group": "",
            "observed_subscription_rate": "Observed Subscription Rate",
        },
    )

    fig.update_layout(
        height=300,
        yaxis_tickformat=".0%",
        margin=dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def show_top_model_drivers():
    feature_importance = load_feature_importance()

    st.markdown("#### Top Model Drivers")

    if feature_importance.empty:
        st.info("Feature importance file was not found.")
        return

    top_features = feature_importance.head(12).copy()
    top_features = top_features.sort_values("importance", ascending=True)

    fig = px.bar(
        top_features,
        x="importance",
        y="display_feature",
        orientation="h",
        labels={
            "importance": "Importance",
            "display_feature": "Feature",
        },
    )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These are simplified global model drivers from XGBoost feature importance. They describe model behavior, not causal effects."
    )


def build_single_customer_form():
    with st.form("single_customer_form"):
        st.subheader("Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=17, max_value=100, value=40)
            job = selectbox_from_mapping("Job", JOB_OPTIONS, "admin.")
            marital = selectbox_from_mapping("Marital status", MARITAL_OPTIONS, "married")
            education = selectbox_from_mapping("Education", EDUCATION_OPTIONS, "university.degree")
            default = selectbox_from_mapping("Credit default", YES_NO_UNKNOWN_OPTIONS, "no")
            housing = selectbox_from_mapping("Housing loan", YES_NO_UNKNOWN_OPTIONS, "yes")

        with col2:
            loan = selectbox_from_mapping("Personal loan", YES_NO_UNKNOWN_OPTIONS, "no")
            contact = selectbox_from_mapping("Contact type", CONTACT_OPTIONS, "cellular")
            month = selectbox_from_mapping("Last contact month", MONTH_OPTIONS, "may")
            day_of_week = selectbox_from_mapping("Last contact day", DAY_OPTIONS, "mon")
            campaign = st.number_input("Campaign contacts", min_value=1, max_value=60, value=2)
            pdays = st.number_input("Days since previous contact", min_value=0, max_value=999, value=999)

        with col3:
            previous = st.number_input("Previous campaign contacts", min_value=0, max_value=20, value=0)
            poutcome = selectbox_from_mapping("Previous campaign outcome", POUTCOME_OPTIONS, "nonexistent")
            emp_var_rate = st.number_input("Employment variation rate", value=1.1)
            cons_price_idx = st.number_input("Consumer price index", value=93.994)
            cons_conf_idx = st.number_input("Consumer confidence index", value=-36.4)
            euribor3m = st.number_input("Euribor 3 month rate", value=4.857)
            nr_employed = st.number_input("Number of employees", value=5191.0)

        submitted = st.form_submit_button("Score customer")

    customer_data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }

    return submitted, customer_data


def show_single_customer_result(customer_data):
    try:
        result = score_single_customer(customer_data)
    except Exception as exc:
        st.error(f"Unable to score customer: {exc}")
        return

    probability = result["predicted_probability"]

    col1, col2 = st.columns(2)
    col1.metric("Predicted Subscription Score", f"{probability:.2%}")
    col2.metric("Policy Reference Cutoff", f"{result['policy_reference_cutoff']:.2%}")

    if result["recommendation"] == "High priority for contact":
        st.success(result["recommendation"])
    else:
        st.warning(result["recommendation"])

    st.caption(result["note"])

    show_input_explanations()


def show_input_explanations():
    st.divider()
    st.markdown("#### Input Guide")

    with st.expander("Customer profile fields"):
        st.write(
            """
            - **Age:** Customer age.
            - **Job:** Customer occupation category.
            - **Marital status:** Customer marital status.
            - **Education:** Customer education category.
            - **Credit default:** Whether the customer has credit in default.
            - **Housing loan:** Whether the customer has a housing loan.
            - **Personal loan:** Whether the customer has a personal loan.
            """
        )

    with st.expander("Campaign and contact fields"):
        st.write(
            """
            - **Contact type:** Communication channel used for the campaign.
            - **Last contact month/day:** Timing of the last campaign contact.
            - **Campaign contacts:** Number of contacts during the current campaign.
            - **Days since previous contact:** Use 999 when the customer was not previously contacted.
            - **Previous campaign contacts:** Number of contacts before the current campaign.
            - **Previous campaign outcome:** Result of the previous campaign, if any.
            """
        )

    with st.expander("Advanced economic context fields"):
        st.write(
            """
            These fields come from the original UCI dataset and describe the economic context around the campaign period.

            - **Employment variation rate:** Labor-market context indicator.
            - **Consumer price index:** Price-level indicator.
            - **Consumer confidence index:** Consumer sentiment indicator.
            - **Euribor 3 month rate:** Short-term interest-rate reference.
            - **Number of employees:** Employment-level indicator in the original dataset.
            """
        )


def show_single_customer_scoring():
    st.subheader("Single Customer Scoring")
    st.write(
        "Use this page to test how one customer profile is scored. The result is compared with the selected policy cutoff as a reference."
    )

    st.caption(
        "The full top-k targeting policy is strongest when ranking a batch of customers, not when viewing one customer in isolation."
    )

    submitted, customer_data = build_single_customer_form()

    if submitted:
        show_single_customer_result(customer_data)


def show_batch_scoring(policy_details):
    st.subheader("Batch Customer Scoring")
    st.write(
        "Upload a CSV with the required customer feature columns. The app will score all rows, rank customers, "
        f"and recommend the highest-scored {policy_details['selected_contact_rate_label']}."
    )

    st.caption(
        "Batch scoring is the preferred workflow for this project because the selected policy is based on ranking customers within a campaign list."
    )

    template = get_required_feature_template()
    st.download_button(
        label="Download sample batch input",
        data=template.to_csv(index=False),
        file_name="sample_customer_input.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded_file is None:
        st.dataframe(template, use_container_width=True)
        return

    try:
        input_data = pd.read_csv(uploaded_file)
        scored_data = score_batch_customers(input_data)
    except Exception as exc:
        st.error(f"Unable to score uploaded file: {exc}")
        return

    batch_note = get_batch_size_note(len(input_data))

    if batch_note:
        st.warning(batch_note)

    selected_col = [
        col for col in scored_data.columns
        if col.startswith("recommended_selected_top_")
    ][0]

    total_rows = len(scored_data)
    selected_rows = int(scored_data[selected_col].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Uploaded Customers", f"{total_rows:,}")
    col2.metric("Recommended Customers", f"{selected_rows:,}")
    col3.metric("Average Score", f"{scored_data['predicted_probability'].mean():.2%}")

    if "actual_subscription" in scored_data.columns:
        selected_actual_rate = scored_data.loc[
            scored_data[selected_col],
            "actual_subscription",
        ].mean()
        st.metric("Observed Subscription Rate in Selected Group", f"{selected_actual_rate:.2%}")
        st.caption("This metric is shown only when historical labels are included in the uploaded file.")

    display_columns = [
        col for col in [
            "customer_id",
            "predicted_probability",
            "score_rank",
            "score_percentile",
            selected_col,
            "recommendation",
            "policy_used",
            "age",
            "job",
            "education",
            "contact",
            "month",
            "poutcome",
            "actual_subscription",
        ]
        if col in scored_data.columns
    ]

    st.dataframe(scored_data[display_columns].head(50), use_container_width=True)

    st.download_button(
        label="Download scored customers",
        data=scored_data.to_csv(index=False),
        file_name="scored_customers.csv",
        mime="text/csv",
    )


def main():
    policy_details = load_policy_details()
    show_header(policy_details)

    tab1, tab2, tab3 = st.tabs(
        [
            "Project Summary",
            "Single Customer Scoring",
            "Batch Scoring",
        ]
    )

    with tab1:
        show_project_summary(policy_details)

    with tab2:
        show_single_customer_scoring()

    with tab3:
        show_batch_scoring(policy_details)


if __name__ == "__main__":
    main()
