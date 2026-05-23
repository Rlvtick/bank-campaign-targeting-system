# Bank Campaign Targeting System

## Project Overview

This project builds a small end-to-end machine learning workflow for bank campaign targeting using public data. The goal is to predict which customers are more likely to subscribe to a term deposit and use the prediction scores to support campaign prioritization.

The project is designed as a portfolio project for a data science role. It includes data preparation, exploratory data analysis, machine learning modeling, SQL-based score storage, a simple prediction app, a local API endpoint, and a BI dashboard.

## Business Problem

Banks often run marketing campaigns through customer contact channels such as phone calls. Calling every customer can be inefficient, especially when only a small portion of customers are likely to subscribe.

This project tries to answer:

> Which customers should be prioritized for a term deposit marketing campaign?

## Dataset

The project uses the UCI Bank Marketing dataset. The target variable is `y`, which indicates whether a customer subscribed to a term deposit.

The preferred dataset file is `bank-additional-full.csv`.

## Leakage Handling

The `duration` feature represents the call duration and is only known after a customer has already been contacted. Because this project focuses on pre-call campaign targeting, `duration` will be excluded from the deployable model to avoid target leakage.

## Planned Workflow

1. Data access and raw data storage
2. Data validation and leakage check
3. Exploratory data analysis
4. Preprocessing pipeline
5. Baseline modeling
6. Main model training
7. Evaluation and threshold simulation
8. SQL database layer
9. Streamlit demo
10. FastAPI local endpoint
11. BI dashboard
12. Documentation and presentation

## Tools

- Python
- pandas
- scikit-learn
- XGBoost
- SQLite
- Streamlit
- FastAPI
- Tableau Public

## Limitations

This project uses a public dataset as a proxy for a banking campaign targeting problem. It does not use Rakuten Bank data, Japanese bank data, or real proprietary customer data. The result should be interpreted as a portfolio-scale simulation, not a production-ready banking system.

## References

- UCI Machine Learning Repository: Bank Marketing Dataset
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing.
