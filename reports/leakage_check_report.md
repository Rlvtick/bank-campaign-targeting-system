# Leakage Check Report

## Main leakage decision

The `duration` column is excluded from the deployable modeling dataset.

## Reason

`duration` represents the last contact duration in seconds. In a real pre-call campaign targeting workflow, this value would only be known after the customer has already been contacted.

Because this project is designed to support customer prioritization before contact, using `duration` would create target leakage.

## Duplicate handling

Exact duplicate rows are removed from the model-ready dataset before modeling.

- Raw rows: 41188
- Exact duplicate rows removed: 12
- Model-ready rows: 41176

## Implementation

- Raw dataset columns: 21
- Model-ready columns: 21
- `duration` exists in raw data: True
- `duration` removed from model-ready data: True

## Output

The model-ready dataset is saved locally at:

`data/processed/bank_marketing_model_ready.csv`

This processed CSV is not committed to GitHub because it is generated from the raw public dataset.
