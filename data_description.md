# Sample Dataset Documentation

## Overview

`sample_data.csv` and `sample_data.xlsx` contain a **Customer Churn Prediction** dataset designed to test all features of the ML Model Comparison Dashboard.

- **Rows**: 800 customers
- **Features**: 15 columns (1 ID + 13 features + 1 target)
- **Task**: Binary classification — predict whether a customer will churn
- **Class balance**: ~66% No Churn / ~34% Churn (realistic, within 60-40 range)

---

## Features

| Column | Type | Description |
|---|---|---|
| `CustomerID` | String (ID) | Unique customer identifier (e.g., CUST0001) |
| `Age` | Integer | Customer age in years (18–80) |
| `Gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary (0/1) | Whether the customer is a senior citizen |
| `Tenure_Months` | Integer | Months the customer has been with the company (1–72) |
| `ContractType` | Categorical | Month-to-month / One year / Two year |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `MonthlyCharges` | Float | Monthly bill amount in USD |
| `TotalCharges` | Float | Total charges over the customer's tenure |
| `NumProducts` | Integer | Number of products subscribed (1–4) |
| `ContractLength` | Integer | Contract length in months (12, 24, or 36) |
| `SupportCalls` | Integer | Number of support calls made (0–10) |
| `SatisfactionScore` | Integer | Customer satisfaction score (1–5) |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| **`Churn`** | **Target** | **Yes / No — whether the customer churned** |

---

## Data Quality Notes

### Missing Values (~7% sparsity)
The following columns contain approximately 7% missing values to simulate real-world data:
- `Age`
- `MonthlyCharges`
- `TotalCharges`
- `SatisfactionScore`
- `InternetService`
- `PaymentMethod`

### Outliers
- `MonthlyCharges`: ~10 extreme values (180–250 USD) beyond the normal range
- `TotalCharges`: ~10 extreme values (8,000–15,000 USD) representing long-tenure high-spenders

---

## How to Use with the Dashboard

1. Launch the dashboard: `streamlit run app.py`
2. In the **Upload Data** section, upload either:
   - `sample_data.csv` (CSV upload)
   - `sample_data.xlsx` (Excel upload)
3. Set **Target Column** to `Churn`
4. Optionally drop `CustomerID` (non-informative ID column)
5. Run model training and compare results across all 7 classifiers

### Recommended Settings
- **Test split**: 20–30%
- **Target column**: `Churn`
- **Drop column**: `CustomerID` (identifier, not a feature)

---

## Feature Correlations with Churn

Key patterns baked into the data:
- **Month-to-month** contracts → higher churn risk
- **Low satisfaction score** (1–2) → higher churn risk
- **Many support calls** (>3) → higher churn risk
- **Long tenure** (>36 months) → lower churn risk
- **Senior citizens** → slightly higher churn risk
