"""
generate_sample_data.py
-----------------------
Generates synthetic telecom customer churn data with realistic
feature correlations and ~26% churn rate (industry benchmark).
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 5000

# Demographics
tenure         = np.random.randint(1, 73, N)                          # months 1–72
age            = np.random.randint(18, 75, N)
num_products   = np.random.choice([1, 2, 3, 4], N, p=[0.5, 0.3, 0.15, 0.05])
has_partner    = np.random.choice([0, 1], N)
has_dependents = np.random.choice([0, 1], N, p=[0.7, 0.3])

# Service features
contract       = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20])
internet       = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20])
online_security= np.random.choice([0, 1], N, p=[0.6, 0.4])
tech_support   = np.random.choice([0, 1], N, p=[0.6, 0.4])
paperless      = np.random.choice([0, 1], N, p=[0.4, 0.6])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N,
    p=[0.35, 0.22, 0.22, 0.21]
)

# Financials
monthly_charges = (
    30
    + 20 * (internet == "Fiber optic")
    + 10 * (internet == "DSL")
    + 5  * online_security
    + 5  * tech_support
    + np.random.normal(0, 5, N)
).clip(18, 120).round(2)

total_charges = (monthly_charges * tenure + np.random.normal(0, 50, N)).clip(18).round(2)

# Churn probability — logistic model
log_odds = (
    -2.5
    + 0.03  * monthly_charges
    - 0.05  * tenure
    + 1.2   * (contract == "Month-to-month")
    - 0.5   * (contract == "Two year")
    + 0.8   * (internet == "Fiber optic")
    - 0.4   * online_security
    - 0.3   * tech_support
    + 0.3   * paperless
    + 0.4   * (payment_method == "Electronic check")
    + np.random.normal(0, 0.5, N)
)
churn_prob = 1 / (1 + np.exp(-log_odds))
churn      = (np.random.rand(N) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id":      [f"C{str(i).zfill(5)}" for i in range(N)],
    "tenure":           tenure,
    "age":              age,
    "num_products":     num_products,
    "has_partner":      has_partner,
    "has_dependents":   has_dependents,
    "contract":         contract,
    "internet_service": internet,
    "online_security":  online_security,
    "tech_support":     tech_support,
    "paperless_billing":paperless,
    "payment_method":   payment_method,
    "monthly_charges":  monthly_charges,
    "total_charges":    total_charges,
    "churn":            churn,
})

os.makedirs("data/sample", exist_ok=True)
df.to_csv("data/sample/customer_data.csv", index=False)
print(f"Generated {N} customers")
print(f"Churn rate: {churn.mean():.1%}  ({churn.sum()} churned)")
