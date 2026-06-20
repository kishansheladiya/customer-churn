"""
preprocess.py
-------------
Load customer CSV, encode categoricals, scale numerics,
apply SMOTE to handle class imbalance, return train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn not installed — SMOTE skipped.")

CAT_COLS = ["contract", "internet_service", "payment_method"]
NUM_COLS = ["tenure", "age", "num_products", "monthly_charges", "total_charges"]
BIN_COLS = ["has_partner", "has_dependents", "online_security", "tech_support", "paperless_billing"]


def load_data(path: str = "data/sample/customer_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess(path: str = "data/sample/customer_data.csv", apply_smote: bool = True):
    df = load_data(path)

    # Encode categoricals
    df_enc = df.copy()
    for col in CAT_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])

    feature_cols = NUM_COLS + BIN_COLS + CAT_COLS
    X = df_enc[feature_cols].values
    y = df_enc["churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Churn rate  train: {y_train.mean():.1%}  test: {y_test.mean():.1%}")

    if apply_smote and HAS_SMOTE:
        sm      = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"After SMOTE  train: {X_train.shape} | churn rate: {y_train.mean():.1%}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def evaluate(name, y_test, y_pred, y_prob):
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    print(f"  ROC-AUC : {auc:.4f}  |  F1: {f1:.4f}")
    return {"roc_auc": auc, "f1": f1,
            "report": classification_report(y_test, y_pred, output_dict=True)}
