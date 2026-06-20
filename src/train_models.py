"""
train_models.py
---------------
Train Logistic Regression, Random Forest, and XGBoost classifiers.
Saves ROC curves, confusion matrices, feature importance, and metrics.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score, f1_score, classification_report
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from preprocess import preprocess

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots",   exist_ok=True)


def plot_confusion_matrix(cm, name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"results/plots/cm_{fname}.png", dpi=150)
    plt.close()


def main():
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        scale = (y_train == 0).sum() / (y_train == 1).sum()
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale, use_label_encoder=False,
            eval_metric="logloss", random_state=42
        )

    all_results = {}
    roc_data    = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred)
        print(f"{name:22s} | AUC={auc:.4f}  F1={f1:.4f}")

        plot_confusion_matrix(confusion_matrix(y_test, y_pred), name)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)

        all_results[name] = {
            "roc_auc": auc, "f1": f1,
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
        fname = name.lower().replace(" ", "_")
        with open(f"results/metrics/{fname}.json", "w") as f:
            json.dump(all_results[name], f, indent=2)

    # Combined ROC curve
    colors = ["#2563EB", "#16a34a", "#f97316"]
    plt.figure(figsize=(8, 6))
    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – All Models")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig("results/plots/roc_curves.png", dpi=150)
    plt.close()
    print("Saved ROC curves.")

    # Feature importance (RF)
    rf = models["Random Forest"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(idx)), importances[idx], color="#2563EB")
    plt.xticks(range(len(idx)), [feature_cols[i] for i in idx], rotation=45, ha="right")
    plt.title("Feature Importances – Random Forest")
    plt.tight_layout()
    plt.savefig("results/plots/feature_importance.png", dpi=150)
    plt.close()
    print("Saved feature importance plot.")


if __name__ == "__main__":
    main()
