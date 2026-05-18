import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")


def load_model_data():
    engine = create_engine(f"sqlite:///{os.path.join(DATA_DIR, 'students.db')}")
    df = pd.read_sql(text("SELECT * FROM students"), engine)
    y = df["pass"]
    X = df.drop(columns=["G1", "G2", "G3", "pass"]).select_dtypes(include=[np.number])
    return X, y


def evaluate_model(name, model, X_test, y_test, use_proba=True):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if use_proba else y_pred
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"{name.lower().replace(' ', '_')}_confusion_matrix.png"), dpi=160)
    plt.close()

    report = classification_report(
        y_test, y_pred, target_names=["Fail", "Pass"], output_dict=True
    )
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_score),
        "Fail Recall": report["Fail"]["recall"],
    }


def save_model_comparison(metrics):
    metrics_df = pd.DataFrame(metrics)
    labels = metrics_df["Model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    acc = metrics_df["Accuracy"] * 100
    auc = metrics_df["ROC-AUC"] * 100
    bars1 = ax.bar(x - width / 2, acc, width, label="Accuracy %", color="#2f80ed")
    bars2 = ax.bar(x + width / 2, auc, width, label="ROC-AUC %", color="#27ae60")
    ax.axhline(80, color="red", linestyle="--", linewidth=1.5, label="80% reference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison")
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "model_comparison.png"), dpi=180)
    plt.close()
    metrics_df.to_csv(os.path.join(DATA_DIR, "model_metrics.csv"), index=False)
    return metrics_df


def main():
    X, y = load_model_data()
    joblib.dump(list(X.columns), os.path.join(SRC_DIR, "feature_cols.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    xgb = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)

    lr.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    metrics = [
        evaluate_model("LR", lr, X_test_scaled, y_test),
        evaluate_model("RF", rf, X_test, y_test),
        evaluate_model("XGB", xgb, X_test, y_test),
    ]

    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(lr, X_test_scaled, y_test, name="LR")
    RocCurveDisplay.from_estimator(rf, X_test, y_test, name="RF")
    RocCurveDisplay.from_estimator(xgb, X_test, y_test, name="XGB")
    plt.title("ROC Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "roc_curves.png"), dpi=180)
    plt.close()

    metrics_df = save_model_comparison(metrics)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced"],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV recall score: {grid.best_score_:.3f}")

    best_model = grid.best_estimator_
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="recall")
    print(f"5-fold recall CV mean: {cv_scores.mean():.3f}")
    print(f"5-fold recall CV std: {cv_scores.std():.3f}")

    rf_default_pred = rf.predict(X_test)
    rf_default_score = rf.predict_proba(X_test)[:, 1]
    rf_tuned_pred = best_model.predict(X_test)
    rf_tuned_score = best_model.predict_proba(X_test)[:, 1]

    print("\nDefault vs Tuned RF")
    print(f"Default accuracy: {accuracy_score(y_test, rf_default_pred):.3f}")
    print(f"Default ROC-AUC: {roc_auc_score(y_test, rf_default_score):.3f}")
    print(f"Tuned accuracy: {accuracy_score(y_test, rf_tuned_pred):.3f}")
    print(f"Tuned ROC-AUC: {roc_auc_score(y_test, rf_tuned_score):.3f}")

    tuned_row = {
        "Model": "Tuned RF",
        "Accuracy": accuracy_score(y_test, rf_tuned_pred),
        "ROC-AUC": roc_auc_score(y_test, rf_tuned_score),
        "Fail Recall": classification_report(
            y_test, rf_tuned_pred, target_names=["Fail", "Pass"], output_dict=True
        )["Fail"]["recall"],
    }
    pd.concat([metrics_df, pd.DataFrame([tuned_row])], ignore_index=True).to_csv(
        os.path.join(DATA_DIR, "model_metrics.csv"), index=False
    )

    joblib.dump(best_model, os.path.join(SRC_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(SRC_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(SRC_DIR, "feature_cols.pkl"))
    print("\nSaved model.pkl, scaler.pkl, and feature_cols.pkl.")


if __name__ == "__main__":
    main()
