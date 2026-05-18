import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sqlalchemy import create_engine, text


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")


def _class_one_values(values):
    if isinstance(values, list):
        return values[1]
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def load_data():
    model = joblib.load(os.path.join(SRC_DIR, "model.pkl"))
    feature_cols = joblib.load(os.path.join(SRC_DIR, "feature_cols.pkl"))
    engine = create_engine(f"sqlite:///{os.path.join(DATA_DIR, 'students.db')}")
    df = pd.read_sql(text("SELECT * FROM students"), engine)
    X = df.drop(columns=["G1", "G2", "G3", "pass"]).select_dtypes(include=[np.number])
    X = X[feature_cols]
    y = df["pass"]
    return model, feature_cols, X, y


def get_top_reasons(student_row, explainer, n=3):
    row = student_row.to_frame().T if isinstance(student_row, pd.Series) else student_row
    values = _class_one_values(explainer.shap_values(row))[0]
    reasons = []
    for idx in np.argsort(np.abs(values))[::-1][:n]:
        feature = row.columns[idx]
        value = row.iloc[0, idx]
        shap_value = values[idx]
        direction = "reduces" if shap_value > 0 else "increases"
        reasons.append(
            f"{feature} = {value} {direction} failure risk (impact: {shap_value:+.3f})"
        )
    return reasons


def main():
    model, feature_cols, X, y = load_data()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    class_one_values = _class_one_values(shap_values)

    shap.summary_plot(class_one_values, X, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "shap_global_importance.png"), dpi=180, bbox_inches="tight")
    plt.close()

    shap.summary_plot(class_one_values, X, plot_type="dot", max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "shap_dot_plot.png"), dpi=180, bbox_inches="tight")
    plt.close()

    high_risk_idx = X.loc[y == 0, "risk_index"].idxmax()
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    shap.force_plot(
        expected_value,
        class_one_values[X.index.get_loc(high_risk_idx), :],
        X.loc[high_risk_idx, :],
        matplotlib=True,
        show=False,
    )
    plt.savefig(os.path.join(DATA_DIR, "shap_force_plot.png"), dpi=180, bbox_inches="tight")
    plt.close()

    for idx in X.sample(5, random_state=42).index:
        print(f"\nStudent {idx}")
        for reason in get_top_reasons(X.loc[idx], explainer, n=3):
            print(f"- {reason}")

    joblib.dump(explainer, os.path.join(SRC_DIR, "explainer.pkl"))
    print("\nSaved explainer.pkl.")


if __name__ == "__main__":
    main()
