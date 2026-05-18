import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sqlalchemy import create_engine, text


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.append(SRC_DIR)

from database import (  # noqa: E402
    q1_pass_rate_by_studytime,
    q2_high_risk_students,
    q3_grade_by_parent_edu,
    q4_absence_impact,
    q5_student_ranking,
)


st.set_page_config(page_title="Student Early Warning System", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(SRC_DIR, "model.pkl"))
    feature_cols = joblib.load(os.path.join(SRC_DIR, "feature_cols.pkl"))
    explainer_path = os.path.join(SRC_DIR, "explainer.pkl")
    explainer = joblib.load(explainer_path) if os.path.exists(explainer_path) else shap.TreeExplainer(model)
    return model, feature_cols, explainer


@st.cache_data
def load_students():
    engine = create_engine(f"sqlite:///{os.path.join(DATA_DIR, 'students.db')}")
    return pd.read_sql(text("SELECT * FROM students"), engine)


@st.cache_data
def load_metrics():
    return pd.read_csv(os.path.join(DATA_DIR, "model_metrics.csv"))


@st.cache_data
def load_sql_results():
    engine = create_engine(f"sqlite:///{os.path.join(DATA_DIR, 'students.db')}")
    return {
        "Pass Rate by Study Time": q1_pass_rate_by_studytime(engine),
        "High Risk Students": q2_high_risk_students(engine),
        "Risk by Parent Education": q3_grade_by_parent_edu(engine),
        "Absence Impact": q4_absence_impact(engine),
        "Student Risk Ranking": q5_student_ranking(engine),
    }


def class_one_values(values):
    if isinstance(values, list):
        return values[1]
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def get_top_reasons(student_row, explainer, n=3):
    row = student_row.to_frame().T if isinstance(student_row, pd.Series) else student_row
    values = class_one_values(explainer.shap_values(row))[0]
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


def build_input_row(feature_cols, studytime, absences, failures, medu, internet, higher):
    row = pd.DataFrame([{col: 0.0 for col in feature_cols}])
    defaults = {
        "school": 1,
        "sex": 1,
        "age": 17,
        "address": 1,
        "famsize": 1,
        "Pstatus": 1,
        "Medu": medu,
        "Fedu": 2,
        "traveltime": 1,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": 0,
        "famsup": 1,
        "paid": 0,
        "activities": 1,
        "nursery": 1,
        "higher": 1 if higher == "Yes" else 0,
        "internet": 1 if internet == "Yes" else 0,
        "romantic": 0,
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 2,
        "health": 3,
        "absences": absences,
    }
    for col, value in defaults.items():
        if col in row.columns:
            row.loc[0, col] = value

    row.loc[0, "parent_edu_total"] = row.loc[0, "Medu"] + row.loc[0, "Fedu"]
    row.loc[0, "social_score"] = row.loc[0, "goout"] + row.loc[0, "freetime"]
    row.loc[0, "has_support"] = int((row.loc[0, "schoolsup"] == 1) or (row.loc[0, "famsup"] == 1))
    row.loc[0, "risk_index"] = failures * 3 + absences / 10 + (4 - studytime)
    return row[feature_cols]


model, feature_cols, explainer = load_artifacts()
students_df = load_students()
metrics_df = load_metrics()
tuned = metrics_df.loc[metrics_df["Model"].eq("Tuned RF")].iloc[0]

with st.sidebar:
    page = st.radio("Navigation", ["Risk Predictor", "SQL Analytics", "Model Comparison"])
    st.divider()
    st.metric("Total students", "395")
    st.metric("Pass rate", "67.1%")
    st.metric("Model accuracy", f"{tuned['Accuracy'] * 100:.1f}%")
    st.divider()
    st.write("Model info")
    st.write("Type: Random Forest")
    st.write(f"Features: {len(feature_cols)}")
    st.write(f"Fail recall: {tuned['Fail Recall'] * 100:.1f}%")


if page == "Risk Predictor":
    st.title("Student Early Warning System")
    left, right = st.columns([1, 2])
    with left:
        studytime = st.slider("studytime", 1, 4, 2)
        absences = st.slider("absences", 0, 40, 6)
        failures = st.slider("failures", 0, 3, 0)
        medu = st.slider("Medu", 0, 4, 2)
        internet = st.selectbox("internet", ["Yes", "No"])
        higher = st.selectbox("higher", ["Yes", "No"])
        predict = st.button("Predict", type="primary")

    with right:
        if predict:
            row = build_input_row(feature_cols, studytime, absences, failures, medu, internet, higher)
            pass_probability = model.predict_proba(row)[0, 1]
            failure_probability = 1 - pass_probability
            high_risk = failure_probability >= 0.5
            badge_color = "#b42318" if high_risk else "#027a48"
            badge_text = "HIGH RISK" if high_risk else "LOW RISK"
            st.markdown(
                f"<div style='display:inline-block;background:{badge_color};color:white;"
                "padding:10px 16px;border-radius:6px;font-weight:700;'>"
                f"{badge_text}</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(failure_probability), text=f"Failure probability: {failure_probability * 100:.1f}%")
            st.subheader("Top SHAP reasons")
            for reason in get_top_reasons(row.iloc[0], explainer, n=3):
                st.write(f"- {reason}")
        else:
            st.info("Set student attributes and click Predict.")

elif page == "SQL Analytics":
    st.title("SQL Analytics")
    for title, df in load_sql_results().items():
        st.subheader(title)
        st.dataframe(df, use_container_width=True)

else:
    st.title("Model Comparison")
    st.image(os.path.join(DATA_DIR, "model_comparison.png"), use_container_width=True)
    display_metrics = metrics_df[metrics_df["Model"].isin(["LR", "RF", "XGB"])].copy()
    for col in ["Accuracy", "ROC-AUC", "Fail Recall"]:
        display_metrics[col] = (display_metrics[col] * 100).map("{:.1f}%".format)
    st.dataframe(display_metrics.rename(columns={"ROC-AUC": "ROC-AUC"}), use_container_width=True)
    st.info(
        "Random Forest was selected because it handled mixed engineered features well, supported "
        "class_weight='balanced' for the fail/pass imbalance, and produced interpretable tree-based "
        "SHAP explanations for the warning dashboard."
    )
