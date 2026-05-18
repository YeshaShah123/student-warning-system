import os

import pandas as pd
from sqlalchemy import create_engine


YES_NO_COLS = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]


def extract(path):
    df = pd.read_csv(path, sep=";")
    print(f"Loaded raw data shape: {df.shape}")
    return df


def transform(df):
    df = df.copy()
    df["pass"] = (df["G3"] >= 10).astype(int)

    for col in YES_NO_COLS:
        df[col] = df[col].map({"yes": 1, "no": 0}).astype(int)

    df["school"] = (df["school"] == "GP").astype(int)
    df["address"] = (df["address"] == "U").astype(int)
    df["sex"] = (df["sex"] == "F").astype(int)
    df["famsize"] = (df["famsize"] == "GT3").astype(int)
    df["Pstatus"] = (df["Pstatus"] == "T").astype(int)

    df = pd.get_dummies(
        df,
        columns=["Mjob", "Fjob", "reason", "guardian"],
        drop_first=True,
        dtype=int,
    )

    df["parent_edu_total"] = df["Medu"] + df["Fedu"]
    df["social_score"] = df["goout"] + df["freetime"]
    df["has_support"] = ((df["schoolsup"] == 1) | (df["famsup"] == 1)).astype(int)
    df["risk_index"] = (
        df["failures"] * 3
        + df["absences"] / 10
        + (4 - df["studytime"])
    )
    print(f"Transformed clean data shape: {df.shape}")
    return df


def load(df, engine, table_name="students"):
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into table '{table_name}'")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    csv_path = os.path.join(base_dir, "data", "student-mat.csv")
    db_path = os.path.join(base_dir, "data", "students.db")

    engine = create_engine(f"sqlite:///{db_path}")
    raw_df = extract(csv_path)
    clean_df = transform(raw_df)
    load(clean_df, engine)
    print("ETL complete.")
