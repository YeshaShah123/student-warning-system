import os

import pandas as pd
from sqlalchemy import create_engine, text


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DB_PATH = os.path.join(BASE_DIR, "data", "students.db")


def get_engine(db_path=DB_PATH):
    return create_engine(f"sqlite:///{db_path}")


def q1_pass_rate_by_studytime(engine=None):
    engine = engine or get_engine()
    query = text(
        """
        SELECT studytime, COUNT(*) AS total_students,
        SUM(pass) AS total_passed,
        ROUND(AVG(pass)*100,1) AS pass_rate_pct
        FROM students GROUP BY studytime ORDER BY studytime ASC
        """
    )
    return pd.read_sql(query, engine)


def q2_high_risk_students(engine=None):
    engine = engine or get_engine()
    query = text(
        """
        SELECT absences, failures, studytime, Medu, pass
        FROM students WHERE absences > 10 AND failures >= 1
        ORDER BY absences DESC, failures DESC LIMIT 20
        """
    )
    return pd.read_sql(query, engine)


def q3_grade_by_parent_edu(engine=None):
    engine = engine or get_engine()
    query = text(
        """
        SELECT Medu AS mother_edu_level,
        ROUND(AVG(risk_index),2) AS avg_risk_index,
        ROUND(AVG(pass)*100,1) AS pass_rate_pct,
        COUNT(*) AS student_count
        FROM students GROUP BY Medu ORDER BY Medu ASC
        """
    )
    return pd.read_sql(query, engine)


def q4_absence_impact(engine=None):
    engine = engine or get_engine()
    query = text(
        """
        SELECT CASE WHEN absences=0 THEN '0-None'
        WHEN absences BETWEEN 1 AND 5 THEN '1-5-Low'
        WHEN absences BETWEEN 6 AND 15 THEN '6-15-Medium'
        ELSE '15+-High' END AS absence_bucket,
        COUNT(*) AS students,
        ROUND(AVG(pass)*100,1) AS pass_rate_pct,
        ROUND(AVG(risk_index),2) AS avg_risk_index
        FROM students GROUP BY absence_bucket
        ORDER BY pass_rate_pct DESC
        """
    )
    return pd.read_sql(query, engine)


def q5_student_ranking(engine=None):
    engine = engine or get_engine()
    query = text(
        """
        SELECT studytime, absences, failures, risk_index, pass,
        RANK() OVER (PARTITION BY studytime
                     ORDER BY risk_index DESC) AS risk_rank_in_group
        FROM students ORDER BY studytime, risk_rank_in_group LIMIT 30
        """
    )
    return pd.read_sql(query, engine)
