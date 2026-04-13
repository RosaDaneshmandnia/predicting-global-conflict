import pandas as pd
import numpy as np


def add_worldbank_engineered_features(df):
    """
    Engineered features from World Bank indicators.
    Based on analysis in worldbank_analyse.ipynb.

    Conceptual model:
        Structural pressure: income_inequality
        Mobilization capacity: youth_unemployment
        Shock trigger: inflation
        Institutional buffer: income_level_code

    All inputs are (t-1) columns (no leakage)
    """

    df = df.copy()

    ineq   = df["income_inequality (t-1)"]
    youth  = df["youth_unemployment (t-1)"]
    infl   = df["inflation (t-1)"]
    income = df["income_level_code (t-1)"]

    # inequality × youth unemployment , chronic grievance × mobilizable population
    df["structural_vulnerability"] = ineq * youth

    # top 10% inflation = shock condition (nonlinear and binary)
    threshold = infl.quantile(0.9)
    df["inflation_shock"] = (infl > threshold).astype(int)

    #  Shock × Structural Vulnerability 
    df["shock_vulnerability"] = df["inflation_shock"] * df["structural_vulnerability"]

    # income level as institutional buffer (reduces instability)
    # +3 shift avoids division issues (income_level_code is -2,-1,1,2)
    df["capacity_adjusted_risk"] = df["structural_vulnerability"] / (income + 3)

    # Inflation Change 
    df["inflation_change"] = (
        df.groupby(level="matched_admin1_id")["inflation (t-1)"].diff()
    )

    # High Inequality flag (threshold effect)
    df["high_inequality_flag"] = (
        ineq > ineq.quantile(0.75)
    ).astype(int)

    # Inflation Shock in Poor Countries 
    df["inflation_shock_poor"] = (
        df["inflation_shock"] * (income < 0).astype(int)
    )

    # Conflict Momentum
    df["conflict_momentum"] = (
        df["Battles (t-1)"] +
        df["Violence against civilians (t-1)"] +
        df["Battles_neighbours (t-1)"] +
        df["Violence against civilians_neighbours (t-1)"]
    )

    # Violence Escalation Ratio (violent vs non-violent activity ratio)
    df["violence_escalation"] = (
        df["Violence against civilians (t-1)"] + df["Battles (t-1)"]
    ) / (df["Protests (t-1)"] + df["Riots (t-1)"] + 1)

    #  Macro × Conflict Pressure
    df["macro_conflict_pressure"] = (
        df["structural_vulnerability"] * df["conflict_momentum"]
    )

    return df


def get_wbd_feature_names():

    return [
        "structural_vulnerability",
        "inflation_shock",
        "shock_vulnerability",
        "capacity_adjusted_risk",
        "inflation_change",
        "high_inequality_flag",
        "inflation_shock_poor",
        "conflict_momentum",
        "violence_escalation",
        "macro_conflict_pressure",
    ]