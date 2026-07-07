import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def create_hour_feature(df):
    df = df.copy()
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24
    return df

def create_is_night_feature(df):
    df = df.copy()
    if "Hour" not in df.columns:
        df = create_hour_feature(df)

    df["Is_Night"] = (
        (df["Hour"] >= 22) | (df["Hour"] <= 6)
    ).astype(int)
    return df

def create_log_amount(df):
    df = df.copy()
    if "Amount" in df.columns:
        df["Log_Amount"] = np.log1p(df["Amount"])

    return df

def scale_amount(df):
    df = df.copy()
    if "Amount" in df.columns:
        scaler = RobustScaler()
        df["Scaled_Amount"] = scaler.fit_transform(df[["Amount"]])

    return df

def engineer_features(df):
    df = create_hour_feature(df)
    df = create_is_night_feature(df)
    df = create_log_amount(df)
    df = scale_amount(df)
    return df
