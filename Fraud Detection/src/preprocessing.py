import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.feature_engineering import build_feature_engineering_transformer

def load_data(path, target_col="Class"):
    df = pd.read_csv(path)
    return df.dropna(subset=[target_col])


def drop_high_correlation_features(df, threshold=0.85, exclude=("Class",)):
    """Drop one feature from any pair whose absolute correlation exceeds threshold."""
    corr_matrix = df.drop(columns=list(exclude), errors="ignore").corr(numeric_only=True).abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    return df.drop(columns=to_drop), to_drop


def temporal_train_test_split(df, target_col="Class", time_col="Time", test_size=0.2):
    """Chronological split — prevents 'future' transactions leaking into training."""
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    X_sorted = df_sorted.drop(columns=[target_col])
    y_sorted = df_sorted[target_col]

    split_index = int(len(df_sorted) * (1 - test_size))
    X_train, X_test = X_sorted.iloc[:split_index], X_sorted.iloc[split_index:]
    y_train, y_test = y_sorted.iloc[:split_index], y_sorted.iloc[split_index:]
    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(top_v_features):
    """engineering -> median imputation -> robust scaling."""
    return Pipeline(steps=[
        ("engineering", build_feature_engineering_transformer(top_v_features)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
