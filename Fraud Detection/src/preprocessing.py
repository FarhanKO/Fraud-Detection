import pandas as pd
import numpy as np

def remove_missing_target(df):
    return df.dropna(subset=["Class"])


def remove_highly_correlated_features(df, threshold=0.85):

    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]

    df_filtered = df.drop(columns=to_drop)
    return df_filtered, to_drop

def preprocess(df):
    df = remove_missing_target(df)
    df, dropped = remove_highly_correlated_features(df)
    print(f"Dropped Features: {dropped}" if dropped else "No Features Dropped")
    return df
