import os
import joblib
import numpy as np
import pandas as pd

RANDOM_STATE = 42
ALPHA = 1.0   # $ weight for missed-fraud amount in the banking risk cost function
BETA = 50.0   # $ weight per false alarm in the banking risk cost function

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index between two continuous distributions (train vs test)."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_percents = np.clip(expected_percents, 1e-4, None)
    actual_percents = np.clip(actual_percents, 1e-4, None)

    return float(np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents)))


def drift_report(X_train, X_test, columns=None):
    """Rank features by PSI drift severity between the train and test windows."""
    columns = columns or X_train.columns
    rows = [{"Feature": col, "PSI": calculate_psi(X_train[col].values, X_test[col].values)} for col in columns]
    df_psi = pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)
    df_psi["Drift Severity"] = pd.cut(
        df_psi["PSI"], bins=[-1, 0.1, 0.2, np.inf], labels=["Stable", "Monitor", "High Drift"]
    )
    return df_psi


def save_artifact(obj, filename, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    joblib.dump(obj, path)
    return path


def load_artifact(filename, models_dir=MODELS_DIR):
    return joblib.load(os.path.join(models_dir, filename))

