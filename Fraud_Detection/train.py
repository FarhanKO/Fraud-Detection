"""
Fraud Sentinel — training entry point.

This is the ONLY place in the `Fraud Detection/` codebase (ignoring the
notebook) that reads the raw dataset. Every src/ module is path-agnostic —
they take `path` as a parameter and never hardcode one — so this file is
where a path belongs, not inside src/.

Run from inside the `Fraud Detection/` folder:
    python train.py
    python train.py --data /some/other/location/creditcard.csv

By default it looks for data/creditcard.csv (see data/README.md for how to
get that file — it is intentionally not committed to the repo).
"""

import argparse
from pathlib import Path

from src.preprocessing import load_data, temporal_train_test_split, build_preprocessing_pipeline
from src.feature_engineering import get_top_fraud_features
from src.models import get_supervised_model_grids, train_supervised_models, get_unsupervised_models, calibrate_classifier
from src.evaluation import evaluate_supervised_models
from src.utils import save_artifact, MODELS_DIR

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
RESULTS_DIR = BASE_DIR / "results"


def main(data_path: Path):
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    print(f"{len(df):,} rows loaded, {int(df['Class'].sum())} labeled fraud.")

    top_v_features = get_top_fraud_features(df)
    print(f"Top fraud-correlated features for interaction terms: {top_v_features}")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df)
    amount_test = X_test["Amount"].values

    print("Fitting preprocessing pipeline (feature engineering + imputer + scaler)...")
    processor = build_preprocessing_pipeline(top_v_features)
    X_train_processed = processor.fit_transform(X_train, y_train)
    X_test_processed = processor.transform(X_test)
    save_artifact(processor, "fraud_processor.pkl")

    print("Training Layer 1 (Isolation Forest)...")
    layer1 = get_unsupervised_models()["Isolation Forest"]
    layer1.fit(X_train_processed)
    save_artifact(layer1, "layer1_isolation_forest.pkl")

    print("Training Layer 2 (CatBoost) — grid search across 12 configs x 5-fold CV, this can take a while on CPU...")
    grids = {"CatBoost": get_supervised_model_grids()["CatBoost"]}
    results = train_supervised_models(grids, X_train_processed, y_train, cv=5)
    best_catboost = results["CatBoost"].best_estimator_
    save_artifact(best_catboost, "layer2_catboost.pkl")

    print("Calibrating CatBoost probabilities (isotonic)...")
    calibrated = calibrate_classifier(best_catboost, X_train_processed, y_train)
    save_artifact(calibrated, "layer2_calibrated_catboost.pkl")

    print("Evaluating on the chronological hold-out set...")
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_df = evaluate_supervised_models(
        {"CatBoost (calibrated)": calibrated}, X_test_processed, y_test, amount_test
    )
    metrics_df.to_csv(RESULTS_DIR / "metrics_calibrated.csv")
    print(metrics_df)

    print(f"\nDone. Artifacts saved to {MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Fraud Sentinel dual-layer cascade.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the raw creditcard.csv")
    args = parser.parse_args()
    main(args.data)
