from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
 
from src.utils import RANDOM_STATE
 
 
def get_supervised_model_grids():
    """Return {name: (pipeline, param_grid)} for every supervised model, CPU-only."""
    grids = {}
 
    grids["Logistic Regression"] = (
        ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]),
        {"classifier__C": [0.01, 0.1, 1, 10, 100]},
    )
 
    grids["KNN"] = (
        ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)),
            ("classifier", KNeighborsClassifier(n_jobs=-1)),
        ]),
        {
            "classifier__n_neighbors": [3, 5, 7],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"],
        },
    )
 
    grids["Decision Tree"] = (
        ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)),
            ("classifier", XGBClassifier(n_estimators=1, random_state=RANDOM_STATE, tree_method="hist")),
        ]),
        {"classifier__max_depth": [5, 10, 15, 20], "classifier__learning_rate": [1.0]},
    )
 
    grids["Random Forest"] = (
        ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 15, 20],
            "classifier__max_features": ["sqrt", "log2"],
        },
    )
 
    grids["Naive Bayes"] = (
        ImbPipeline([
            ("minmax", MinMaxScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)),
            ("classifier", ComplementNB()),
        ]),
        {"classifier__alpha": [0.1, 0.5, 1.0, 2.0]},
    )
 
    grids["XGBoost"] = (
        ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)),
            ("classifier", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", tree_method="hist")),
        ]),
        {
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__n_estimators": [100, 200, 300],
        },
    )
 
    grids["CatBoost"] = (
        ImbPipeline([
            ("classifier", CatBoostClassifier(
                task_type="CPU", auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0
            )),
        ]),
        {
            "classifier__depth": [4, 6, 8],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__iterations": [200, 500],
        },
    )
 
    return grids
 
 
def train_supervised_models(model_grids, X_train, y_train, cv=5, scoring="average_precision", n_jobs=-1):
    """Fit GridSearchCV for every entry in model_grids. Returns {name: fitted_grid}."""
    results = {}
    for name, (pipeline, params) in model_grids.items():
        grid = GridSearchCV(pipeline, params, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid.fit(X_train, y_train)
        results[name] = grid
    return results
 
 
def get_unsupervised_models(contamination=0.002):
    """Return dict of unsupervised anomaly detectors. Requires: pip install pyod"""
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.auto_encoder import AutoEncoder
 
    return {
        "Isolation Forest": IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True),
        "ECOD": ECOD(contamination=contamination),
        "COPOD": COPOD(contamination=contamination),
        "Autoencoder": AutoEncoder(contamination=contamination),
    }
 
 
def calibrate_classifier(best_pipeline, X_train, y_train, method="isotonic", cv=3):
    """Wrap a fitted pipeline/estimator with probability calibration."""
    calibrated = CalibratedClassifierCV(estimator=best_pipeline, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated
 
 
def fraud_cascade_predict(
    raw_transaction_row, ae_model, cb_model, processor, amount_col="Amount",
    high_value_threshold=5000, high_value_trigger=4.0, default_trigger=5.0,
    high_amount_supervised_threshold=1000, high_amount_prob_threshold=0.25,
    default_prob_threshold=0.5,
):

    
    """
    Layer 1 (Autoencoder): blocks structural / zero-day anomalies before pattern matching.
    Layer 2 (calibrated classifier, e.g. CatBoost): historical pattern-based fraud probability.
    Both thresholds tighten for high-value transactions since a missed high-value
    fraud is far costlier than a false alarm.
    """
    tx_amount = raw_transaction_row[amount_col].values[0]
    processed = processor.transform(raw_transaction_row)
 
    anomaly_score = ae_model.decision_function(processed)[0]
    unsupervised_trigger = high_value_trigger if tx_amount > high_value_threshold else default_trigger
    if anomaly_score > unsupervised_trigger:
        return {
            "status": "BLOCKED_LAYER_1",
            "reason": "Extreme unsupervised structural anomaly",
            "amount": tx_amount,
            "anomaly_score": float(anomaly_score),
        }
 
    supervised_probability = cb_model.predict_proba(processed)[0, 1]
    supervised_threshold = (
        high_amount_prob_threshold if tx_amount > high_amount_supervised_threshold else default_prob_threshold
    )
    if supervised_probability >= supervised_threshold:
        return {
            "status": "REJECTED_LAYER_2",
            "reason": "Matches historical fraud pattern",
            "amount": tx_amount,
            "fraud_probability": float(supervised_probability),
        }
 
    return {
        "status": "APPROVED",
        "reason": "Passed both structural and pattern-matching checks",
        "amount": tx_amount,
        "fraud_probability": float(supervised_probability),
    }
