from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_logistic_regression():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


def get_gradient_boosting():
    return GradientBoostingClassifier(
        random_state=42
    )


def get_xgboost():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric="logloss"
    )


def get_lightgbm():
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )


def get_catboost():
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        verbose=False,
        random_state=42
    )


def get_all_models():
    return {
        "Logistic Regression": get_logistic_regression(),
        "Random Forest": get_random_forest(),
        "Gradient Boosting": get_gradient_boosting(),
        "XGBoost": get_xgboost(),
        "LightGBM": get_lightgbm(),
        "CatBoost": get_catboost()
    }
