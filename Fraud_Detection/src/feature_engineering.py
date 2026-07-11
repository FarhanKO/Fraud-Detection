import numpy as np
from functools import partial
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def get_top_fraud_features(df, target_col="Class", n=5, prefix="V"):
    """Return the n V-features most correlated (absolute) with the target."""
    corr = df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
    return [c for c in corr.index if c.startswith(prefix)][:n]


def engineer_fraud_features(data, top_v_features, time_col="Time", amount_col="Amount"):
    """Derive temporal, amount, and interaction features from raw transaction rows."""
    df_feat = data.copy()

    # --- Temporal features ---
    df_feat["Hour"] = (df_feat[time_col] // 3600) % 24
    df_feat["Is_Night"] = df_feat["Hour"].apply(lambda x: 1 if 0 <= x <= 5 else 0)

    # --- Amount features ---
    df_feat["Log_Amount"] = np.log1p(df_feat[amount_col])
    scaler = RobustScaler()
    df_feat["Scaled_Amount"] = scaler.fit_transform(df_feat[[amount_col]])

    # --- Interaction features ---
    for v in top_v_features:
        df_feat[f"{v}_x_Amount"] = df_feat[v] * df_feat["Log_Amount"]

    return df_feat.drop(columns=[time_col, amount_col])


def build_feature_engineering_transformer(top_v_features):
    """Wrap engineer_fraud_features as a sklearn FunctionTransformer with top_v_features baked in."""
    fn = partial(engineer_fraud_features, top_v_features=top_v_features)
    return FunctionTransformer(fn)
