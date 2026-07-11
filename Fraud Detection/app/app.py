import sys
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import RobustScaler

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # lets `import src.xxx` resolve regardless of cwd
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))   # lets `from assets... import ...` resolve
 
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
 
from assets.sample_transactions import MOCK_LEGIT, MOCK_FRAUD, RAW_COLS  # noqa: E402


top_v_features = ["V17", "V14", "V12", "V10", "V16"]  # from the notebook's correlation ranking
 
 
def engineer_fraud_features(data):
    df_feat = data.copy()
    df_feat["Hour"] = (df_feat["Time"] // 3600) % 24
    df_feat["Is_Night"] = df_feat["Hour"].apply(lambda x: 1 if 0 <= x <= 5 else 0)
    df_feat["Log_Amount"] = np.log1p(df_feat["Amount"])

    scaler = RobustScaler()
    df_feat["Scaled_Amount"] = scaler.fit_transform(df_feat[["Amount"]])
    for v in top_v_features:
        df_feat[f"{v}_x_Amount"] = df_feat[v] * df_feat["Log_Amount"]
    return df_feat.drop(columns=["Time", "Amount"])

st.set_page_config(
    page_title="Fraud Sentinel | Dual-Layer Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css_path = APP_DIR / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained artifacts…")
def load_models(use_autoencoder: bool):
    processor = joblib.load(MODELS_DIR / "fraud_processor.pkl")
    layer2 = joblib.load(MODELS_DIR / "layer2_calibrated_catboost.pkl")
 
    if use_autoencoder:
        try:
            layer1 = joblib.load(MODELS_DIR / "layer1_autoencoder.pkl")
            engine_name = "Deep Autoencoder (PyOD)"
        except Exception as e:  # noqa: BLE001 — deliberately broad, this is a soft fallback
            st.warning(
                f"Couldn't load the Autoencoder Layer 1 model ({type(e).__name__}: {e}). "
                "This almost always means `pyod`/`torch` aren't installed in this "
                "environment — they're intentionally left out of requirements.txt to "
                "keep cloud deployments lightweight. Falling back to Isolation Forest."
            )

            layer1 = joblib.load(MODELS_DIR / "layer1_isolation_forest.pkl")
            engine_name = "Isolation Forest (fallback)"
    else:
        layer1 = joblib.load(MODELS_DIR / "layer1_isolation_forest.pkl")
        engine_name = "Isolation Forest"
 
    return processor, layer1, layer2, engine_name

def layer1_is_anomaly(model, processed) -> bool:
    pred = model.predict(processed)
    if "pyod" in type(model).__module__:
        return bool(pred[0] == 1)
    return bool(pred[0] == -1)
 
 
def layer1_score(model, processed) -> float:
    """Higher = more anomalous, regardless of which engine is active."""
    if "pyod" in type(model).__module__:
        return float(model.decision_function(processed)[0])
    return float(-model.score_samples(processed)[0])


def cascade_predict_single(row_df, processor, layer1_model, layer2_model):
    tx_amount = float(row_df["Amount"].values[0])
    processed = processor.transform(row_df[RAW_COLS])
 
    if layer1_is_anomaly(layer1_model, processed):
        return {
            "status": "BLOCKED_LAYER1",
            "label": "🚨 Blocked by Layer 1",
            "reason": "Flagged as a structural anomaly relative to legitimate transaction history",
            "action": "Route to fraud-forensics review (possible zero-day pattern)",
            "amount": tx_amount,
            "anomaly_score": layer1_score(layer1_model, processed),
            "probability": None,
        }
 
    probability = float(layer2_model.predict_proba(processed)[0, 1])
    threshold = 0.25 if tx_amount > 1000 else 0.50
 
    if probability >= threshold:
        return {
            "status": "REJECTED_LAYER2",
            "label": "❌ Rejected by Layer 2",
            "reason": "Matches historical fraud behavior profile",
            "action": "Auto-decline transaction and flag account for review",
            "amount": tx_amount,
            "anomaly_score": layer1_score(layer1_model, processed),
            "probability": probability,
            "threshold": threshold,
        }
 
    return {
        "status": "APPROVED",
        "label": "✅ Approved",
        "reason": "Passed both structural and pattern-matching checks",
        "action": "Authorize secure funds clearance",
        "amount": tx_amount,
        "anomaly_score": layer1_score(layer1_model, processed),
        "probability": probability,
        "threshold": threshold,
    }
 
 
def cascade_predict_batch(df_in, processor, layer1_model, layer2_model):
    model_input = df_in[RAW_COLS].copy()
    processed = processor.transform(model_input)
    amounts = model_input["Amount"].values
 
    l1_preds = layer1_model.predict(processed)
    if "pyod" in type(layer1_model).__module__:
        blocked = l1_preds == 1
        scores = layer1_model.decision_function(processed)
    else:
        blocked = l1_preds == -1
        scores = -layer1_model.score_samples(processed)
 
    probs = np.full(len(df_in), np.nan)
    remaining = ~blocked
    if remaining.any():
        probs[remaining] = layer2_model.predict_proba(processed[remaining])[:, 1]
 
    thresholds = np.where(amounts > 1000, 0.25, 0.50)
    rejected = remaining & (probs >= thresholds)
    approved = remaining & ~rejected
 
    decision = np.select(
        [blocked, rejected, approved],
        ["Blocked (Layer 1)", "Rejected (Layer 2)", "Approved"],
        default="Approved",
    )
 
    out = df_in.copy()
    out["Anomaly_Score"] = scores
    out["Fraud_Probability"] = probs
    out["Decision"] = decision
    return out

with st.sidebar:
    st.markdown("### 🛡️ Fraud Sentinel")
    st.caption("Dual-layer cascade — served from the real trained artifacts, not a synthetic demo.")
    st.divider()
 
    use_ae = st.checkbox(
        "Use Deep Autoencoder for Layer 1",
        value=False,
        help="Matches the notebook's original Layer 1 exactly, but requires pyod + torch "
             "(not in requirements.txt by default — heavy for cloud deployment). "
             "Isolation Forest is the recommended default.",
    )
 
    st.divider()
    st.markdown("**Training dataset**")
    st.caption("284,807 transactions · 492 confirmed fraud (0.172%) · European cardholders, Sept 2013")
 
    st.divider()
    st.markdown("[📓 View training notebook](https://github.com/FarhanKO/Fraud-Detection/blob/main/Fraud_Detection.ipynb)")
    st.markdown("[📄 Repo README](https://github.com/FarhanKO/Fraud-Detection)")
 
try:
    processor, layer1_model, layer2_model, engine_name = load_models(use_ae)
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    st.error(
        f"Couldn't find a required model file: {e}. Make sure `models/` contains "
        "fraud_processor.pkl, layer1_isolation_forest.pkl, and layer2_calibrated_catboost.pkl."
    )
