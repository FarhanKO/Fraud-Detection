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
