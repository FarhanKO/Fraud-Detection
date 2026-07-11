import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import RobustScaler

# ──────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────
# COMPATIBILITY SHIM — DO NOT REMOVE
# ──────────────────────────────────────────────────────────────────────────
# models/fraud_processor.pkl was pickled straight out of the training
# notebook, where `engineer_fraud_features` was a top-level function in the
# notebook's `__main__` namespace, closing over a global `top_v_features`.
# Python's pickle format stores plain functions *by reference* (module +
# name), never by bytecode — so unpickling this FunctionTransformer needs a
# function literally named `engineer_fraud_features` to exist in `__main__`
# (this script, since Streamlit runs it as the entry point) at load time.
# It's recreated verbatim below so `joblib.load(fraud_processor.pkl)`
# succeeds. Without this block the app fails with:
#   AttributeError: Can't get attribute 'engineer_fraud_features' on
#   <module '__main__' ...>
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


# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Sentinel | Dual-Layer Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css_path = APP_DIR / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained artifacts…")
def load_models(use_autoencoder: bool):
    processor = joblib.load(MODELS_DIR / "fraud_processor.pkl")
    layer2 = joblib.load(MODELS_DIR / "layer2_calibrated_catboost.pkl")

    if use_autoencoder:
        try:
            # Cheap pre-check before touching the pickle at all. PyOD's
            # AutoEncoder is a thin wrapper around a PyTorch model, and in
            # environments where `pyod` is installed but `torch` is NOT
            # (a real possibility — they're separate packages), unpickling
            # or using it doesn't raise a clean Python exception: it can
            # hard-crash the whole process (verified — no try/except can
            # catch that). Confirming both imports succeed first keeps this
            # a normal, catchable failure instead.
            import torch  # noqa: F401
            import pyod  # noqa: F401
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


# ──────────────────────────────────────────────────────────────────────────
# CASCADE SCORING
# Isolation Forest and PyOD's Autoencoder use different label/score
# conventions (sklearn: -1 = anomaly; PyOD: 1 = anomaly), so Layer 1 is
# dispatched by model type rather than reusing one hardcoded threshold.
# Layer 2's amount-aware thresholds (0.25 / 0.50) are the notebook's actual
# business thresholds and are reused as-is — they're a property of the
# classifier's calibrated probabilities, not of whichever Layer 1 engine
# is active.
# ──────────────────────────────────────────────────────────────────────────
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


def safe_image(path: Path, **kwargs):
    """st.image, but a missing/empty/corrupt file degrades to a caption
    instead of crashing the whole script run. (Caught in testing: an empty
    placeholder PNG committed to images/ took down the entire app with an
    uncaught PIL.UnidentifiedImageError — st.image has no built-in guard
    against that.)"""
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        st.image(str(path), **kwargs)
    except Exception:  # noqa: BLE001 — genuinely any failure here should degrade, not crash
        st.caption(f"⚠️ Couldn't render {path.name} — the file may be corrupted or empty.")


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


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="fs-hero">
        <h1>🛡️ Fraud Sentinel</h1>
        <p>A dual-layer fraud detection cascade — an unsupervised anomaly gate followed by a
        calibrated supervised classifier — served here from the actual serialized models
        trained and evaluated in the accompanying notebook.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📊 Overview", "📈 Model Performance", "🧠 Explainability",
    "⚡ Live Transaction Check", "📁 Batch Scoring", "ℹ️ How It Works",
])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Total Transactions", "Confirmed Fraud", "Fraud Rate", "Production Model PR-AUC"],
        ["284,807", "492", "0.172%", "0.804 (calibrated CatBoost)"],
    ):
        col.markdown(
            f"""<div class="fs-card"><div class="fs-metric-label">{label}</div>
            <div class="fs-metric-value">{value}</div></div>""",
            unsafe_allow_html=True,
        )

    safe_image(IMAGES_DIR / "dashboard.png", width='stretch')

    st.markdown("#### Class Imbalance")
    dist_df = pd.DataFrame({"Class": ["Legitimate", "Fraud"], "Count": [284315, 492]})
    fig = px.bar(
        dist_df, x="Class", y="Count", color="Class", text="Count", log_y=True,
        color_discrete_map={"Legitimate": "#3b82f6", "Fraud": "#ef4444"},
    )
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    st.plotly_chart(fig, width='stretch')
    st.caption("Log scale — fraud is 0.172% of all transactions, which is why accuracy alone is a meaningless metric here.")

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────
with tabs[1]:
    metrics_path = RESULTS_DIR / "metrics.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        st.markdown("#### Supervised Model Comparison")
        st.dataframe(
            df_metrics.style.background_gradient(cmap="Greens", subset=["PR_AUC", "ROC_AUC"])
            .background_gradient(cmap="Reds", subset=["Banking_Risk_Cost_USD"])
            .format({
                "Accuracy": "{:.4f}", "Precision_Fraud": "{:.4f}", "Recall_Fraud": "{:.4f}",
                "F1_Score_Fraud": "{:.4f}", "ROC_AUC": "{:.4f}", "PR_AUC": "{:.4f}",
                "Missed_Fraud_Amount_USD": "${:,.2f}", "Banking_Risk_Cost_USD": "${:,.2f}",
            }),
            width='stretch',
        )
    else:
        st.warning(f"results/metrics.csv not found at {metrics_path}.")

    st.markdown("#### Production Model — Isotonic-Calibrated CatBoost")
    p1, p2, p3, p4, p5 = st.columns(5)
    for col, label, value in zip(
        [p1, p2, p3, p4, p5],
        ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
        ["0.9483", "0.7333", "0.8271", "0.9793", "0.8042"],
    ):
        col.markdown(
            f"""<div class="fs-card"><div class="fs-metric-label">{label}</div>
            <div class="fs-metric-value">{value}</div></div>""",
            unsafe_allow_html=True,
        )
    st.caption("Isotonic calibration trades a little recall for substantially better precision and much more reliable probability estimates — see Explainability for the calibration curve.")

    for fname, caption in [
        ("roc_pr_curve.png", "ROC and Precision-Recall curves across every supervised model."),
        ("model_comparison.png", "Head-to-head comparison across accuracy, precision, recall, F1, and AUC."),
        ("threshold_analysis.png", "Confusion matrix and business-impact sensitivity across decision thresholds."),
    ]:
        safe_image(IMAGES_DIR / fname, width='stretch', caption=caption)

# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — EXPLAINABILITY
# ──────────────────────────────────────────────────────────────────────────
with tabs[2]:
    fi_path = RESULTS_DIR / "feature_importance.csv"
    if fi_path.exists():
        df_fi = pd.read_csv(fi_path).sort_values("Importance_Mean", ascending=True).tail(15)
        st.markdown("#### Permutation Importance — Top 15 Features")
        fig_fi = px.bar(
            df_fi, x="Importance_Mean", y="Feature", orientation="h",
            color="Importance_Mean", color_continuous_scale="Blues",
        )
        fig_fi.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", showlegend=False)
        st.plotly_chart(fig_fi, width='stretch')
        st.caption("Permutation importance measures overall predictive drop when a feature is shuffled — it can rank differently from SHAP, which explains individual predictions (see below).")

    for fname, caption in [
        ("feature_importance.png", "SHAP summary — feature impact distribution and macro importance ranking."),
        ("pca_tsne_analysis.png", "PCA (linear) vs t-SNE (non-linear) projections — fraud forms isolated micro-clusters, motivating the cascade design over pure clustering."),
        ("optimal_cluster_selection.png", "Silhouette analysis used to pick k for the K-Means exploratory clustering."),
        ("anomaly_checker_analysis.png", "Layer 1 anomaly-score separation between legitimate and fraud instances."),
    ]:
        safe_image(IMAGES_DIR / fname, width='stretch', caption=caption)

# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — LIVE TRANSACTION CHECK
# ──────────────────────────────────────────────────────────────────────────
with tabs[3]:
    if not models_loaded:
        st.error("Models failed to load — see the error above.")
    else:
        st.markdown(f"#### Score a Single Transaction — Layer 1 engine: `{engine_name}`")
        st.caption(
            "⚠️ Single-row scoring has a known limitation: `Scaled_Amount` always evaluates to 0 "
            "for a one-row batch (see the code comment in app.py). It doesn't break the app, but "
            "it does mean Layer 2's view of transaction size is weaker here than in batch scoring."
        )

        preset_cols = st.columns(3)
        if preset_cols[0].button("📥 Load sample: legitimate transaction", width='stretch'):
            st.session_state["tx_values"] = dict(MOCK_LEGIT)
        if preset_cols[1].button("📥 Load sample: fraudulent transaction", width='stretch'):
            st.session_state["tx_values"] = dict(MOCK_FRAUD)
        if preset_cols[2].button("🧹 Clear form", width='stretch'):
            st.session_state["tx_values"] = dict(MOCK_LEGIT)

        defaults = st.session_state.get("tx_values", dict(MOCK_LEGIT))

        with st.form("transaction_form"):
            top_row = st.columns(2)
            time_val = top_row[0].number_input("Time (seconds since first transaction)", value=float(defaults.get("Time", 0.0)), step=1.0)
            amount_val = top_row[1].number_input("Amount ($)", value=float(defaults.get("Amount", 0.0)), min_value=0.0, step=1.0)

            st.caption("PCA-anonymized features V1–V28 (as provided by the source dataset):")
            v_values = {}
            v_cols_layout = st.columns(4)
            for i, vcol in enumerate([f"V{n}" for n in range(1, 29)]):
                with v_cols_layout[i % 4]:
                    v_values[vcol] = st.number_input(vcol, value=float(defaults.get(vcol, 0.0)), format="%.4f", key=f"v_{vcol}")

            submitted = st.form_submit_button("🔍 Run Cascade Check", type="primary", width='stretch')

        if submitted:
            row = {**v_values, "Time": time_val, "Amount": amount_val}
            row_df = pd.DataFrame([row])[RAW_COLS]
            with st.spinner("Scoring transaction through both cascade layers…"):
                result = cascade_predict_single(row_df, processor, layer1_model, layer2_model)

            css_class = {
                "APPROVED": "fs-verdict-approved",
                "BLOCKED_LAYER1": "fs-verdict-layer1",
                "REJECTED_LAYER2": "fs-verdict-layer2",
            }[result["status"]]

            if result["probability"] is not None:
                prob_line = f"Fraud probability: <b>{result['probability']:.2%}</b> (threshold {result.get('threshold', 0):.0%})"
            else:
                prob_line = "Blocked before reaching the Layer 2 classifier."

            st.markdown(
                f"""
                <div class="fs-verdict {css_class}">
                    <h2>{result['label']}</h2>
                    <div><b>Reason:</b> {result['reason']}</div>
                    <div><b>Recommended action:</b> {result['action']}</div>
                    <div class="fs-verdict-detail">
                        Amount: <b>${result['amount']:,.2f}</b> &nbsp;|&nbsp;
                        Layer 1 anomaly score: <b>{result['anomaly_score']:.4f}</b> &nbsp;|&nbsp;
                        {prob_line}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────────────────────────────────
# TAB 5 — BATCH SCORING
# ──────────────────────────────────────────────────────────────────────────
with tabs[4]:
    if not models_loaded:
        st.error("Models failed to load — see the error above.")
    else:
        st.markdown("#### Score a Batch of Transactions")
        st.caption("Upload a CSV with columns Time, V1–V28, Amount (Class optional, used only for an accuracy summary).")

        batch_file = st.file_uploader("Upload transactions CSV", type=["csv"], key="batch_upload")

        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            missing = [c for c in RAW_COLS if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                run_batch = st.button("⚙️ Run Batch Scoring", type="primary")
                if run_batch:
                    with st.spinner(f"Scoring {len(batch_df):,} transactions…"):
                        scored = cascade_predict_batch(batch_df, processor, layer1_model, layer2_model)

                    b1, b2, b3 = st.columns(3)
                    n_blocked = (scored["Decision"] == "Blocked (Layer 1)").sum()
                    n_rejected = (scored["Decision"] == "Rejected (Layer 2)").sum()
                    n_approved = (scored["Decision"] == "Approved").sum()
                    for col, label, value in zip(
                        [b1, b2, b3],
                        ["Blocked — Layer 1", "Rejected — Layer 2", "Approved"],
                        [n_blocked, n_rejected, n_approved],
                    ):
                        col.markdown(
                            f"""<div class="fs-card"><div class="fs-metric-label">{label}</div>
                            <div class="fs-metric-value">{value:,}</div></div>""",
                            unsafe_allow_html=True,
                        )

                    if "Class" in scored.columns:
                        flagged = scored["Decision"] != "Approved"
                        caught = int(((scored["Class"] == 1) & flagged).sum())
                        total_fraud = int((scored["Class"] == 1).sum())
                        if total_fraud > 0:
                            st.caption(f"Of {total_fraud} labeled fraud cases in this file, the cascade flagged **{caught}** across both layers.")

                    st.dataframe(scored, width='stretch', height=420)

                    st.download_button(
                        "⬇️ Download scored results (CSV)",
                        data=scored.to_csv(index=False),
                        file_name="fraud_sentinel_scored_transactions.csv",
                        mime="text/csv",
                        width='stretch',
                    )

# ──────────────────────────────────────────────────────────────────────────
# TAB 6 — HOW IT WORKS
# ──────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("#### System Design")
    st.markdown(
        """
        <div class="fs-card">
        <div class="fs-step"><div class="fs-step-num">1</div><div><b>Feature engineering</b> — Hour, Is_Night,
        Log_Amount, Robust-scaled Amount, and interaction terms between Amount and the 5 features most
        correlated with fraud (V17, V14, V12, V10, V16).</div></div>
        <div class="fs-step"><div class="fs-step-num">2</div><div><b>Chronological split</b> — the final 20% of
        transactions by time were held out for evaluation, rather than a random shuffle, since fraud patterns
        evolve over time.</div></div>
        <div class="fs-step"><div class="fs-step-num">3</div><div><b>Layer 1 — unsupervised anomaly gate</b> —
        by default an Isolation Forest, chosen for cloud deployment because it's lightweight and dependency-free.
        The notebook's PyOD Deep Autoencoder is available as a drop-in alternative from the sidebar if
        <code>pyod</code>/<code>torch</code> are installed locally.</div></div>
        <div class="fs-step"><div class="fs-step-num">4</div><div><b>Layer 2 — calibrated CatBoost</b> — trained
        on SMOTE-balanced data, then wrapped with isotonic calibration so its probability outputs are
        trustworthy, with amount-aware decision thresholds (0.25 above $1,000, 0.50 otherwise).</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "This app loads the real serialized artifacts from `models/` — it does not retrain on "
        "synthetic data. Layer 1 defaults to Isolation Forest rather than the notebook's PyOD "
        "Autoencoder purely for deployment weight (Autoencoder needs `pyod` + `torch`, which "
        "significantly increase build size and cold-start time on free-tier hosting); both were "
        "trained and saved during the original notebook run, and either is a legitimate Layer 1 choice."
    )

    st.warning(
        "**Known limitation:** the `Scaled_Amount` feature is computed by fitting a fresh "
        "`RobustScaler` on whatever batch is passed to the preprocessing pipeline at call time, "
        "rather than reusing statistics fixed at training time. This is a reasonable approximation "
        "for batch scoring (hundreds of rows), but for a *single* transaction it always evaluates "
        "to exactly 0, regardless of the real amount — see the Live Transaction Check tab. The "
        "correct fix is retraining `fraud_processor.pkl` with a `RobustScaler` fit once on the "
        "training set and reused for every future call, rather than refit per call."
    )

st.markdown(
    '<div class="fs-footer">Fraud Sentinel — built for demonstration and educational purposes. '
    'Not intended for production financial decisioning without further validation.</div>',
    unsafe_allow_html=True,
)
