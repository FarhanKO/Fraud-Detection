import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# A transaction with every field genuinely zeroed out — used by "Clear form".
# (Previously "Clear form" incorrectly reset to MOCK_LEGIT, same as the
# legitimate-sample button, so it never actually cleared anything.)
ZERO_TX = {col: 0.0 for col in RAW_COLS}

# ──────────────────────────────────────────────────────────────────────────
# COMPATIBILITY SHIM — DO NOT REMOVE
# ──────────────────────────────────────────────────────────────────────────
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

# Supplementary styling for the new sidebar-navigation layout, the two-phase
# Live Transaction Check flow, the "How It Works" flow diagram, and the
# footer. Kept self-contained here (rather than relying on assets/style.css)
# so the new layout renders correctly even before that stylesheet is updated.
EXTRA_CSS = """
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.55rem 0.75rem;
    border-radius: 10px;
    margin-bottom: 0.15rem;
    transition: background-color 0.15s ease, transform 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: rgba(59, 130, 246, 0.15);
    transform: translateX(2px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background-color: rgba(59, 130, 246, 0.28);
}
.fs-engine-badge {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #86efac;
    font-size: 0.82rem;
    font-weight: 600;
}
.fs-engine-badge.fallback {
    background: rgba(249, 115, 22, 0.15);
    border-color: rgba(249, 115, 22, 0.4);
    color: #fdba74;
}
.fs-mini-arch { display: flex; flex-direction: column; align-items: stretch; margin: 0.4rem 0; }
.fs-mini-node {
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 10px;
    padding: 0.5rem 0.65rem;
    font-size: 0.8rem;
    font-weight: 700;
    line-height: 1.5;
}
.fs-mini-node span { font-weight: 500; font-size: 0.74rem; color: #94a3b8; }
.fs-mini-node.l1 { border-left: 3px solid #3b82f6; }
.fs-mini-node.l2 { border-left: 3px solid #a855f7; }
.fs-mini-node.risk { border-left: 3px solid #facc15; }
.fs-mini-node.final { border-left: 3px solid #22c55e; text-align: center; }
.fs-mini-arrow { text-align: center; color: #64748b; font-size: 1rem; line-height: 1.4; }
.fs-fade-in { animation: fsFadeIn 0.45s ease; }
@keyframes fsFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fs-verdict { border-radius: 16px; padding: 1.4rem 1.6rem; margin: 0.5rem 0 1rem 0; }
.fs-verdict h2 { margin-top: 0; }
.fs-chart-caption {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: -0.4rem;
    margin-bottom: 1rem;
}
.fs-flow-wrap {
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    gap: 0.4rem;
    margin: 1rem 0 1.5rem 0;
}
.fs-flow-card {
    flex: 1 1 150px;
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 12px;
    padding: 0.85rem 0.9rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.fs-flow-card:hover {
    transform: translateY(-3px);
    border-color: rgba(59, 130, 246, 0.6);
}
.fs-flow-card summary {
    cursor: pointer;
    font-weight: 700;
    font-size: 0.95rem;
    list-style: none;
}
.fs-flow-card summary::-webkit-details-marker { display: none; }
.fs-flow-card p { color: #94a3b8; font-size: 0.82rem; margin: 0.5rem 0 0 0; }
.fs-flow-arrow {
    align-self: center;
    font-size: 1.4rem;
    color: #64748b;
    padding: 0 0.15rem;
}
.fs-flow-card.layer1 { border-left: 3px solid #3b82f6; }
.fs-flow-card.layer2 { border-left: 3px solid #a855f7; }
.fs-flow-card.approve { border-left: 3px solid #22c55e; }
.fs-flow-card.reject { border-left: 3px solid #ef4444; }
.fs-footer2 {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    border-top: 1px solid rgba(148, 163, 184, 0.2);
    margin-top: 2.5rem;
    padding: 1.2rem 0.2rem 0.4rem 0.2rem;
    color: #94a3b8;
    font-size: 0.9rem;
}
.fs-footer2 .fs-footer-links a {
    color: #94a3b8;
    text-decoration: none;
    margin-left: 1.1rem;
    transition: color 0.15s ease;
}
.fs-footer2 .fs-footer-links a:hover { color: #60a5fa; }
</style>
"""
st.markdown(EXTRA_CSS, unsafe_allow_html=True)

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
            engine_is_full = True
        except Exception as e:  # noqa: BLE001 — deliberately broad, this is a soft fallback
            st.warning(
                f"Couldn't load the Autoencoder Layer 1 model ({type(e).__name__}: {e}). "
                "This almost always means `pyod`/`torch` aren't installed in this "
                "environment — they're intentionally left out of requirements.txt to "
                "keep cloud deployments lightweight. Falling back to Isolation Forest."
            )
            layer1 = joblib.load(MODELS_DIR / "layer1_isolation_forest.pkl")
            engine_name = "Isolation Forest (fallback)"
            engine_is_full = False
    else:
        layer1 = joblib.load(MODELS_DIR / "layer1_isolation_forest.pkl")
        engine_name = "Isolation Forest"
        engine_is_full = False

    return processor, layer1, layer2, engine_name, engine_is_full


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


def explainability_block(title: str, image_name: str, caption: str, interpretation: str, why_it_matters: str):
    """Image + caption + an expandable Interpretation / Why It Matters
    write-up underneath — used for every static image in the Explainability
    page, mirroring the structure of the interactive Permutation Importance
    section above it."""
    st.markdown(f"##### {title}")
    safe_image(IMAGES_DIR / image_name, width='stretch')
    if caption:
        st.markdown(f'<div class="fs-chart-caption">{caption}</div>', unsafe_allow_html=True)
    with st.expander("📋 Interpretation & why it matters"):
        st.markdown(f"**Interpretation**\n\n{interpretation}")
        st.markdown(f"**Why it matters**\n\n{why_it_matters}")


PLOT_BG = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")

# ──────────────────────────────────────────────────────────────────────────
# LOAD MODELS — Layer 1 always attempts the full Deep Autoencoder
# architecture first (matching the notebook exactly) and only falls back to
# Isolation Forest automatically if pyod/torch aren't installed in this
# environment. There's no manual toggle for this any more — the app always
# tries to run the full architecture on its own.
# ──────────────────────────────────────────────────────────────────────────
try:
    processor, layer1_model, layer2_model, engine_name, engine_is_full = load_models(True)
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    engine_name, engine_is_full = "unavailable", False
    processor = layer1_model = layer2_model = None

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — navigation now lives here, along with engine status, dataset
# info, and links. Everything that used to be a top row of tabs is now a
# vertical nav list in this left-hand panel.
# ──────────────────────────────────────────────────────────────────────────
PAGES = [
    "⚡ Live Transaction Check",
    "📊 Overview",
    "📈 Model Performance",
    "🧠 Explainability",
    "📁 Batch Scoring",
    "ℹ️ How It Works",
]

with st.sidebar:
    st.markdown("### 🛡️ Fraud Sentinel")
    st.caption("Dual-layer cascade — served from the real trained artifacts, not a synthetic demo.")
    st.divider()

    st.markdown("**Navigate**")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="nav_page")

    st.divider()
    st.markdown("**System Architecture**")
    badge_class = "fs-engine-badge" if engine_is_full else "fs-engine-badge fallback"
    st.markdown(
        f"""
        <div class="fs-mini-arch">
            <div class="fs-mini-node l1">Layer 1<br><span class="{badge_class}">{engine_name}</span></div>
            <div class="fs-mini-arrow">↓</div>
            <div class="fs-mini-node l2">Layer 2<br><span>Calibrated CatBoost</span></div>
            <div class="fs-mini-arrow">↓</div>
            <div class="fs-mini-node risk">Risk Engine<br><span>Amount-aware thresholds</span></div>
            <div class="fs-mini-arrow">↓</div>
            <div class="fs-mini-node final">Final Decision</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Layer 1 always tries the full Deep Autoencoder architecture from the notebook first, "
        "falling back to Isolation Forest automatically only if `pyod`/`torch` aren't installed — "
        "no manual switch, and this diagram doesn't change based on any selection."
    )

    st.divider()
    st.markdown("**Training dataset**")
    st.caption("284,807 transactions · 492 confirmed fraud (0.172%) · European cardholders, Sept 2013")

    st.divider()
    st.markdown("[📓 View training notebook](https://github.com/FarhanKO/Fraud-Detection/blob/main/Fraud_Detection.ipynb)")
    st.markdown("[📄 Repo README](https://github.com/FarhanKO/Fraud-Detection)")

if not models_loaded:
    st.error(
        "Couldn't find a required model file. Make sure `models/` contains "
        "fraud_processor.pkl, layer1_isolation_forest.pkl (and ideally "
        "layer1_autoencoder.pkl), and layer2_calibrated_catboost.pkl."
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

# ──────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ──────────────────────────────────────────────────────────────────────────
st.session_state.setdefault("tx_values", dict(MOCK_LEGIT))
st.session_state.setdefault("tx_result", None)
st.session_state.setdefault("tx_row", None)
st.session_state.setdefault("tx_history", [])

# ══════════════════════════════════════════════════════════════════════════
# PAGE — LIVE TRANSACTION CHECK  (default landing page)
# ══════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    if not models_loaded:
        st.error("Models failed to load — see the error above.")
    else:
        result = st.session_state["tx_result"]

        if result is None:
            # ── PHASE 1: INPUT FORM ────────────────────────────────────
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
                st.session_state["tx_values"] = dict(ZERO_TX)

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
                    computed = cascade_predict_single(row_df, processor, layer1_model, layer2_model)

                st.session_state["tx_result"] = computed
                st.session_state["tx_row"] = row
                st.session_state["tx_history"].append({
                    "Check #": len(st.session_state["tx_history"]) + 1,
                    "Amount": computed["amount"],
                    "Decision": computed["label"],
                    "Anomaly Score": computed["anomaly_score"],
                    "Fraud Probability": computed["probability"],
                })
                st.rerun()

        else:
            # ── PHASE 2: VERDICT ────────────────────────────────────────
            row = st.session_state.get("tx_row") or {}

            css_class = {
                "APPROVED": "fs-verdict-approved",
                "BLOCKED_LAYER1": "fs-verdict-layer1",
                "REJECTED_LAYER2": "fs-verdict-layer2",
            }[result["status"]]

            # Presentational-only derivations for the result view — these read
            # the cascade's own outputs, they don't change what the cascade
            # decided or how (see cascade_predict_single above).
            layer1_result = "🚫 Blocked" if result["status"] == "BLOCKED_LAYER1" else "✅ Passed"
            if result["status"] == "BLOCKED_LAYER1":
                layer2_result = "— Not reached"
            elif result["status"] == "REJECTED_LAYER2":
                layer2_result = "❌ Rejected"
            else:
                layer2_result = "✅ Approved"

            if result["status"] == "BLOCKED_LAYER1":
                risk_level = "🔴 High"
            else:
                thr = result.get("threshold", 0.5)
                prob = result["probability"] or 0.0
                if prob >= thr:
                    risk_level = "🔴 High"
                elif prob >= thr * 0.5:
                    risk_level = "🟠 Medium"
                else:
                    risk_level = "🟢 Low"

            if result["probability"] is not None:
                prob_line = f"Fraud probability: <b>{result['probability']:.2%}</b> (threshold {result.get('threshold', 0):.0%})"
            else:
                prob_line = "Blocked before reaching the Layer 2 classifier."

            st.markdown(
                f"""
                <div class="fs-verdict fs-fade-in {css_class}">
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

            r1, r2, r3 = st.columns(3)
            for col, label, value in zip(
                [r1, r2, r3],
                ["Layer 1 Result", "Layer 2 Result", "Risk Level"],
                [layer1_result, layer2_result, risk_level],
            ):
                col.markdown(
                    f"""<div class="fs-card"><div class="fs-metric-label">{label}</div>
                    <div class="fs-metric-value">{value}</div></div>""",
                    unsafe_allow_html=True,
                )

            m1, m2, m3, m4 = st.columns(4)
            for col, label, value in zip(
                [m1, m2, m3, m4],
                ["Amount", "Layer 1 Anomaly Score", "Fraud Probability", "Decision Threshold"],
                [
                    f"${result['amount']:,.2f}",
                    f"{result['anomaly_score']:.4f}",
                    f"{result['probability']:.2%}" if result["probability"] is not None else "N/A",
                    f"{result.get('threshold', 0):.0%}" if result.get("threshold") is not None else "—",
                ],
            ):
                col.markdown(
                    f"""<div class="fs-card"><div class="fs-metric-label">{label}</div>
                    <div class="fs-metric-value">{value}</div></div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("#### Why this verdict")
            chart_cols = st.columns(2)

            with chart_cols[0]:
                if result["probability"] is not None:
                    threshold = result.get("threshold", 0.5)
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["probability"] * 100,
                        number={"suffix": "%"},
                        title={"text": "Layer 2 Fraud Probability"},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                            "bar": {"color": "#e2e8f0"},
                            "steps": [
                                {"range": [0, threshold * 100], "color": "rgba(34,197,94,0.5)"},
                                {"range": [threshold * 100, 100], "color": "rgba(239,68,68,0.5)"},
                            ],
                            "threshold": {
                                "line": {"color": "#facc15", "width": 4},
                                "thickness": 0.9,
                                "value": threshold * 100,
                            },
                        },
                    ))
                    gauge.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20), **PLOT_BG)
                    st.plotly_chart(gauge, width='stretch')
                    st.caption(f"Yellow line marks the decision threshold ({threshold:.0%}) — above it, the transaction is rejected.")
                else:
                    st.markdown(
                        f"""<div class="fs-card"><div class="fs-metric-label">Layer 1 Anomaly Score</div>
                        <div class="fs-metric-value">{result['anomaly_score']:.4f}</div></div>""",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "This transaction never reached Layer 2 — it was already flagged as a "
                        "structural anomaly by Layer 1, so no fraud probability was computed. "
                        "Higher anomaly scores indicate a bigger departure from normal transaction structure."
                    )

            with chart_cols[1]:
                v_items = [(k, v) for k, v in row.items() if k.startswith("V")]
                if v_items:
                    v_df = pd.DataFrame(v_items, columns=["Feature", "Value"])
                    v_df["AbsValue"] = v_df["Value"].abs()
                    v_df = v_df.sort_values("AbsValue", ascending=False).head(10).sort_values("Value")
                    v_df["Sign"] = np.where(v_df["Value"] >= 0, "Positive", "Negative")
                    vfig = px.bar(
                        v_df, x="Value", y="Feature", orientation="h", color="Sign",
                        color_discrete_map={"Positive": "#3b82f6", "Negative": "#ef4444"},
                        title="Top 10 Contributing V-Features (this transaction)",
                    )
                    vfig.update_layout(height=300, margin=dict(t=60, b=10, l=10, r=10), showlegend=False, **PLOT_BG)
                    st.plotly_chart(vfig, width='stretch')
                    st.caption("The 10 anonymized V-features with the largest magnitude for this transaction — the ones most likely driving the cascade's verdict.")

            if len(st.session_state["tx_history"]) > 1:
                st.markdown("#### This session's checks")
                hist_df = pd.DataFrame(st.session_state["tx_history"])
                hfig = px.scatter(
                    hist_df, x="Check #", y="Amount", color="Decision", size=hist_df["Amount"].clip(lower=1),
                    hover_data=["Anomaly Score", "Fraud Probability"],
                )
                hfig.update_layout(height=320, **PLOT_BG)
                st.plotly_chart(hfig, width='stretch')
                st.caption("Every transaction you've checked this session — a quick visual log, not persisted after you close the tab.")

            st.button(
                "🔁 Check another transaction",
                type="primary",
                width='stretch',
                on_click=lambda: (st.session_state.update(tx_result=None, tx_row=None)),
            )

# ══════════════════════════════════════════════════════════════════════════
# PAGE — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.markdown("#### Dataset & Production Model at a Glance")
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

    dcol1, dcol2 = st.columns([3, 2])
    with dcol1:
        scale_choice = st.radio("Scale", ["Logarithmic", "Linear"], horizontal=True, key="dist_scale")
        fig = px.bar(
            dist_df, x="Class", y="Count", color="Class", text="Count",
            log_y=(scale_choice == "Logarithmic"),
            color_discrete_map={"Legitimate": "#3b82f6", "Fraud": "#ef4444"},
        )
        fig.update_layout(showlegend=False, height=380, **PLOT_BG)
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Logarithmic scale — fraud is 0.172% of all transactions, which is why accuracy alone "
            "is a meaningless metric here. Switch to linear to see just how invisible fraud is at true scale."
        )

    with dcol2:
        pie = go.Figure(data=[go.Pie(
            labels=["Legitimate", "Fraud"], values=[284315, 492], hole=0.6,
            marker=dict(colors=["#3b82f6", "#ef4444"]),
            textinfo="percent", hovertemplate="%{label}: %{value:,}<extra></extra>",
        )])
        pie.update_layout(
            height=380, showlegend=True,
            annotations=[dict(text="284,807<br>total", x=0.5, y=0.5, font_size=13, showarrow=False, font_color="#e2e8f0")],
            **PLOT_BG,
        )
        st.plotly_chart(pie, width='stretch')
        st.caption("Same split, proportional view.")

    st.markdown("#### Production Model Snapshot")
    metric_names = ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    metric_vals = [0.9483, 0.7333, 0.8271, 0.9793, 0.8042]
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=metric_vals + [metric_vals[0]], theta=metric_names + [metric_names[0]],
        fill="toself", line_color="#3b82f6", fillcolor="rgba(59,130,246,0.35)",
        name="Calibrated CatBoost",
    ))
    radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], color="#94a3b8"),
            angularaxis=dict(color="#e2e8f0"),
        ),
        showlegend=False, height=420, **PLOT_BG,
    )
    st.plotly_chart(radar, width='stretch')
    st.caption("The five headline metrics for the production Layer 2 model, in one shape — full write-up and comparison against other candidate models is on the Model Performance page.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
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

        st.markdown("#### Model Comparison — Interactive")
        metric_cols = [c for c in ["Precision_Fraud", "Recall_Fraud", "F1_Score_Fraud", "ROC_AUC", "PR_AUC"] if c in df_metrics.columns]
        if metric_cols and "Model" in df_metrics.columns:
            long_df = df_metrics.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            cmp_fig = px.bar(
                long_df, x="Metric", y="Score", color="Model", barmode="group",
                hover_data={"Score": ":.4f"},
            )
            cmp_fig.update_layout(height=420, yaxis_range=[0, 1], **PLOT_BG)
            st.plotly_chart(cmp_fig, width='stretch')
            st.caption("Live chart built directly from results/metrics.csv — hover any bar for the exact score. Grouped by metric so it's easy to see which model wins where.")
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

    st.markdown("#### Diagnostic Plots")
    perf_details = {
        "roc_pr_curve.png": (
            "ROC & Precision-Recall Curves",
            "ROC and Precision-Recall curves across every supervised model.",
            "The ROC curve plots true-positive rate against false-positive rate — but with fraud at "
            "0.172% of transactions, a model can look great on ROC while still missing most fraud "
            "cases. The Precision-Recall curve is the more honest read here: it shows the real "
            "trade-off between catching fraud (recall) and not over-flagging legitimate customers "
            "(precision), which is why **PR-AUC**, not ROC-AUC, is used as the primary selection metric.",
        ),
        "model_comparison.png": (
            "Head-to-Head Model Comparison",
            "Head-to-head comparison across accuracy, precision, recall, F1, and AUC.",
            "Every candidate model from the notebook — trained on the same SMOTE-balanced data and "
            "evaluated on the same chronological holdout — compared side by side across accuracy, "
            "precision, recall, F1, and AUC. Calibrated CatBoost was selected as the production model "
            "for the best balance of precision and PR-AUC, not because it topped every single metric.",
        ),
        "threshold_analysis.png": (
            "Threshold & Business-Impact Analysis",
            "Confusion matrix and business-impact sensitivity across decision thresholds.",
            "Sweeps the decision threshold across its full range and tracks the resulting confusion "
            "matrix and estimated dollar cost. This is exactly where the amount-aware thresholds used "
            "live (0.25 above $1,000, 0.50 otherwise) come from — high-value transactions get a "
            "lower bar to flag because a missed high-value fraud costs far more than a false alarm.",
        ),
    }
    for fname, (title, caption, detail) in perf_details.items():
        st.markdown(f"##### {title}")
        safe_image(IMAGES_DIR / fname, width='stretch')
        st.markdown(f'<div class="fs-chart-caption">{caption}</div>', unsafe_allow_html=True)
        with st.expander("📋 What this shows"):
            st.markdown(detail)

# ══════════════════════════════════════════════════════════════════════════
# PAGE — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.info(
        "The Permutation Importance chart below is fully interactive because its underlying numbers "
        "are saved to `results/feature_importance.csv`. The static images further down are exports "
        "straight from the notebook — export their underlying arrays (e.g. SHAP values, 2-D embedding "
        "coordinates, silhouette scores) to `results/*.csv` the same way and they can become interactive too."
    )

    fi_path = RESULTS_DIR / "feature_importance.csv"
    if fi_path.exists():
        df_fi = pd.read_csv(fi_path).sort_values("Importance_Mean", ascending=True).tail(15)
        st.markdown("#### Permutation Importance — Top 15 Features")
        fi_view = st.radio("View", ["Bar", "Treemap"], horizontal=True, key="fi_view")
        if fi_view == "Bar":
            fig_fi = px.bar(
                df_fi, x="Importance_Mean", y="Feature", orientation="h",
                color="Importance_Mean", color_continuous_scale="Blues",
            )
            fig_fi.update_layout(showlegend=False, height=460, **PLOT_BG)
        else:
            fig_fi = px.treemap(
                df_fi, path=["Feature"], values="Importance_Mean",
                color="Importance_Mean", color_continuous_scale="Blues",
            )
            fig_fi.update_layout(height=460, **PLOT_BG)
        st.plotly_chart(fig_fi, width='stretch')
        st.caption("Permutation importance measures overall predictive drop when a feature is shuffled — it can rank differently from SHAP, which explains individual predictions (see below).")
        with st.expander("📋 Interpretation & why it matters"):
            st.markdown(
                "**Interpretation**\n\n"
                "Each feature is randomly shuffled, one at a time, and the model is re-scored — the "
                "resulting drop in performance is that feature's importance."
            )
            st.markdown(
                "**Why it matters**\n\n"
                "Unlike SHAP, this measures **overall** predictive power rather than per-transaction "
                "impact, and is model-agnostic: it works the same way regardless of which classifier "
                "produced it, making it a good sanity check against the SHAP ranking below."
            )

    explainability_block(
        "SHAP Summary",
        "feature_importance.png",
        "SHAP summary — feature impact distribution and macro importance ranking.",
        "Each dot is one transaction; its position on the x-axis is that feature's push toward "
        "'fraud' (right) or 'legitimate' (left) for that specific transaction, and color usually "
        "encodes the feature's own value. Features are ranked top-to-bottom by average impact.",
        "This is what makes SHAP different from permutation importance: it explains *individual* "
        "predictions rather than just an overall feature ranking. A feature like V14 having a wide, "
        "clearly-colored spread means it strongly and consistently pushes many individual predictions "
        "toward fraud — exactly the kind of signal a fraud analyst would want to see before trusting a verdict.",
    )
    explainability_block(
        "PCA vs t-SNE Projection",
        "pca_tsne_analysis.png",
        "PCA (linear) vs t-SNE (non-linear) projections — fraud forms isolated micro-clusters, motivating the cascade design over pure clustering.",
        "PCA finds the directions of maximum linear variance in the data; t-SNE instead tries to "
        "preserve local neighborhoods non-linearly, which is usually better at revealing tight clusters.",
        "Fraud transactions form small, isolated pockets rather than one clean region — which is "
        "exactly why a single clustering pass isn't enough, and why the pipeline needs Layer 1 "
        "(anomaly detection) followed by Layer 2 (a supervised classifier) rather than one model alone.",
    )
    explainability_block(
        "Optimal Cluster Selection",
        "optimal_cluster_selection.png",
        "Silhouette analysis used to pick k for the K-Means exploratory clustering.",
        "Silhouette score measures how well-separated clusters are — closer to 1 means points sit "
        "clearly inside their own cluster and far from neighboring ones.",
        "This chart was used during exploratory analysis (not the production pipeline) to sanity-check "
        "how naturally the data separates before committing to the two-layer cascade design.",
    )
    explainability_block(
        "Layer 1 Anomaly Score Separation",
        "anomaly_checker_analysis.png",
        "Layer 1 anomaly-score separation between legitimate and fraud instances.",
        "Shows how cleanly the Layer 1 engine's anomaly scores separate known fraud from legitimate "
        "transactions before any labels are used.",
        "The better this separation, the more fraud Layer 1 can catch on structure alone — before a "
        "transaction ever reaches the supervised Layer 2 classifier, which is what makes it a useful "
        "first gate rather than a redundant step.",
    )

# ══════════════════════════════════════════════════════════════════════════
# PAGE — BATCH SCORING
# ══════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
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

                    decision_colors = {"Approved": "#22c55e", "Rejected (Layer 2)": "#ef4444", "Blocked (Layer 1)": "#f97316"}

                    st.markdown("##### Batch Breakdown")
                    ch1, ch2 = st.columns(2)
                    with ch1:
                        bpie = px.pie(
                            scored, names="Decision", color="Decision", hole=0.55,
                            color_discrete_map=decision_colors,
                        )
                        bpie.update_layout(height=340, **PLOT_BG)
                        st.plotly_chart(bpie, width='stretch')
                        st.caption("Share of this batch by final cascade decision.")
                    with ch2:
                        bhist = px.histogram(
                            scored.dropna(subset=["Fraud_Probability"]), x="Fraud_Probability",
                            color="Decision", nbins=30, color_discrete_map=decision_colors,
                        )
                        bhist.update_layout(height=340, **PLOT_BG)
                        st.plotly_chart(bhist, width='stretch')
                        st.caption("Distribution of Layer 2 fraud probabilities among transactions that weren't already blocked at Layer 1.")

                    st.dataframe(scored, width='stretch', height=420)

                    st.download_button(
                        "⬇️ Download scored results (CSV)",
                        data=scored.to_csv(index=False),
                        file_name="fraud_sentinel_scored_transactions.csv",
                        mime="text/csv",
                        width='stretch',
                    )

# ══════════════════════════════════════════════════════════════════════════
# PAGE — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.markdown("#### System Design")
    st.caption("Click any card below to expand it — the pipeline runs left to right.")

    st.markdown(
        """
        <div class="fs-flow-wrap">
          <details class="fs-flow-card" open>
            <summary>1️⃣ Raw Transaction</summary>
            <p>Time, Amount, and 28 PCA-anonymized features (V1–V28) as provided by the source dataset — nothing engineered yet.</p>
          </details>
          <div class="fs-flow-arrow">→</div>
          <details class="fs-flow-card">
            <summary>2️⃣ Feature Engineering</summary>
            <p>Derives Hour and Is_Night from Time, Log_Amount and a Robust-scaled Amount, plus interaction terms between Amount and the 5 features most correlated with fraud (V17, V14, V12, V10, V16).</p>
          </details>
          <div class="fs-flow-arrow">→</div>
          <details class="fs-flow-card layer1">
            <summary>3️⃣ Layer 1 — Anomaly Gate</summary>
            <p>Always tries the full Deep Autoencoder from the notebook first; falls back to an Isolation Forest automatically if pyod/torch aren't installed. Flags structural outliers before any label is consulted.</p>
          </details>
          <div class="fs-flow-arrow">→</div>
          <details class="fs-flow-card reject">
            <summary>🚨 Blocked at Layer 1</summary>
            <p>Routed straight to fraud-forensics review as a possible zero-day pattern — never reaches Layer 2.</p>
          </details>
        </div>
        <div class="fs-flow-wrap">
          <details class="fs-flow-card layer1">
            <summary>↳ Passes Layer 1</summary>
            <p>Transactions that look structurally normal continue on to the supervised classifier.</p>
          </details>
          <div class="fs-flow-arrow">→</div>
          <details class="fs-flow-card layer2">
            <summary>4️⃣ Layer 2 — Calibrated CatBoost</summary>
            <p>Trained on SMOTE-balanced data, then wrapped with isotonic calibration so its probability outputs are trustworthy. Applies amount-aware thresholds: 0.25 above $1,000, 0.50 otherwise.</p>
          </details>
          <div class="fs-flow-arrow">→</div>
          <details class="fs-flow-card reject">
            <summary>❌ Rejected</summary>
            <p>Probability clears the threshold — auto-declined and the account is flagged for review.</p>
          </details>
          <div class="fs-flow-arrow">→</div>
          <details class="fs-flow-card approve">
            <summary>✅ Approved</summary>
            <p>Passed both the structural check and the pattern-match check — funds clearance is authorized.</p>
          </details>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Why Two Layers?")
    lc1, lc2 = st.columns(2)
    with lc1:
        with st.expander("🧠 Layer 1 — unsupervised anomaly gate", expanded=True):
            st.markdown(
                "- Never sees fraud labels — learns what *normal* looks like and flags departures from it\n"
                "- Catches genuinely novel fraud patterns a supervised model has never been trained on\n"
                "- Deep Autoencoder (full architecture) tried first; Isolation Forest is the automatic, "
                "dependency-free fallback for lightweight deployment\n"
                "- Fast and cheap — acts as a first-pass gate before the heavier classifier"
            )
    with lc2:
        with st.expander("🎯 Layer 2 — calibrated supervised classifier", expanded=True):
            st.markdown(
                "- CatBoost trained on SMOTE-balanced data so it isn't overwhelmed by the 99.83% legitimate class\n"
                "- Isotonic calibration makes the output probabilities trustworthy, not just rank-ordered\n"
                "- Amount-aware thresholds reflect real business cost: a missed high-value fraud costs "
                "far more than a false alarm on a small purchase\n"
                "- Only ever sees transactions that already passed the Layer 1 structural check"
            )

    st.markdown("#### Decision Threshold by Amount")
    amt_range = np.linspace(0, 3000, 300)
    thr_curve = np.where(amt_range > 1000, 0.25, 0.50)
    thr_fig = go.Figure()
    thr_fig.add_trace(go.Scatter(
        x=amt_range, y=thr_curve, mode="lines", line=dict(color="#facc15", width=3, shape="hv"),
        name="Decision threshold",
    ))
    thr_fig.add_vline(x=1000, line_dash="dot", line_color="#94a3b8", annotation_text="$1,000", annotation_position="top")
    thr_fig.update_layout(
        height=320, xaxis_title="Transaction Amount ($)", yaxis_title="Fraud-probability threshold to reject",
        yaxis=dict(range=[0, 0.6]), **PLOT_BG,
    )
    st.plotly_chart(thr_fig, width='stretch')
    st.caption("Transactions over $1,000 get flagged at a *lower* probability threshold — the cascade is deliberately more cautious with larger amounts.")

    st.info(
        "This app loads the real serialized artifacts from `models/` — it does not retrain on "
        "synthetic data. Layer 1 always attempts the notebook's full Deep Autoencoder architecture "
        "and only drops to Isolation Forest automatically when `pyod`/`torch` aren't available; both "
        "were trained and saved during the original notebook run, and either is a legitimate Layer 1 choice."
    )

    st.warning(
        "**Known limitation:** the `Scaled_Amount` feature is computed by fitting a fresh "
        "`RobustScaler` on whatever batch is passed to the preprocessing pipeline at call time, "
        "rather than reusing statistics fixed at training time. This is a reasonable approximation "
        "for batch scoring (hundreds of rows), but for a *single* transaction it always evaluates "
        "to exactly 0, regardless of the real amount — see the Live Transaction Check page. The "
        "correct fix is retraining `fraud_processor.pkl` with a `RobustScaler` fit once on the "
        "training set and reused for every future call, rather than refit per call."
    )

# ──────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="fs-footer2">
        <div>A product of Farhan</div>
        <div class="fs-footer-links">
            <a href="https://www.linkedin.com/in/md-farhan-cse/" target="_blank">LinkedIn</a>
            <a href="https://github.com/FarhanKO" target="_blank">GitHub</a>
            <a href="mailto:farhanzian22@gmail.com">Mail</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
