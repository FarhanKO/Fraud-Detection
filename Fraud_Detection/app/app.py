import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
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
.fs-page-banner {
    border-left: 4px solid #3b82f6;
    background: rgba(59, 130, 246, 0.08);
    border-radius: 0 12px 12px 0;
    padding: 0.75rem 1.1rem;
    margin: 0.2rem 0 1.4rem 0;
}
.fs-page-banner h2 { margin: 0 0 0.15rem 0; font-size: 1.3rem; }
.fs-page-banner p { margin: 0; color: #94a3b8; font-size: 0.9rem; }
.fs-insight-card {
    background: linear-gradient(135deg, rgba(250, 204, 21, 0.10), rgba(59, 130, 246, 0.08));
    border: 1px solid rgba(250, 204, 21, 0.35);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin: 0.8rem 0 1.3rem 0;
}
.fs-insight-card h4 { margin-top: 0; }
.fs-cost-fn-card {
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin: 0.9rem auto;
    max-width: 720px;
    text-align: center;
}
.fs-cost-fn-card code {
    display: block;
    font-size: 1.05rem;
    background: rgba(0,0,0,0.25);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin: 0.6rem 0;
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
# ANIMATED BACKGROUND — a full-page, cursor-reactive "security scan" network:
# faint drifting nodes connect with thin lines, and any link near the cursor
# lights up green like a live scan sweeping the page. Pure browser-side
# canvas/JS, no Streamlit reruns triggered. It's injected via components.html
# (which normally only renders inside its own little iframe box) — the script
# reaches into `window.parent.document` to attach the canvas to the *actual*
# page behind everything else (z-index -1, pointer-events disabled so it
# never blocks clicks), and a one-line guard skips re-creating it if a canvas
# is already running from an earlier script rerun.
# NOTE: this relies on undocumented Streamlit DOM internals (parent-document
# access from a component iframe), not an official API — it's a known,
# widely-used trick but isn't guaranteed to survive future Streamlit
# versions. If a future update ever renders it inert, this whole block can be
# safely deleted with no effect on the rest of the app.
# ──────────────────────────────────────────────────────────────────────────
def render_cyber_background():
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            if (doc.getElementById('fs-cyber-canvas')) return;  // already running from a previous rerun

            const style = doc.createElement('style');
            style.textContent = `
                [data-testid="stAppViewContainer"],
                [data-testid="stHeader"],
                .stApp { background: transparent !important; }
                #fs-cyber-canvas {
                    position: fixed;
                    top: 0; left: 0;
                    width: 100vw; height: 100vh;
                    z-index: -1;
                    pointer-events: none;
                }
            `;
            doc.head.appendChild(style);

            const canvas = doc.createElement('canvas');
            canvas.id = 'fs-cyber-canvas';
            doc.body.prepend(canvas);
            const ctx = canvas.getContext('2d');

            let w, h;
            function resize() {
                w = canvas.width = window.parent.innerWidth;
                h = canvas.height = window.parent.innerHeight;
            }
            resize();
            window.parent.addEventListener('resize', resize);

            const NODE_COUNT = Math.max(30, Math.floor((w * h) / 18000));
            const LINK_DIST = 140;
            const CURSOR_RADIUS = 180;
            const nodes = [];
            for (let i = 0; i < NODE_COUNT; i++) {
                nodes.push({
                    x: Math.random() * w,
                    y: Math.random() * h,
                    vx: (Math.random() - 0.5) * 0.25,
                    vy: (Math.random() - 0.5) * 0.25,
                });
            }

            const mouse = { x: -9999, y: -9999 };
            doc.addEventListener('mousemove', (e) => {
                mouse.x = e.clientX;
                mouse.y = e.clientY;
            });
            doc.addEventListener('mouseleave', () => {
                mouse.x = -9999;
                mouse.y = -9999;
            });

            function step() {
                ctx.clearRect(0, 0, w, h);

                for (const n of nodes) {
                    n.x += n.vx;
                    n.y += n.vy;
                    if (n.x < 0 || n.x > w) n.vx *= -1;
                    if (n.y < 0 || n.y > h) n.vy *= -1;
                }

                for (let i = 0; i < nodes.length; i++) {
                    for (let j = i + 1; j < nodes.length; j++) {
                        const a = nodes[i], b = nodes[j];
                        const dx = a.x - b.x, dy = a.y - b.y;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < LINK_DIST) {
                            const midX = (a.x + b.x) / 2, midY = (a.y + b.y) / 2;
                            const cdx = midX - mouse.x, cdy = midY - mouse.y;
                            const near = Math.sqrt(cdx * cdx + cdy * cdy) < CURSOR_RADIUS;
                            ctx.strokeStyle = near
                                ? `rgba(34, 197, 94, ${0.55 * (1 - dist / LINK_DIST)})`
                                : `rgba(59, 130, 246, ${0.12 * (1 - dist / LINK_DIST)})`;
                            ctx.lineWidth = near ? 1.2 : 0.6;
                            ctx.beginPath();
                            ctx.moveTo(a.x, a.y);
                            ctx.lineTo(b.x, b.y);
                            ctx.stroke();
                        }
                    }
                }

                for (const n of nodes) {
                    const dx = n.x - mouse.x, dy = n.y - mouse.y;
                    const near = Math.sqrt(dx * dx + dy * dy) < CURSOR_RADIUS;
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, near ? 2.4 : 1.6, 0, Math.PI * 2);
                    ctx.fillStyle = near ? 'rgba(34, 197, 94, 0.9)' : 'rgba(148, 163, 184, 0.5)';
                    ctx.fill();
                }

                window.parent.requestAnimationFrame(step);
            }
            step();
        })();
        </script>
        """,
        height=0,
        width=0,
    )


render_cyber_background()


# ──────────────────────────────────────────────────────────────────────────
# SCROLL-TO-TOP ON PAGE CHANGE — Streamlit never actually reloads the
# browser page when a sidebar widget changes; it patches the DOM over the
# same websocket connection, so whatever scroll position you were at on the
# previous page just carries over onto the new one (a freshly-selected page
# can render already scrolled halfway down). This detects an actual page
# change and smooth-scrolls back to top when — and only when — that
# happens, so switching pages doesn't yank the scroll on every rerun (e.g.
# ticking a checkbox on the *same* page shouldn't trigger it).
# ──────────────────────────────────────────────────────────────────────────
def scroll_to_top_on_page_change(current_page: str):
    """State lives on `window.parent` itself rather than any browser
    storage API: since Streamlit reruns patch the existing page rather
    than reloading it, that JS object persists across reruns for the
    whole browser tab, which is exactly the lifetime this needs."""
    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            const win = window.parent;
            const currentPage = {current_page!r};

            if (win.__fsLastPage === undefined) {{
                win.__fsLastPage = currentPage;  // first load — nothing to scroll away from
                return;
            }}
            if (win.__fsLastPage === currentPage) return;  // same page, some other widget changed
            win.__fsLastPage = currentPage;

            // Streamlit's actual scrolling element has moved around across
            // versions (window vs. an inner container), so try every likely
            // candidate rather than betting on one.
            const candidates = [
                doc.querySelector('[data-testid="stAppViewContainer"]'),
                doc.querySelector('[data-testid="stMain"]'),
                doc.querySelector('section.main'),
                doc.scrollingElement,
                doc.body,
                win,
            ];
            for (const el of candidates) {{
                if (el && typeof el.scrollTo === 'function') {{
                    el.scrollTo({{ top: 0, behavior: 'smooth' }});
                }}
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


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
            import torch
            import pyod  # noqa: F401

            # `layer1_autoencoder.pkl` was pickled on whatever machine ran
            # the training notebook. If that machine had a GPU, the tensors
            # inside the pickled PyOD model are tagged for a CUDA device —
            # and unpickling those tensors on a CPU-only host raises:
            #   RuntimeError: Attempting to deserialize object on a CUDA
            #   device but torch.cuda.is_available() is False.
            # This is unrelated to whether torch/pyod are installed; it's a
            # device mismatch between where the file was saved and where
            # it's being loaded. `torch.load(..., map_location="cpu")` is
            # the fix, but joblib.load doesn't expose a map_location param —
            # the CUDA tensors get unpickled deep inside joblib's call stack
            # via torch's own tensor-rebuild machinery, which itself goes
            # through torch.load. So we patch torch.load, just for the
            # duration of this one call, to always remap storages to CPU.
            _original_torch_load = torch.load

            def _cpu_mapped_load(f, *args, **kwargs):
                kwargs.setdefault("map_location", torch.device("cpu"))
                return _original_torch_load(f, *args, **kwargs)

            torch.load = _cpu_mapped_load
            try:
                layer1 = joblib.load(MODELS_DIR / "layer1_autoencoder.pkl")
            finally:
                torch.load = _original_torch_load  # don't leave the patch in place globally

            # The map_location patch above only remaps *tensor storages*
            # while unpickling — it does nothing for a plain attribute like
            # `self.device`. PyOD's AutoEncoder stores that as an ordinary
            # `torch.device` object, resolved once at fit time and pickled
            # as-is; since training ran on a GPU box, it survives unpickling
            # as literally `cuda:0` no matter what map_location says. Every
            # later call — `.predict()`, `.decision_function()` — does
            # `x.to(self.device)` internally, and on a host with no NVIDIA
            # driver at all, that's not a "no GPU found" — it's an attempt
            # to initialize CUDA itself, which fails harder:
            #   RuntimeError: Found no NVIDIA driver on your system...
            # Force every device-carrying attribute back to CPU explicitly.
            if hasattr(layer1, "device"):
                layer1.device = torch.device("cpu")
            if getattr(layer1, "model", None) is not None:
                layer1.model = layer1.model.to("cpu")

            engine_name = "Deep Autoencoder (PyOD)"
            engine_is_full = True
        except ModuleNotFoundError as e:
            st.warning(
                f"Couldn't load the Autoencoder Layer 1 model — `{e.name}` isn't installed "
                "in this environment. Falling back to Isolation Forest."
            )
            layer1 = joblib.load(MODELS_DIR / "layer1_isolation_forest.pkl")
            engine_name = "Isolation Forest (fallback)"
            engine_is_full = False
        except Exception as e:  # noqa: BLE001 — deliberately broad, this is a soft fallback
            st.warning(
                f"Couldn't load the Autoencoder Layer 1 model ({type(e).__name__}: {e}). "
                "Falling back to Isolation Forest."
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
# Layer 2's amount-aware thresholds (0.20 / 0.50) are the notebook's actual
# business thresholds and are reused as-is — they're a property of the
# classifier's calibrated probabilities, not of whichever Layer 1 engine
# is active.
#
# CUSTOM BANKING RISK COST
# Every verdict also carries a Risk_Cost figure, from the same cost-sensitive
# function used to pick the production threshold in the notebook:
#   Banking Risk Cost = (ALPHA × Missed Fraud Amount) + (BETA × False Alarm Count)
# Applied per-transaction:
#   - Blocked at Layer 1  -> $0.00, the fraud never reaches funds clearance
#   - Rejected at Layer 2 -> BETA, the fixed friction/support cost of a false alarm
#   - Approved            -> ALPHA * probability * amount, the expected loss if this
#                            approved transaction turns out to be fraud
# ──────────────────────────────────────────────────────────────────────────
ALPHA = 1.0   # cost multiplier on missed-fraud dollar amount
BETA = 50.0   # fixed operational cost of one false alarm (customer friction)


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
            "risk_cost": 0.0,
            "risk_cost_label": "Prevented",
        }

    probability = float(layer2_model.predict_proba(processed)[0, 1])
    threshold = 0.20 if tx_amount > 1000 else 0.50

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
            "risk_cost": BETA,
            "risk_cost_label": "Friction Cost",
        }

    residual_risk = ALPHA * probability * tx_amount
    return {
        "status": "APPROVED",
        "label": "✅ Approved",
        "reason": "Passed both structural and pattern-matching checks",
        "action": "Authorize secure funds clearance",
        "amount": tx_amount,
        "anomaly_score": layer1_score(layer1_model, processed),
        "probability": probability,
        "threshold": threshold,
        "risk_cost": residual_risk,
        "risk_cost_label": "Residual Risk",
    }


def safe_image(path: Path, **kwargs):
    """st.image, but a missing/empty/corrupt file degrades to a caption
    instead of crashing the whole script run. (Caught in testing: an empty
    placeholder PNG committed to images/ took down the entire app with an
    uncaught PIL.UnidentifiedImageError — st.image has no built-in guard
    against that.)

    Missing/empty files used to fail completely silently, which made a
    genuinely-missing asset in a deployed environment indistinguishable from
    "nothing to show here" — surface it instead so it's obvious which file
    needs to be re-added/re-uploaded (check: does the file exist in images/
    on the deployed host, is it non-zero bytes there, and does its filename
    match exactly, including case — Streamlit Cloud runs on case-sensitive
    Linux, so `KMeans.png` on disk won't match a `kmeans.png` reference)."""
    if not path.exists():
        st.caption(f"🖼️ Image not found: `{path.name}` — check it was deployed to `images/`.")
        return
    if path.stat().st_size == 0:
        st.caption(f"🖼️ Image file is empty: `{path.name}`.")
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

    thresholds = np.where(amounts > 1000, 0.20, 0.50)
    rejected = remaining & (probs >= thresholds)
    approved = remaining & ~rejected

    decision = np.select(
        [blocked, rejected, approved],
        ["Blocked (Layer 1)", "Rejected (Layer 2)", "Approved"],
        default="Approved",
    )

    # Custom Banking Risk Cost, same cost-sensitive function used to pick the
    # production threshold: Blocked -> $0 (prevented), Rejected -> BETA (fixed
    # friction cost), Approved -> ALPHA * probability * amount (expected loss
    # if this approved transaction turns out to be fraud).
    risk_cost = np.select(
        [blocked, rejected, approved],
        [0.0, BETA, ALPHA * np.nan_to_num(probs) * amounts],
        default=0.0,
    )

    out = df_in.copy()
    out["Anomaly_Score"] = scores
    out["Fraud_Probability"] = probs
    out["Decision"] = decision
    out["Risk_Cost"] = risk_cost
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
    scroll_to_top_on_page_change(page)

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
        "falling back to Isolation Forest automatically if it can't be loaded (missing dependencies "
        "or a device mismatch) — no manual switch, and this diagram doesn't change based on any selection."
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
# PAGE HEADLINE — changes with the sidebar nav selection, sits just below
# the static "Fraud Sentinel" brand hero above.
# ──────────────────────────────────────────────────────────────────────────
PAGE_HEADLINES = {
    PAGES[0]: ("⚡ Live Transaction Check", "Score a single transaction through both cascade layers in real time."),
    PAGES[1]: ("📊 Overview", "Dataset snapshot and production model health, at a glance."),
    PAGES[2]: ("📈 Model Performance", "Precision, recall, thresholds, and business cost — in depth."),
    PAGES[3]: ("🧠 Explainability", "Why the cascade makes the decisions it does."),
    PAGES[4]: ("📁 Batch Scoring", "Score an entire file of transactions at once."),
    PAGES[5]: ("ℹ️ How It Works", "The architecture behind the cascade, step by step."),
}
_headline, _subline = PAGE_HEADLINES[page]
st.markdown(
    f"""
    <div class="fs-page-banner fs-fade-in">
        <h2>{_headline}</h2>
        <p>{_subline}</p>
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


def apply_preset(values: dict):
    """Sets tx_values AND directly seeds every underlying widget key.
    Streamlit widgets that have a `key` ignore their `value=` argument on
    reruns once that key already exists in session_state — which is exactly
    why "Clear form" previously looked like it did nothing to the V-boxes:
    tx_values was updated, but the 28 number_input widgets (keyed v_V1..v_V28)
    already had their own stale session_state entries from the first render
    and kept showing those instead. Writing directly to those keys here,
    before the widgets are instantiated, is what actually resets them."""
    st.session_state["tx_values"] = dict(values)
    st.session_state["tx_time_input"] = float(values.get("Time", 0.0))
    st.session_state["tx_amount_input"] = float(values.get("Amount", 0.0))
    for _vcol in [f"V{n}" for n in range(1, 29)]:
        st.session_state[f"v_{_vcol}"] = float(values.get(_vcol, 0.0))

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
                apply_preset(MOCK_LEGIT)
            if preset_cols[1].button("📥 Load sample: fraudulent transaction", width='stretch'):
                apply_preset(MOCK_FRAUD)
            if preset_cols[2].button("🧹 Clear form", width='stretch'):
                apply_preset(ZERO_TX)

            defaults = st.session_state.get("tx_values", dict(MOCK_LEGIT))

            with st.form("transaction_form"):
                top_row = st.columns(2)
                time_val = top_row[0].number_input("Time (seconds since first transaction)", value=float(defaults.get("Time", 0.0)), step=1.0, key="tx_time_input")
                amount_val = top_row[1].number_input("Amount ($)", value=float(defaults.get("Amount", 0.0)), min_value=0.0, step=1.0, key="tx_amount_input")

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

            # Confidence = how far the classifier's probability sits from the
            # 50/50 midpoint, i.e. max(p, 1-p) — standard reading of a binary
            # probability as a certainty score. Layer 1 has no probability
            # scale to do the same math with, so it gets a qualitative label.
            if result["probability"] is not None:
                confidence_level = f"{max(result['probability'], 1 - result['probability']):.1%}"
            else:
                confidence_level = "High (structural anomaly)"

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
                        {prob_line} &nbsp;|&nbsp;
                        Banking risk cost: <b>${result['risk_cost']:,.2f}</b> ({result['risk_cost_label']})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            r1, r2, r3, r4 = st.columns(4)
            for col, label, value in zip(
                [r1, r2, r3, r4],
                ["Layer 1 Result", "Layer 2 Result", "Risk Level", "Confidence Level"],
                [layer1_result, layer2_result, risk_level, confidence_level],
            ):
                col.markdown(
                    f"""<div class="fs-card"><div class="fs-metric-label">{label}</div>
                    <div class="fs-metric-value">{value}</div></div>""",
                    unsafe_allow_html=True,
                )

            m1, m2, m3, m4, m5 = st.columns(5)
            for col, label, value in zip(
                [m1, m2, m3, m4, m5],
                ["Amount", "Layer 1 Anomaly Score", "Fraud Probability", "Decision Threshold", "Banking Risk Cost"],
                [
                    f"${result['amount']:,.2f}",
                    f"{result['anomaly_score']:.4f}",
                    f"{result['probability']:.2%}" if result["probability"] is not None else "N/A",
                    f"{result.get('threshold', 0):.0%}" if result.get("threshold") is not None else "—",
                    f"${result['risk_cost']:,.2f}",
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

    naive_accuracy = 284315 / 284807
    st.markdown(
        f"""
        <div class="fs-insight-card">
            <h4>💡 Why accuracy alone is misleading here</h4>
            <p>A model that predicts <b>"legitimate" for every single transaction</b> — never once
            looking at the data — would score <b>{naive_accuracy:.3%} accuracy</b> on this dataset,
            while catching exactly 0% of fraud. That's the trap of a 0.172% positive rate: accuracy
            barely moves no matter how bad the model is at the one thing that actually matters. It's
            why every metric on this app leans on <b>Precision, Recall, and PR-AUC</b> instead —
            they're the only ones that actually penalize missed fraud.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown("#### Dataset Behavior")
    behavior_charts = {
        "amount_distribution.png": ("Transaction Amount Distribution", "Overall distribution of transaction amounts across the dataset."),
        "fraud_by_hour.png": ("Fraud by Hour", "Fraud frequency by hour of day — temporal pattern in when fraud occurs."),
        "amount_comparison_boxplot.png": ("Fraud vs. Legitimate Amount", "Box-plot comparison of transaction amounts between fraud and legitimate classes."),
    }
    bcols = st.columns(3)
    for col, (fname, (title, caption)) in zip(bcols, behavior_charts.items()):
        with col:
            st.markdown(f"###### {title}")
            safe_image(IMAGES_DIR / fname, width='stretch')
            st.caption(caption)

    st.markdown("#### Production Model Snapshot")
    snap_cols = st.columns([3, 2])
    with snap_cols[0]:
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
            showlegend=False, height=380, **PLOT_BG,
        )
        st.plotly_chart(radar, width='stretch')
        st.caption("The five headline metrics for the production Layer 2 model, in one shape.")
    with snap_cols[1]:
        st.markdown(
            """<div class="fs-card"><div class="fs-metric-label">Model Status</div>
            <div class="fs-metric-value" style="font-size:1rem;">Calibrated CatBoost</div>
            <div style="color:#94a3b8; font-size:0.82rem; margin-top:0.3rem;">
                Calibration: Isotonic &nbsp;·&nbsp; Deployment: ✅ Production-ready
            </div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="fs-card" style="margin-top:0.7rem;"><div class="fs-metric-label">Recommended Operating Threshold</div>
            <div class="fs-metric-value" style="font-size:1rem;">0.20 (&gt;$1,000) · 0.50 (otherwise)</div>
            <div style="color:#94a3b8; font-size:0.82rem; margin-top:0.3rem;">
                Amount-aware: a missed high-value fraud costs far more than a false alarm, so
                large transactions are held to a lower bar to flag.
            </div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### Operational Risk Summary")
    metrics_path_ov = RESULTS_DIR / "metrics.csv"
    if metrics_path_ov.exists():
        df_metrics_ov = pd.read_csv(metrics_path_ov)
        if {"Model", "Recall_Fraud", "Precision_Fraud"}.issubset(df_metrics_ov.columns):
            prod_rows_ov = df_metrics_ov[df_metrics_ov["Model"].astype(str).str.contains("CatBoost", case=False, na=False)]
            prod_row_ov = prod_rows_ov.iloc[0] if len(prod_rows_ov) else df_metrics_ov.loc[df_metrics_ov["PR_AUC"].idxmax()]
            recall_ov = float(prod_row_ov["Recall_Fraud"])
            precision_ov = float(prod_row_ov["Precision_Fraud"])
            tp_ov = recall_ov * 492
            fp_ov = tp_ov * (1 - precision_ov) / precision_ov if precision_ov > 0 else 0.0

            o1, o2, o3 = st.columns(3)
            o1.markdown(
                f"""<div class="fs-card"><div class="fs-metric-label">False Alarm Rate</div>
                <div class="fs-metric-value">{fp_ov / 284315:.3%}</div>
                <div style="color:#94a3b8; font-size:0.8rem;">≈ {fp_ov:,.0f} legitimate transactions flagged</div></div>""",
                unsafe_allow_html=True,
            )
            o2.markdown(
                f"""<div class="fs-card"><div class="fs-metric-label">Fraud Capture Rate</div>
                <div class="fs-metric-value">{recall_ov:.2%}</div>
                <div style="color:#94a3b8; font-size:0.8rem;">≈ {tp_ov:,.0f} of 492 fraud cases caught</div></div>""",
                unsafe_allow_html=True,
            )
            impact_bits = []
            if "Banking_Risk_Cost_USD" in df_metrics_ov.columns:
                impact_bits.append(f"Estimated banking risk cost: <b>${float(prod_row_ov['Banking_Risk_Cost_USD']):,.0f}</b>")
            if "Missed_Fraud_Amount_USD" in df_metrics_ov.columns:
                impact_bits.append(f"Missed fraud exposure: <b>${float(prod_row_ov['Missed_Fraud_Amount_USD']):,.0f}</b>")
            o3.markdown(
                f"""<div class="fs-card"><div class="fs-metric-label">Business Impact</div>
                <div style="font-size:0.85rem; margin-top:0.4rem; line-height:1.6;">{"<br>".join(impact_bits) if impact_bits else "Add Banking_Risk_Cost_USD / Missed_Fraud_Amount_USD to results/metrics.csv for a dollar estimate."}</div></div>""",
                unsafe_allow_html=True,
            )
            st.caption("Full interactive version — with an adjustable threshold slider and cost assumptions — is on the Model Performance page.")
        else:
            st.info("Operational risk summary needs `Recall_Fraud` and `Precision_Fraud` columns in results/metrics.csv.")
    else:
        st.info(f"results/metrics.csv not found at {metrics_path_ov}.")

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
            "live (0.20 above $1,000, 0.50 otherwise) come from — high-value transactions get a "
            "lower bar to flag because a missed high-value fraud costs far more than a false alarm.",
        ),
    }
    for fname, (title, caption, detail) in perf_details.items():
        st.markdown(f"##### {title}")
        safe_image(IMAGES_DIR / fname, width='stretch')
        st.markdown(f'<div class="fs-chart-caption">{caption}</div>', unsafe_allow_html=True)
        with st.expander("📋 What this shows"):
            st.markdown(detail)

    st.markdown("#### Interactive Cost Simulator")
    if metrics_path.exists() and {"Model", "Recall_Fraud", "Precision_Fraud"}.issubset(df_metrics.columns):
        prod_rows = df_metrics[df_metrics["Model"].astype(str).str.contains("CatBoost", case=False, na=False)]
        prod_row = prod_rows.iloc[0] if len(prod_rows) else df_metrics.loc[df_metrics["PR_AUC"].idxmax()]

        TOTAL_FRAUD, TOTAL_LEGIT = 492, 284315
        prod_threshold = 0.35  # midpoint of the real amount-aware 0.20 / 0.50 split
        tp_prod = float(prod_row["Recall_Fraud"]) * TOTAL_FRAUD
        fn_prod = TOTAL_FRAUD - tp_prod
        precision = float(prod_row["Precision_Fraud"])
        fp_prod = tp_prod * (1 - precision) / precision if precision > 0 else 0.0
        default_fn_cost = (
            float(prod_row["Missed_Fraud_Amount_USD"]) / fn_prod
            if "Missed_Fraud_Amount_USD" in df_metrics.columns and fn_prod > 0 else 500.0
        )

        st.caption(
            "A simplified, interactive approximation — not a re-run of the model at every threshold. "
            "It interpolates false-positive and false-negative counts between three known points: "
            "flag-everything (threshold 0), flag-nothing (threshold 1), and the real production model's "
            "measured recall/precision at its actual operating threshold."
        )

        sim_cols = st.columns([2, 1, 1])
        sim_threshold = sim_cols[0].slider("Decision threshold", 0.0, 1.0, prod_threshold, 0.01)
        fp_cost = sim_cols[1].number_input("Avg. cost per false alarm ($)", value=50.0, min_value=0.0, step=5.0)
        fn_cost = sim_cols[2].number_input("Avg. loss per missed fraud ($)", value=round(default_fn_cost, 2), min_value=0.0, step=10.0)

        def _interp_counts(t):
            fp = np.interp(t, [0, prod_threshold, 1], [TOTAL_LEGIT, fp_prod, 0])
            fn = np.interp(t, [0, prod_threshold, 1], [0, fn_prod, TOTAL_FRAUD])
            return fp, fn

        cur_fp, cur_fn = _interp_counts(sim_threshold)
        cur_fp_cost, cur_fn_cost = cur_fp * fp_cost, cur_fn * fn_cost
        cur_total = cur_fp_cost + cur_fn_cost

        cc1, cc2, cc3 = st.columns(3)
        for col, label, value in zip(
            [cc1, cc2, cc3],
            ["False-Positive Cost", "False-Negative Loss", "Total Estimated Cost"],
            [f"${cur_fp_cost:,.0f} ({cur_fp:,.0f} alerts)", f"${cur_fn_cost:,.0f} ({cur_fn:,.0f} missed)", f"${cur_total:,.0f}"],
        ):
            col.markdown(
                f"""<div class="fs-card"><div class="fs-metric-label">{label}</div>
                <div class="fs-metric-value">{value}</div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="fs-cost-fn-card">
                <b>Custom Banking Risk Cost Function</b>
                <code>Total Cost(t) = FP(t) &times; ${fp_cost:,.0f} + FN(t) &times; ${fn_cost:,.0f}</code>
                <div style="color:#94a3b8; font-size:0.85rem; text-align:left;">
                    FP(t) is the number of false alarms at threshold t — annoyed legitimate customers and
                    review overhead. FN(t) is the number of missed fraud cases at threshold t — the actual
                    dollar loss that goes uncaught. Lower the threshold and more fraud gets caught but more
                    legitimate customers get flagged; raise it and false alarms drop but more fraud slips
                    through. The bank's real optimum sits wherever this curve is lowest — rarely at either extreme.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sweep = np.linspace(0, 1, 60)
        fp_sweep, fn_sweep = _interp_counts(sweep)
        total_sweep = fp_sweep * fp_cost + fn_sweep * fn_cost

        cost_cols = st.columns(2)
        with cost_cols[0]:
            line_fig = go.Figure()
            line_fig.add_trace(go.Scatter(x=sweep, y=total_sweep, mode="lines", line=dict(color="#facc15", width=3), name="Total cost"))
            line_fig.add_vline(x=sim_threshold, line_dash="dot", line_color="#3b82f6")
            line_fig.update_layout(height=340, xaxis_title="Decision threshold", yaxis_title="Estimated total cost ($)", **PLOT_BG)
            st.plotly_chart(line_fig, width='stretch')
            st.caption("Cost vs. threshold — the dotted line marks your current slider position.")
        with cost_cols[1]:
            sample_thresholds = sorted(set([0.1, 0.20, 0.35, 0.5, 0.75, round(sim_threshold, 2)]))
            fp_s, fn_s = _interp_counts(np.array(sample_thresholds))
            bd_long = pd.DataFrame({
                "Threshold": [f"{t:.2f}" for t in sample_thresholds] * 2,
                "Cost Type": ["False-Positive Cost"] * len(sample_thresholds) + ["False-Negative Loss"] * len(sample_thresholds),
                "Cost ($)": list(fp_s * fp_cost) + list(fn_s * fn_cost),
            })
            bd_fig = px.bar(
                bd_long, x="Threshold", y="Cost ($)", color="Cost Type", barmode="stack",
                color_discrete_map={"False-Positive Cost": "#3b82f6", "False-Negative Loss": "#ef4444"},
            )
            bd_fig.update_layout(height=340, **PLOT_BG)
            st.plotly_chart(bd_fig, width='stretch')
            st.caption("Cost breakdown at a few sample thresholds (including your current one) — shows how the FP/FN mix shifts.")
    else:
        st.info("The interactive cost simulator needs `Recall_Fraud` and `Precision_Fraud` columns (ideally also `Missed_Fraud_Amount_USD`) in results/metrics.csv.")

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
        "K-Means Clustering",
        "kmeans_clustering.png",
        "K-Means applied at the chosen k, visualizing how transactions group before any fraud label is used.",
        "Each color is one K-Means cluster in the reduced feature space. This is the direct follow-up to "
        "the silhouette analysis above — once k is chosen, this is what the resulting clusters actually look like.",
        "If fraud transactions consistently fall into the same handful of clusters, that's supporting "
        "evidence for the anomaly-gate design of Layer 1; if fraud is scattered evenly across clusters "
        "instead, it suggests fraud doesn't have one single 'shape' — reinforcing why Layer 2's supervised "
        "classifier does the heavier lifting rather than relying on clustering alone.",
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

                    b1, b2, b3, b4 = st.columns(4)
                    n_blocked = (scored["Decision"] == "Blocked (Layer 1)").sum()
                    n_rejected = (scored["Decision"] == "Rejected (Layer 2)").sum()
                    n_approved = (scored["Decision"] == "Approved").sum()
                    total_risk_cost = float(scored["Risk_Cost"].sum())
                    for col, label, value in zip(
                        [b1, b2, b3, b4],
                        ["Blocked — Layer 1", "Rejected — Layer 2", "Approved", "Total Banking Risk Cost"],
                        [f"{n_blocked:,}", f"{n_rejected:,}", f"{n_approved:,}", f"${total_risk_cost:,.2f}"],
                    ):
                        col.markdown(
                            f"""<div class="fs-card"><div class="fs-metric-label">{label}</div>
                            <div class="fs-metric-value">{value}</div></div>""",
                            unsafe_allow_html=True,
                        )
                    st.caption(
                        "Total Banking Risk Cost = Σ (Blocked → $0, Rejected → $50 friction cost, "
                        "Approved → probability × amount expected loss) across this batch."
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
            <p>Always tries the full Deep Autoencoder from the notebook first; falls back to an Isolation Forest automatically if it can't be loaded. Flags structural outliers before any label is consulted.</p>
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
            <p>Trained on SMOTE-balanced data, then wrapped with isotonic calibration so its probability outputs are trustworthy. Applies amount-aware thresholds: 0.20 above $1,000, 0.50 otherwise.</p>
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
    thr_curve = np.where(amt_range > 1000, 0.20, 0.50)
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
        "were trained and saved during the original notebook run, and either is a legitimate Layer 1 choice. "
        "Every verdict also carries a live Custom Banking Risk Cost (Blocked → $0, Rejected → $50 "
        "fixed friction cost, Approved → probability × amount expected loss) — the same cost function "
        "used offline in the Model Performance simulator, now computed per transaction."
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
