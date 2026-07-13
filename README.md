# Fraud Sentinel — Enterprise Credit Card Fraud Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-CatBoost%20%7C%20XGBoost%20%7C%20Autoencoder-green.svg)]()
[![App](https://img.shields.io/badge/App-Streamlit-red.svg)]()

## 📌 Project Overview

This repository implements a production-ready machine learning pipeline for detecting credit card fraud. Financial fraud detection is a highly imbalanced, time-dependent problem where the cost of missing a fraudulent transaction (a false negative) vastly outweighs the cost of a false alarm (a false positive).

To address that, this project moves beyond standard classification and implements a **Dual-Layer Cascade Architecture** combining unsupervised deep learning with supervised gradient boosting.

**[Live demo →](#)** *https://fraud-detection-farhan.streamlit.app/*

---

## ⚙️ Core Architecture: The Dual-Layer Engine

```
                     Incoming Transaction
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │   Layer 1: Unsupervised Autoencoder   │
        │   (structural anomaly check)          │
        └──────────────────┬─────────────────────┘
                            │
              High reconstruction error?
                    │               │
                   Yes              No
                    │               │
                    ▼               ▼
        🚨 BLOCKED: Zero-Day   ┌──────────────────────────────────────┐
           Anomaly              │  Layer 2: Calibrated CatBoost        │
                                 │  (historical pattern matching)       │
                                 └──────────────────┬─────────────────────┘
                                                    │
                                     Fraud probability ≥ threshold?
                                          │                │
                                         Yes               No
                                          │                │
                                          ▼                ▼
                              ❌ REJECTED: Matches     ✅ APPROVED
                                 Fraud Profile
```

1. **Layer 1 (Unsupervised Deep Autoencoder)** — trained strictly on legitimate transaction history, it learns the latent representation of normal spending behavior. A transaction with an abnormally high reconstruction error is immediately flagged as a zero-day / structural anomaly and routed for manual review, before it ever reaches Layer 2.
2. **Layer 2 (Calibrated CatBoost Classifier)** — transactions that clear Layer 1 are scored against historical fraud patterns. Raw boosting outputs are calibrated with Isotonic Regression so a "70% risk" prediction actually corresponds to a ~70% empirical fraud rate.

Both layers use **amount-aware dynamic thresholds** — a $6,000 transaction is held to a stricter bar than a $20 one, since the cost of missing it is proportionally higher.

---

## 🚀 Key Methodology & Features

### 1. Temporal Validation & Concept Drift Monitoring
- **Anti-leakage split:** a strictly chronological 80/20 split simulates deployment against genuinely future transactions, rather than an optimistic random split.
- **Population Stability Index (PSI):** quantifies feature drift between the training window and the live/test window.
  - High drift (PSI ≥ 0.2): `Time`, `V1`, `V3`, `V28`, `V11`, `V25`
  - Moderate drift (0.1 ≤ PSI < 0.2): `V15`, `V12`, `V5`, `V22`

### 2. Advanced Feature Engineering
- **Temporal indicators:** hour-of-day extraction plus an `Is_Night` flag (12 AM–5 AM window).
- **Non-linear transforms:** log-transformed amount (`Log_Amount`) and a `RobustScaler` to absorb extreme monetary outliers.
- **Interaction terms:** cross-multiplies the top predictive principal components (`V17`, `V14`, `V12`, `V10`, `V16`) with `Log_Amount`.

### 3. Cost-Sensitive Banking Risk Function
Standard metrics treat a missed $500,000 fraud the same as an accidental flag on a $10 coffee purchase. This pipeline instead scores every model on a custom financial utility function:

```
Banking Risk Cost = (α × Missed Fraud Amount) + (β × False Alarm Count)
```

- `α = 1.0` — direct financial loss multiplier
- `β = 50.0` — operational/support cost per customer-friction incident

---

## 📊 Dataset Profile

- **Total samples:** 284,807 transactions
- **Total features:** 31 (`Time`, `V1`–`V28` PCA components, `Amount`, `Class`)
- **Class balance:** extreme imbalance
  - Legit (`0`): 284,315 samples
  - Fraud (`1`): 492 samples (0.172%)

---

## 📈 Model Performance Matrix

### Supervised models
Evaluated on the chronological hold-out set (56,962 transactions, 75 ground-truth fraud cases):

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC | PR-AUC | Missed Cases | Missed $ | Banking Risk Cost |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CatBoost** | 0.9996 | 0.9048 | 0.7600 | 0.8261 | 0.9651 | 0.8095 | 18 | $1,661.42 | **$1,961.42** |
| Random Forest | 0.9995 | 0.8769 | 0.7600 | 0.8143 | 0.9774 | 0.8173 | 18 | $1,661.42 | $2,061.42 |
| XGBoost | 0.9994 | 0.7917 | 0.7600 | 0.7755 | 0.9772 | 0.7899 | 18 | $1,661.42 | $2,411.42 |
| Naive Bayes | 0.9991 | 0.9600 | 0.3200 | 0.4800 | 0.9626 | 0.6946 | 51 | $3,234.48 | $3,284.48 |
| KNN | 0.9983 | 0.4326 | 0.8133 | 0.5648 | 0.9128 | 0.6282 | 14 | $1,399.05 | $5,399.05 |
| Decision Tree | 0.9960 | 0.2226 | 0.8133 | 0.3496 | 0.9635 | 0.7024 | 14 | $1,426.65 | $12,076.65 |
| Neural Network | 0.9549 | 0.0252 | 0.8800 | 0.0489 | 0.9736 | 0.6648 | 9 | $1,322.39 | $129,222.39 |
| Logistic Regression | 0.9416 | 0.0209 | 0.9467 | 0.0409 | 0.9920 | 0.5792 | 4 | $371.00 | $166,471.00 |

**Isotonic-calibrated CatBoost** (production model): Precision 0.9483, Recall 0.7333, F1 0.8271, ROC-AUC 0.9793, PR-AUC 0.8042.

> ⚠️ Logistic Regression and the Neural Network post the highest *recall*, but their false-positive volume makes the Banking Risk Cost explode — CatBoost minimizes actual financial exposure, not just missed cases.

### Unsupervised models (structural anomaly detection)

| Model | PR-AUC | Missed Cases | Risk Cost |
|---|:---:|:---:|:---:|
| **Autoencoder** | 0.1870 | 60 | $5,692.79 |
| COPOD | 0.1097 | 67 | $6,637.02 |
| ECOD | 0.0759 | 74 | $7,084.75 |
| Isolation Forest | 0.0520 | 72 | $7,568.58 |
| Local Outlier Factor | 0.0111 | 71 | $16,121.49 |

---

## 🔮 Calibration & Explainable AI

- **Isotonic calibration** pulls raw CatBoost output probabilities toward true empirical fraud likelihood — necessary for cost-benefit thresholding and auditability.
- **SHAP (TreeExplainer)** shows `V14`, `V4`, and `V1` as the strongest global drivers of a fraud classification.
- **Permutation importance** confirms overall model performance drops most when `V14`, `V12`, and `V4` are shuffled — corroborating the SHAP ranking.

---

## 💻 Production Implementation

```python
def real_time_fraud_cascade_pipeline(raw_transaction_row, ae_model_instance, cb_pipeline_instance, processor_instance):
    tx_amount = raw_transaction_row['Amount'].values[0]
    processed_features = processor_instance.transform(raw_transaction_row)

    # LAYER 1: Unsupervised structural anomaly check
    anomaly_score = ae_model_instance.decision_function(processed_features)[0]
    unsupervised_trigger = 4.0 if tx_amount > 5000 else 5.0

    if anomaly_score > unsupervised_trigger:
        return {
            "Status": "🚨 BLOCKED BY LAYER 1",
            "Reason": "Extreme unsupervised structural anomaly",
            "Action Needed": "Route to cyber-forensics team (zero-day threat risk)",
            "Details": f"Amount: ${tx_amount:,.2f} | Anomaly Score: {anomaly_score:.4f}"
        }

    # LAYER 2: Calibrated historical pattern matching
    supervised_probability = cb_pipeline_instance.predict_proba(processed_features)[0, 1]
    supervised_threshold = 0.25 if tx_amount > 1000 else 0.50

    if supervised_probability >= supervised_threshold:
        return {
            "Status": "❌ REJECTED BY LAYER 2",
            "Reason": "Matches historical fraud behavior profile",
            "Action Needed": "Auto-decline transaction and flag credit profile",
            "Details": f"Amount: ${tx_amount:,.2f} | Confidence: {supervised_probability:.2%}"
        }

    return {
        "Status": "✅ APPROVED",
        "Reason": "Passed both structural and pattern-matching checks",
        "Action Needed": "Authorize secure funds clearance",
        "Details": f"Amount: ${tx_amount:,.2f} | Estimated Risk: {supervised_probability:.2%}"
    }
```

```python
import joblib
import pandas as pd

loaded_processor = joblib.load('models/fraud_processor.pkl')
loaded_ae = joblib.load('models/layer1_autoencoder.pkl')
loaded_cb = joblib.load('models/layer2_calibrated_catboost.pkl')

incoming_tx = pd.DataFrame([mock_data])
result = real_time_fraud_cascade_pipeline(incoming_tx, loaded_ae, loaded_cb, loaded_processor)
print(result)
```

---

## 🗂️ Repo Structure

```
credit-card-fraud-detection/
├── data/            # Dataset download instructions (raw CSV not committed)
├── src/             # Reusable pipeline: preprocessing, features, models, evaluation
├── models/          # Serialized artifacts (processor, autoencoder, calibrated CatBoost)
├── app/             # Streamlit app (Fraud Sentinel)
├── images/          # Saved plots for this README and the app
├── results/         # Metrics, feature importance, drift reports (CSV)
└── notebooks/       # Original exploratory notebook (EDA, PCA/t-SNE, K-Means)
```

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Download the dataset per `data/README.md`, then:

```bash
python train.py           # runs the full pipeline, saves artifacts to models/
streamlit run app/app.py  # launches Fraud Sentinel
```

> **Research stack vs. deployed stack:** the original research notebook used cuML and GPU-accelerated CatBoost/PyTorch (Skorch) for faster experimentation on Colab. The deployed pipeline in `src/models.py` uses CPU-only equivalents (scikit-learn, XGBoost, CatBoost CPU mode) so it trains and runs anywhere, including free-tier hosting like Streamlit Cloud.

---

## 🗺️ Future Roadmap

- **Real-time streaming** — Kafka/Spark integration for sub-millisecond edge validation.
- **Graph Neural Networks** — map transaction networks to catch coordinated, multi-node fraud rings.
- **Automated MLOps retraining** — trigger retraining/recalibration whenever PSI drift exceeds safe thresholds.

---
