# Fraud Sentinel — Enterprise Credit Card Fraud Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-CatBoost%20%7C%20XGBoost%20%7C%20Autoencoder-green.svg)]()
[![App](https://img.shields.io/badge/App-Streamlit-red.svg)]()

## 📌 Project Overview

This repository implements a production-ready machine learning pipeline for detecting credit card fraud. Financial fraud detection is a highly imbalanced, time-dependent problem where the cost of missing a fraudulent transaction (a false negative) vastly outweighs the cost of a false alarm (a false positive).

To address that, this project moves beyond standard classification and implements a **Dual-Layer Cascade Architecture** combining unsupervised deep learning with supervised gradient boosting.

**[Live demo →](#)** *(add your Streamlit Cloud link here)*

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

## License

See `LICENSE`. Dataset license and citation are documented separately in `data/README.md`.Unsupervised Engine Comparisons
For structural filtering capabilities, deep autoencoders drastically outmatched traditional outlier metrics:

Autoencoder: PR-AUC: 0.1870 | Missed Cases: 60 | Operational Risk Cost: $5,692.79

COPOD: PR-AUC: 0.1097 | Missed Cases: 67 | Operational Risk Cost: $6,637.02

ECOD: PR-AUC: 0.0759 | Missed Cases: 74 | Operational Risk Cost: $7,084.75

Isolation Forest: PR-AUC: 0.0520 | Missed Cases: 72 | Operational Risk Cost: $7,568.58

Local Outlier Factor: PR-AUC: 0.0111 | Missed Cases: 71 | Operational Risk Cost: $16,121.49

🔮 Model Calibration & Explainable AI (XAI)
Isotonic Calibration
To ensure predicted risk percentages represent real-world probabilities, we apply Isotonic Regression. This pushes the model output to align closely with perfect calibration curves, boosting accuracy and safety for automated high-dollar decisions.

Global & Local Interpretability
SHAP TreeExplainer Analysis: Uncovers top model features. High values in V14, V1 and low values in V4 show strong correlations toward fraud classifications.

Permutation Importance: Proves that structural consistency collapses most when global attributes like V14, V12, and V4 are shuffled.

💻 Production Implementation
Inference Pipeline Script
Python
import numpy as np
import pandas as pd

def real_time_fraud_cascade_pipeline(raw_transaction_row, ae_model_instance, cb_pipeline_instance, processor_instance):
    # Extract structural cash details
    tx_amount = raw_transaction_row['Amount'].values[0]
    
    # Run through full transformation space
    processed_features = processor_instance.transform(raw_transaction_row)
    
    # LAYER 1: Deep Structural Filtering
    anomaly_score = ae_model_instance.decision_function(processed_features)[0]
    unsupervised_trigger = 4.0 if tx_amount > 5000 else 5.0
    
    if anomaly_score > unsupervised_trigger:
        return {
            "Status": "🚨 BLOCKED BY LAYER 1",
            "Reason": "Extreme Unsupervised Structural Anomaly",
            "Action Needed": "Route immediately to cyber-forensics team (Zero-Day Threat Risk)",
            "Details": f"Amount: ${tx_amount:,.2f} | Anomaly Score: {anomaly_score:.4f}"
        }
        
    # LAYER 2: Calibrated Historical Machine Learning Pattern Recognition
    supervised_probability = cb_pipeline_instance.predict_proba(processed_features)[0, 1]
    supervised_threshold = 0.25 if tx_amount > 1000 else 0.50
    
    if supervised_probability > supervised_threshold:
        return {
            "Status": "❌ REJECTED BY LAYER 2",
            "Reason": "Matches Historical Fraud Behavior Profile",
            "Action Needed": "Auto-decline transaction and flag user credit profile",
            "Details": f"Amount: ${tx_amount:,.2f} | CatBoost Confidence: {supervised_probability:.2%}"
        }
        
    return {
        "Status": "✅ APPROVED",
        "Reason": "Transaction passed both structural and pattern-matching criteria",
        "Action Needed": "Authorize secure funds clearance",
        "Details": f"Amount: ${tx_amount:,.2f} | Estimated Risk Profile: {supervised_probability:.2%}"
    }
Loading the Production Environment
Python
import joblib

# Load serialized components
loaded_processor = joblib.load('Saved_Models/fraud_processor.pkl')
loaded_ae = joblib.load('Saved_Models/layer1_autoencoder.pkl')
loaded_cb = joblib.load('Saved_Models/layer2_calibrated_catboost.pkl')

# Mock transaction ingestion
incoming_tx = pd.DataFrame([mock_data])
result = real_time_fraud_cascade_pipeline(incoming_tx, loaded_ae, loaded_cb, loaded_processor)
print(result)
🗺️ Future Roadmap
High-Throughput Streaming: Move inference loop logic directly into Apache Kafka and Apache Spark Streams for sub-millisecond edge validations.

Graph Neural Networks (GNN): Build transaction network maps to capture interconnected device usage, rapid peer-to-peer loops, and multi-node organized financial rings.

MLOps Drift Triggers: Set up automated pipelines that trigger updates when the Population Stability Index (PSI) exceeds safe limits.

Can youwrite this in full code style so ican copy paste in github.?

Here is the complete, raw Markdown code for the README.md file. You can copy the entire code block below and paste it directly into your GitHub repository.

Markdown
# Enterprise Credit Card Fraud Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-cuml%20%7C%20PyTorch%20%7C%20Skorch-orange.svg)]()
[![Model](https://img.shields.io/badge/Model-CatBoost%20%7C%20XGBoost%20%7C%20Autoencoder-green.svg)]()

## 📌 Project Overview
[cite_start]This repository implements a production-ready, highly robust Machine Learning pipeline for detecting credit card fraud[cite: 3]. [cite_start]Financial fraud detection is a highly imbalanced, time-dependent problem where the cost of missing a fraudulent transaction (False Negative) vastly outweighs the cost of a false alarm (False Positive)[cite: 4]. 

[cite_start]To address these real-world business constraints, this project moves beyond standard classification and implements a Dual-Layer Cascade Architecture combining Unsupervised Deep Learning and Supervised Gradient Boosting[cite: 5].

---

## ⚙️ Core Architecture: The Dual-Layer Engine

[ Incoming Transaction ]
│
▼
┌──────────────────────────────────────┐
│  Layer 1: Unsupervised Autoencoder   │ ──► High Reconstruction Error ──► [ BLOCKED: Zero-Day Anomaly ]
└──────────────────────────────────────┘
│
▼ (Passed Structural Filter)
┌──────────────────────────────────────┐
│  Layer 2: Calibrated CatBoost / RF   │ ──► Dynamic Cost-Aware Threshold ──► [ REJECTED: Fraud Profile ]
└──────────────────────────────────────┘
│
▼
[ APPROVED ]


1. **Layer 1 (Unsupervised Deep Autoencoder):** Acts as a structural filter[cite: 18]. It learns the latent representation of legitimate transactions[cite: 18]. If an incoming transaction has a massive reconstruction error, it is immediately flagged as a Zero-Day/Structural Anomaly[cite: 19].
2. **Layer 2 (Calibrated CatBoost Classifier):** For transactions that pass Layer 1, this model uses historical pattern-matching to predict fraud probabilities[cite: 20]. The probabilities are calibrated using Isotonic Regression for extreme reliability[cite: 21].

---

## 🚀 Key Highlights & Methodology

### 1. Temporal Validation & Concept Drift Monitoring
* **Anti-Data Leakage Split:** Avoids data-leakage by using a chronological time-based split rather than random splitting[cite: 8].
* **Population Stability Index (PSI):** Implements PSI to monitor feature drift and ensure the model remains reliable as transaction patterns evolve over time[cite: 9].
  * *High Drift ($PSI \ge 0.2$):* Alert triggered (`Time`, `V1`, `V3`, `V28`, `V11`, `V25` showed high drift)[cite: 223, 253].
  * *Moderate Drift ($0.1 \le PSI < 0.2$):* Flagged for monitoring (`V15`, `V12`, `V5`, `V22`)[cite: 222, 253].

### 2. Advanced Feature Engineering
* **Temporal Indicators:** Derives temporal features (e.g., time of day, night-time transaction flags)[cite: 11].
* **Non-Linear Transformations:** Applies non-linear transformations and robust scaling to handle extreme monetary outliers[cite: 12].
* **Cross Interaction Terms:** Generates interaction features between the most predictive principal components and transaction amounts[cite: 13].

### 3. Cost-Sensitive Business Evaluation
Standard metrics treat all errors equally, but this pipeline evaluates models not just on F1-Score or ROC-AUC, but on PR-AUC (Precision-Recall) and a custom Banking Risk Cost Function[cite: 15, 471, 473]. It calculates exact dollar amounts at risk to balance the financial penalty of missed fraud against customer friction from false alarms[cite: 16].

$$\text{Banking Risk Cost} = (\alpha \times \text{Missed Fraud Amount}) + (\beta \times \text{False Alarms Count})$$

* $\alpha = 1.0$ (Direct financial loss multiplier) [cite: 476]
* $\beta = 50.0$ (Operational support cost per customer friction incident) [cite: 477]

---

## 📊 Dataset Profile

* **Total Samples:** 284,807 transactions [cite: 61, 80]
* **Total Features:** 31 (`Time`, `V1`–`V28`, `Amount`, `Class`) [cite: 60, 62, 80]
* **Class Disparity:** Extreme Class Imbalance Check [cite: 81, 97]
  * **Legit Class (`0`):** 284,315 samples [cite: 86, 90]
  * **Fraud Class (`1`):** 492 samples (0.172%) [cite: 87, 91]

---

## 📈 Model Performance Matrix

### Supervised Models Evaluation
Evaluated strictly on the future simulation horizon (56,962 transactions; 75 ground-truth fraud instances)[cite: 215, 218, 276]:

| Supervised Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | ROC-AUC | PR-AUC (AP) | Missed Fraud Cases | Missed Fraud Amount | Banking Risk Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost** | 0.9996 | 0.9048 | 0.7600 | 0.8261 | 0.9651 | 0.8095 | 18 | \$1,661.42 | **\$1,961.42** |
| **Random Forest** | 0.9995 | 0.8769 | 0.7600 | 0.8143 | 0.9774 | 0.8173 | 18 | \$1,661.42 | \$2,061.42 |
| **XGBoost** | 0.9994 | 0.7917 | 0.7600 | 0.7755 | 0.9772 | 0.7899 | 18 | \$1,661.42 | \$2,411.42 |
| **Naive Bayes** | 0.9991 | 0.9600 | 0.3200 | 0.4800 | 0.9626 | 0.6946 | 51 | \$3,234.48 | \$3,284.48 |
| **KNN** | 0.9983 | 0.4326 | 0.8133 | 0.5648 | 0.9128 | 0.6282 | 14 | \$1,399.05 | \$5,399.05 |
| **Decision Tree** | 0.9960 | 0.2226 | 0.8133 | 0.3496 | 0.9635 | 0.7024 | 14 | \$1,426.65 | \$12,076.65 |
| **Neural Network** | 0.9549 | 0.0252 | 0.8800 | 0.0489 | 0.9736 | 0.6648 | 9 | \$1,322.39 | \$129,222.39 |
| **Logistic Regression** | 0.9416 | 0.0209 | 0.9467 | 0.0409 | 0.9920 | 0.5792 | 4 | \$371.00 | \$166,471.00 |

*(Note: All results extracted from source table [cite: 573])*

### Unsupervised Engine Comparisons
For structural filtering capabilities, deep autoencoders outmatched traditional geometric and statistical outlier algorithms[cite: 1212, 1213]:

* **Autoencoder:** PR-AUC: `0.1870` | Missed Cases: `60` | Operational Risk Cost: **\$5,692.79** [cite: 1212]
* **COPOD:** PR-AUC: `0.1097` | Missed Cases: `67` | Operational Risk Cost: \$6,637.02 [cite: 1212]
* **ECOD:** PR-AUC: `0.0759` | Missed Cases: `74` | Operational Risk Cost: \$7,084.75 [cite: 1212]
* **Isolation Forest:** PR-AUC: `0.0520` | Missed Cases: `72` | Operational Risk Cost: \$7,568.58 [cite: 1212]
* **Local Outlier Factor:** PR-AUC: `0.0111` | Missed Cases: `71` | Operational Risk Cost: \$16,121.49 [cite: 1212]

---

## 🔮 Model Calibration & Explainable AI (XAI)

### Isotonic Calibration
Fraud systems require trustworthy probabilities for cost-benefit analysis and dynamic thresholding[cite: 838]. Isotonic Regression calibrates raw probabilities from the CatBoost model, ensuring the predicted probability tightly matches the true empirical likelihood of fraud[cite: 839].

### Global & Local Interpretability
* **SHAP Analysis:** Integrates SHAP (SHapley Additive exPlanations) to ensure every model decision is transparent and interpretable[cite: 23]. Features like `V14`, `V4`, and `V1` act as core drivers[cite: 1007].
* **Permutation Importance:** Shows how shuffling each feature affects overall predictive performance [cite: 1015], revealing that global performance drops heavily when features like `V14` and `V12` are altered[cite: 1033].

---

## 💻 Production Implementation

### Inference Pipeline Script
```python
import numpy as np
import pandas as pd

def real_time_fraud_cascade_pipeline(raw_transaction_row, ae_model_instance, cb_pipeline_instance, processor_instance):
    # Extract the true transaction currency value for financially-aware logic
    tx_amount = raw_transaction_row['Amount'].values[0]
    
    # Prepare raw incoming data using your pre-defined preprocessing pipeline
    processed_features = processor_instance.transform(raw_transaction_row)
    
    # LAYER 1: Compute Unsupervised Anomaly Risk Factor
    anomaly_score = ae_model_instance.decision_function(processed_features)[0]
    
    # Dynamic Financial Trigger: Adjust sensitivity based on monetary weight exposure
    unsupervised_trigger = 4.0 if tx_amount > 5000 else 5.0
    
    if anomaly_score > unsupervised_trigger:
        return {
            "Status": "🚨 BLOCKED BY LAYER 1",
            "Reason": "Extreme Unsupervised Structural Anomaly",
            "Action Needed": "Route immediately to cyber-forensics team (Zero-Day Threat Risk)",
            "Details": f"Amount: ${tx_amount:,.2f} | Anomaly Score: {anomaly_score:.4f}"
        }
        
    # LAYER 2: Supervised Historical Classification Checks
    supervised_probability = cb_pipeline_instance.predict_proba(processed_features)[0, 1]
    
    # Cost-Aware Safety Threshold Matrix
    if tx_amount > 1000:
        supervised_threshold = 0.25 # High-value exposure makes system ultra-sensitive
    else:
        supervised_threshold = 0.50 # Balanced operational threshold for small charges
        
    if supervised_probability > supervised_threshold:
        return {
            "Status": "❌ REJECTED BY LAYER 2",
            "Reason": "Matches Historical Fraud Behavior Profile",
            "Action Needed": "Auto-decline transaction and flag user credit profile",
            "Details": f"Amount: ${tx_amount:,.2f} | Confidence: {supervised_probability:.2%}"
        }
        
    # All Good! Approve...
    return {
        "Status": "✅ APPROVED",
        "Reason": "Transaction passed both structural and pattern-matching criteria",
        "Action Needed": "Authorize secure funds clearance",
        "Details": f"Amount: ${tx_amount:,.2f} | Estimated Risk Profile: {supervised_probability:.2%}"
    }
