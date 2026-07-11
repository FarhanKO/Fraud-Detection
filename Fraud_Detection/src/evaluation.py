import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix,
)

from src.utils import ALPHA, BETA


def get_eval_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC (AP)": average_precision_score(y_true, y_prob),
    }


def banking_risk_cost(y_true, y_pred, amounts, alpha=ALPHA, beta=BETA):
    """Cost = alpha * (missed fraud $) + beta * (false alarm count)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    missed_mask = (np.asarray(y_true) == 1) & (np.asarray(y_pred) == 0)
    missed_amount = float(np.sum(np.asarray(amounts)[missed_mask]))
    cost = alpha * missed_amount + beta * fp
    return {
        "Missed Fraud Cases": int(fn),
        "Missed Fraud Amount ($)": missed_amount,
        "False Alarms": int(fp),
        "Banking Risk Cost ($)": cost,
    }


def evaluate_supervised_models(model_results, X_test, y_test, amounts):
    """Full comparison table (metrics + banking cost) for every fitted model."""
    rows = []
    for name, grid in model_results.items():
        best_model = grid.best_estimator_ if hasattr(grid, "best_estimator_") else grid
        preds = best_model.predict(X_test)
        probas = best_model.predict_proba(X_test)[:, 1]

        row = {"Model": name}
        row.update(get_eval_metrics(y_test, preds, probas))
        row.update(banking_risk_cost(y_test, preds, amounts))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def plot_roc_pr_curves(model_results, X_test, y_test, save_path=None):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 6))
    for name, grid in model_results.items():
        best_model = grid.best_estimator_ if hasattr(grid, "best_estimator_") else grid
        probas = best_model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probas)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probas):.3f})")

        precision, recall, _ = precision_recall_curve(y_test, probas)
        ax_pr.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, probas):.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title("ROC Curve"); ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.legend(fontsize=8)
    ax_pr.set_title("Precision-Recall Curve"); ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(y_test, probas, threshold=0.5, title="Confusion Matrix", save_path=None):
    preds = np.where(probas >= threshold, 1, 0)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_title(f"{title} (threshold={threshold})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_calibration_curve(y_test, raw_probas, calib_probas, save_path=None):
    from sklearn.calibration import calibration_curve
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, raw_probas, n_bins=10, strategy="quantile")
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, calib_probas, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred_raw, prob_true_raw, "o-", color="red", label="Before calibration")
    ax.plot(prob_pred_cal, prob_true_cal, "s-", color="green", label="After calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def shap_summary(model_object, X_sample, feature_names, save_path=None):
    """SHAP beeswarm summary plot for a fitted tree-based classifier."""
    import shap
    explainer = shap.TreeExplainer(model_object)
    shap_values = explainer(X_sample)
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
