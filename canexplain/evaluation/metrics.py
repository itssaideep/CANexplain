"""Comprehensive evaluation metrics for imbalanced automotive intrusion detection."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_ids_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
    eval_unit: str = "window",
) -> dict[str, Any]:
    """Computes comprehensive IDS evaluation metrics compliant with can-sleuth.

    Handles extreme class imbalance via explicit attack-class binary metrics,
    class prevalence reporting, and confusion counts.

    Args:
        y_true: Ground truth binary labels (0 = normal, 1 = attack).
        y_pred: Predicted binary labels (0 = normal, 1 = attack).
        y_scores: Continuous anomaly scores or probabilities (optional).
        eval_unit: Explicit unit of evaluation ('window' or 'frame').

    Returns:
        Dictionary of computed metric floats and metadata.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    total_count = len(y_true)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # False Positive Rate
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # True Positive Rate (Sensitivity / Recall)
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # True Negative Rate (Specificity)
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0

    # G-mean = sqrt(TPR * TNR)
    g_mean = float(np.sqrt(tpr * tnr))

    # Weighted metrics
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Binary metrics for the attack class (essential under class imbalance)
    prec_binary = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    rec_binary = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    f1_binary = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    accuracy = float(accuracy_score(y_true, y_pred))
    attack_prevalence = float(np.mean(y_true == 1)) if total_count > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "precision_binary": prec_binary,
        "recall_binary": rec_binary,
        "f1_binary": f1_binary,
        "g_mean": g_mean,
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_samples": int(total_count),
        "attack_prevalence": attack_prevalence,
        "eval_unit": eval_unit,
    }

    # Area under curves if continuous scores provided
    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_scores))
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            metrics["pr_auc"] = 0.0
            metrics["roc_auc"] = 0.5
    else:
        metrics["pr_auc"] = 0.0
        metrics["roc_auc"] = 0.5

    return metrics
