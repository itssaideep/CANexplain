"""Unit tests for evaluation metrics and resource profiling."""

import pytest
import numpy as np

from canexplain.evaluation.metrics import compute_ids_metrics
from canexplain.evaluation.benchmark import ResourceProfiler


def test_compute_ids_metrics():
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.array([0] * 88 + [1] * 2 + [0] * 2 + [1] * 8)
    y_scores = np.linspace(0.0, 1.0, 100)

    metrics = compute_ids_metrics(y_true, y_pred, y_scores)

    assert "f1_weighted" in metrics
    assert "g_mean" in metrics
    assert "fpr" in metrics
    assert "tpr" in metrics
    assert "pr_auc" in metrics

    assert 0.0 <= metrics["fpr"] <= 1.0
    assert 0.0 <= metrics["tpr"] <= 1.0
    assert 0.0 <= metrics["g_mean"] <= 1.0
    assert metrics["tp"] == 8
    assert metrics["fp"] == 2
    assert metrics["tn"] == 88
    assert metrics["fn"] == 2


def test_resource_profiler():
    dummy_windows = np.random.rand(10, 32, 11)

    def dummy_score_fn(w):
        return np.mean(w, axis=(1, 2))

    prof = ResourceProfiler.profile_inference(dummy_score_fn, dummy_windows, num_warmup=2)
    assert "latency_ms_per_window" in prof
    assert prof["latency_ms_per_window"] >= 0.0
    assert prof["num_windows_evaluated"] == 10

    def dummy_train_fn():
        arr = np.ones((500, 500))
        return arr.sum()

    _, train_prof = ResourceProfiler.profile_training(dummy_train_fn)
    assert "training_time_sec" in train_prof
    assert "peak_memory_mb" in train_prof
    assert train_prof["peak_memory_mb"] > 0.0
