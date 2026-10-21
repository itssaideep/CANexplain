"""Unit tests for explanation quality, perturbation sensitivity, and explanation stability."""

import numpy as np

from canexplain.explainability.evaluation import (
    evaluate_explanation_stability,
    evaluate_perturbation_sensitivity,
)
from canexplain.models.mps.born_machine import MPSBornMachine
from canexplain.models.mps.dmrg import HeuristicCorrelationOptimizer
from canexplain.preprocessing.encoding import trigonometric_feature_map


def test_perturbation_sensitivity_fidelity():
    num_sites = 6
    # Train on benign samples tightly clustered around 0.1
    benign_data = np.clip(np.random.normal(0.1, 0.02, size=(50, num_sites)), 0.0, 1.0)
    phi_train = trigonometric_feature_map(benign_data)

    bm = MPSBornMachine(num_sites=num_sites, physical_dim=2, bond_dimension=8, seed=42)
    opt = HeuristicCorrelationOptimizer(learning_rate=0.08, num_sweeps=3)
    opt.fit(bm, phi_train)

    # Test baseline sample near center of distribution
    test_sample = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    target_idx = 2  # site 2 (e.g. byte_0)

    res = evaluate_perturbation_sensitivity(
        born_machine=bm,
        benign_sample=test_sample,
        target_feature_idx=target_idx,
        perturbation_magnitudes=[0.2, 0.4, 0.6, 0.8],
    )

    assert res["target_feature_idx"] == 2
    assert len(res["nll_anomaly_scores"]) == 5  # 0.0 + 4 steps
    # Perturbing away from 0.1 towards 0.9 must increase the anomaly NLL
    assert res["nll_anomaly_scores"][-1] > res["nll_anomaly_scores"][0]
    assert res["is_sensitive"] is True


def test_explanation_stability():
    num_sites = 5
    bm = MPSBornMachine(num_sites=num_sites, physical_dim=2, bond_dimension=6, seed=42)

    # Similar samples drawn from tight distribution
    samples = np.clip(np.random.normal(0.2, 0.01, size=(20, num_sites)), 0.0, 1.0)
    metrics = evaluate_explanation_stability(bm, samples)

    assert "mean_feature_variance" in metrics
    # Variances across tight samples should be small
    assert metrics["mean_feature_variance"] < 0.05
