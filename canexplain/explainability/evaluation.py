"""Quantitative evaluation of tensor network explanations and perturbation fidelity."""

from typing import Any

import numpy as np

from canexplain.explainability.feature_prob import compute_feature_marginal_probabilities
from canexplain.explainability.rdm import RDMExtractor
from canexplain.models.mps.born_machine import MPSBornMachine
from canexplain.preprocessing.encoding import trigonometric_feature_map


def evaluate_perturbation_sensitivity(
    born_machine: MPSBornMachine,
    benign_sample: np.ndarray,
    target_feature_idx: int,
    perturbation_magnitudes: list[float] | None = None,
) -> dict[str, Any]:
    """Tests explanation fidelity by injecting controlled feature perturbations.

    Scientific Principle:
        If a tensor-network attribution mechanism has high fidelity, perturbing a specific
        feature dimension should cause a monotonic decrease in its marginal probability P(v_k)
        and an increase in total anomaly Negative Log-Likelihood (NLL).

    Args:
        born_machine: Trained MPSBornMachine instance.
        benign_sample: 1D array of shape (N,) in [0, 1].
        target_feature_idx: Index of feature site to perturb.
        perturbation_magnitudes: List of shift amounts to test.

    Returns:
        Dictionary containing perturbation values, marginal probabilities, NLL scores,
        and fidelity flag.
    """
    if perturbation_magnitudes is None:
        perturbation_magnitudes = [0.1, 0.3, 0.5, 0.7, 0.9]
    if born_machine.mps is None:
        raise ValueError("Born machine contains no MPS.")

    extractor = RDMExtractor(born_machine.mps)
    rdms = [extractor.get_single_site_rdm(k) for k in range(born_machine.num_sites)]

    base_phi = trigonometric_feature_map(benign_sample)
    base_probs = compute_feature_marginal_probabilities(rdms, base_phi)
    base_nll = float(born_machine.score_samples(base_phi[None, :, :])[0])

    marginal_probs = [float(base_probs[target_feature_idx])]
    nll_scores = [base_nll]

    for delta in perturbation_magnitudes:
        pert_sample = benign_sample.copy()
        # Shift target feature and wrap/clamp in [0, 1]
        pert_sample[target_feature_idx] = float(
            np.clip(benign_sample[target_feature_idx] + delta, 0.0, 1.0)
        )

        pert_phi = trigonometric_feature_map(pert_sample)
        pert_probs = compute_feature_marginal_probabilities(rdms, pert_phi)
        pert_nll = float(born_machine.score_samples(pert_phi[None, :, :])[0])

        marginal_probs.append(float(pert_probs[target_feature_idx]))
        nll_scores.append(pert_nll)

    prob_drop = float(marginal_probs[0] - marginal_probs[-1])
    nll_increase = float(nll_scores[-1] - nll_scores[0])
    is_sensitive = prob_drop > 0.0 or nll_increase > 0.0

    return {
        "target_feature_idx": target_feature_idx,
        "perturbations": [0.0] + list(perturbation_magnitudes),
        "target_marginal_probabilities": marginal_probs,
        "nll_anomaly_scores": nll_scores,
        "marginal_prob_drop": prob_drop,
        "nll_increase": nll_increase,
        "is_sensitive": is_sensitive,
    }


def evaluate_explanation_stability(
    born_machine: MPSBornMachine,
    samples: np.ndarray,
) -> dict[str, float]:
    """Measures variance of feature marginal probabilities across a set of similar samples.

    Low variance indicates stable, reproducible explanation outputs across minor noise.
    """
    if born_machine.mps is None:
        raise ValueError("Born machine contains no MPS.")

    extractor = RDMExtractor(born_machine.mps)
    rdms = [extractor.get_single_site_rdm(k) for k in range(born_machine.num_sites)]

    all_probs = []
    for s in samples:
        phi = trigonometric_feature_map(s)
        p = compute_feature_marginal_probabilities(rdms, phi)
        all_probs.append(p)

    probs_arr = np.array(all_probs)  # (N_samples, N_features)
    feature_variances = np.var(probs_arr, axis=0)

    return {
        "mean_feature_variance": float(np.mean(feature_variances)),
        "max_feature_variance": float(np.max(feature_variances)),
        "min_feature_variance": float(np.min(feature_variances)),
    }
