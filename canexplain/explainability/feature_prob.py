"""Marginal feature probabilities extraction from single-site RDMs."""

import numpy as np


def compute_feature_marginal_probabilities(
    rdms: list[np.ndarray],
    phi_window: np.ndarray,
) -> np.ndarray:
    """Computes marginal feature probabilities P(v_k) = <phi(v_k) | rho_k | phi(v_k)>.

    Args:
        rdms: List of single-site RDMs rho_k of shape (p, p) for k = 0..N-1.
        phi_window: Array of shape (N, p) containing local state vectors for each site.

    Returns:
        Array of shape (N,) containing marginal probabilities in [0, 1].
    """
    num_sites = len(rdms)
    probs = np.zeros(num_sites, dtype=np.float64)

    for k in range(num_sites):
        rho_k = rdms[k]
        phi_k = phi_window[k]
        # Quadratic form: phi_k^T * rho_k * phi_k
        prob_k = float(np.dot(phi_k, np.dot(rho_k, phi_k)))
        probs[k] = np.clip(prob_k, 1e-12, 1.0)

    return probs


def isolate_anomalous_features(
    feature_probs: np.ndarray,
    baseline_probs: np.ndarray,
    feature_names: list[str],
    drop_threshold_ratio: float = 0.5,
) -> list[tuple[str, float, float]]:
    """Identifies features that exhibit significant probability drops relative to baseline.

    Returns:
        List of tuples: (feature_name, observed_probability, baseline_probability)
    """
    anomalous = []
    for i, name in enumerate(feature_names):
        obs = feature_probs[i]
        base = baseline_probs[i]
        if obs < base * drop_threshold_ratio:
            anomalous.append((name, float(obs), float(base)))
    return anomalous
