"""Unit tests for Reduced Density Matrices, von Neumann entropy, and mutual information."""

import pytest
import numpy as np

from canexplain.models.mps.tensor_core import MatrixProductState
from canexplain.explainability.rdm import RDMExtractor
from canexplain.explainability.entropy import von_neumann_entropy, compute_all_site_entropies
from canexplain.explainability.feature_prob import compute_feature_marginal_probabilities
from canexplain.explainability.mutual_info import compute_mutual_information_matrix
from canexplain.preprocessing.encoding import trigonometric_feature_map


def test_rdm_extractor_single_site():
    num_sites = 6
    mps = MatrixProductState.initialize_random(num_sites=num_sites, physical_dim=2, bond_dimension=8)
    extractor = RDMExtractor(mps)

    for k in range(num_sites):
        rho_k = extractor.get_single_site_rdm(k)
        assert rho_k.shape == (2, 2)
        # Trace must equal 1.0
        np.testing.assert_allclose(np.trace(rho_k), 1.0, atol=1e-10)
        # Hermitian
        np.testing.assert_allclose(rho_k, rho_k.T, atol=1e-10)
        # Positive semi-definite
        eigenvals = np.linalg.eigvalsh(rho_k)
        assert np.all(eigenvals >= -1e-10)


def test_von_neumann_entropy():
    # Pure state: [1, 0], [0, 0] -> entropy = 0
    rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]])
    s_pure = von_neumann_entropy(rho_pure)
    np.testing.assert_allclose(s_pure, 0.0, atol=1e-10)

    # Maximally mixed state: 0.5 * I -> entropy = -2 * 0.5 * log2(0.5) = 1.0 bit
    rho_mixed = 0.5 * np.eye(2)
    s_mixed = von_neumann_entropy(rho_mixed)
    np.testing.assert_allclose(s_mixed, 1.0, atol=1e-10)


def test_feature_marginal_probabilities():
    num_sites = 4
    mps = MatrixProductState.initialize_random(num_sites=num_sites, physical_dim=2, bond_dimension=6)
    extractor = RDMExtractor(mps)
    rdms = [extractor.get_single_site_rdm(k) for k in range(num_sites)]

    sample = np.array([0.0, 0.5, 0.8, 1.0])
    phi_window = trigonometric_feature_map(sample)

    probs = compute_feature_marginal_probabilities(rdms, phi_window)
    assert len(probs) == num_sites
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_mutual_information_matrix():
    num_sites = 4
    mps = MatrixProductState.initialize_random(num_sites=num_sites, physical_dim=2, bond_dimension=6)
    extractor = RDMExtractor(mps)

    mi_matrix = compute_mutual_information_matrix(extractor)
    assert mi_matrix.shape == (num_sites, num_sites)
    # Symmetric
    np.testing.assert_allclose(mi_matrix, mi_matrix.T, atol=1e-10)
    # Non-negative
    assert np.all(mi_matrix >= -1e-10)
