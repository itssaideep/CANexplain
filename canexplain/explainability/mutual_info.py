"""Pairwise mutual information extraction from two-site RDMs."""

import numpy as np

from canexplain.explainability.rdm import RDMExtractor
from canexplain.explainability.entropy import von_neumann_entropy


def compute_mutual_information_matrix(rdm_extractor: RDMExtractor) -> np.ndarray:
    """Computes the N x N pairwise mutual information matrix: I(i; j) = S_i + S_j - S_{i, j}.

    Args:
        rdm_extractor: Initialized RDMExtractor.

    Returns:
        Symmetric array of shape (N, N) where MI[i, j] is the mutual information in bits.
    """
    n_sites = rdm_extractor.num_sites
    mi_matrix = np.zeros((n_sites, n_sites), dtype=np.float64)

    # Precompute single-site entropies
    single_entropies = np.zeros(n_sites, dtype=np.float64)
    for i in range(n_sites):
        rho_i = rdm_extractor.get_single_site_rdm(i)
        single_entropies[i] = von_neumann_entropy(rho_i)

    for i in range(n_sites):
        mi_matrix[i, i] = single_entropies[i]
        for j in range(i + 1, n_sites):
            rho_ij = rdm_extractor.get_two_site_rdm(i, j)
            s_ij = von_neumann_entropy(rho_ij)
            mi = single_entropies[i] + single_entropies[j] - s_ij
            mi_val = max(0.0, float(mi))
            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val

    return mi_matrix
