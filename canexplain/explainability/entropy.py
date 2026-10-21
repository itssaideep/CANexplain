"""Von Neumann entanglement entropy calculation for tensor networks."""

import numpy as np


def von_neumann_entropy(rdm: np.ndarray, base: float = 2.0) -> float:
    """Calculates von Neumann entropy: S(rho) = -Tr(rho * log_base(rho)).

    Args:
        rdm: Square Reduced Density Matrix with Tr(rdm) == 1.0.
        base: Logarithm base (default 2.0 for bits / qubits).

    Returns:
        Scalar entanglement entropy (non-negative float).
    """
    # Eigenvalues of Hermitian density matrix
    eigenvals = np.linalg.eigvalsh(rdm)
    # Filter non-positive eigenvalues
    valid_eigs = eigenvals[eigenvals > 1e-12]

    if len(valid_eigs) == 0:
        return 0.0

    log_eigs = np.log2(valid_eigs) if base == 2.0 else np.log(valid_eigs) / np.log(base)

    entropy = -float(np.sum(valid_eigs * log_eigs))
    return max(0.0, entropy)


def compute_all_site_entropies(rdm_list: list[np.ndarray]) -> np.ndarray:
    """Computes von Neumann entropy for all sites in the MPS chain."""
    return np.array([von_neumann_entropy(rho) for rho in rdm_list], dtype=np.float64)
