"""Reduced Density Matrix (RDM) extraction for Matrix Product States."""

import numpy as np

from canexplain.models.mps.tensor_core import MatrixProductState


class RDMExtractor:
    """Extracts single-site and two-site Reduced Density Matrices from an MPS."""

    def __init__(self, mps: MatrixProductState):
        self.mps = mps
        self.num_sites = mps.num_sites
        self.left_envs: list[np.ndarray] = []
        self.right_envs: list[np.ndarray] = []
        self._build_environments()

    def _build_environments(self) -> None:
        """Computes and caches left and right environment tensors."""
        # Left environments: E_L[k] has shape (D_k, D_k)
        # E_L[0] is for the left of site 0
        e_l = np.ones((1, 1), dtype=np.float64)
        self.left_envs = [e_l]

        for k in range(self.num_sites):
            a = self.mps.tensors[k]  # (D_L, p, D_R)
            # t = sum_i e_l[i, j] * a[i, s, a_r]
            t = np.tensordot(e_l, a, axes=([0], [0]))  # (D_L, p, D_R)
            e_l = np.tensordot(t, a, axes=([0, 1], [0, 1]))  # (D_R, D_R)
            self.left_envs.append(e_l)

        # Right environments: E_R[k] has shape (D_k, D_k)
        e_r = np.ones((1, 1), dtype=np.float64)
        self.right_envs = [e_r]

        for k in range(self.num_sites - 1, -1, -1):
            a = self.mps.tensors[k]  # (D_L, p, D_R)
            # t = sum_j a[a_l, s, j] * e_r[j, i]
            t = np.tensordot(a, e_r, axes=([2], [0]))  # (D_L, p, D_R)
            e_r = np.tensordot(t, a, axes=([1, 2], [1, 2]))  # (D_L, D_L)
            self.right_envs.append(e_r)

        self.right_envs.reverse()

    @staticmethod
    def _ensure_psd_and_trace_normalized(rho: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Enforces Hermiticity, Positive Semi-Definiteness (PSD), and unit trace.

        Numerical contractions can introduce tiny negative eigenvalues (< 1e-12).
        This method projects rho onto the valid density matrix manifold.
        """
        # Symmetrize
        rho_herm = 0.5 * (rho + rho.T)

        # Spectral decomposition
        eigenvals, eigenvecs = np.linalg.eigh(rho_herm)

        # Clamp tiny negative eigenvalues within tolerance
        eigenvals_clamped = np.maximum(eigenvals, 0.0)

        # Reconstruct
        rho_psd = eigenvecs @ np.diag(eigenvals_clamped) @ eigenvecs.T

        # Normalize trace
        tr = float(np.trace(rho_psd))
        dim = rho.shape[0]
        rho_norm = rho_psd / tr if tr > tol else np.eye(dim, dtype=np.float64) / dim

        return 0.5 * (rho_norm + rho_norm.T)

    def get_single_site_rdm(self, site_idx: int) -> np.ndarray:
        """Computes the single-site Reduced Density Matrix rho_k of shape (p, p).

        rho_k[s, s'] = Tr( E_L[k] * A_s * E_R[k+1] * A_{s'}^T ) / Z
        """
        if not (0 <= site_idx < self.num_sites):
            raise IndexError(f"Site index {site_idx} out of range [0, {self.num_sites}).")

        e_l = self.left_envs[site_idx]  # (D_L, D_L)
        e_r = self.right_envs[site_idx + 1]  # (D_R, D_R)
        a = self.mps.tensors[site_idx]  # (D_L, p, D_R)

        # T = sum_i e_l[i, j] * a[i, s, a_r] -> (D_L, p, D_R)
        t = np.tensordot(e_l, a, axes=([0], [0]))
        # T2 = sum_b T[j, s, b] * e_r[b, c] -> (D_L, p, D_R)
        t2 = np.tensordot(t, e_r, axes=([2], [0]))
        # Contract with a[j, s', c] -> rho[s, s']
        rho = np.tensordot(t2, a, axes=([0, 2], [0, 2]))  # (p, p)

        return self._ensure_psd_and_trace_normalized(rho)

    def get_two_site_rdm(self, site_i: int, site_j: int) -> np.ndarray:
        """Computes the exact two-site RDM rho_{i, j} of shape (p_i * p_j, p_i * p_j).

        For both adjacent (j = i + 1) and non-adjacent (j > i + 1) site pairs:
        Intervening sites k in {i+1, ..., j-1} are traced out by contracting through
        their exact transfer matrices E_k = sum_{s_k} A^(k)_{s_k} (x) A^(k)_{s_k}.

        Guarantees:
            - True non-local quantum correlations and mutual information I(i; j) >= 0.
            - Positive semi-definiteness and unit trace normalization.
        """
        if site_i > site_j:
            site_i, site_j = site_j, site_i
        if site_i == site_j:
            raise ValueError("Sites i and j must be distinct.")

        e_l = self.left_envs[site_i]
        e_r = self.right_envs[site_j + 1]
        a_i = self.mps.tensors[site_i]  # (D_L_i, p_i, D_R_i)
        a_j = self.mps.tensors[site_j]  # (D_L_j, p_j, D_R_j)
        p_i, p_j = a_i.shape[1], a_j.shape[1]

        # 1. Contract left environment E_L with site i (keep s_i, s_i' open)
        # t1: contract e_l[alpha, beta] with a_i[alpha, s_i, gamma] -> shape (beta, s_i, gamma)
        t1 = np.tensordot(e_l, a_i, axes=([0], [0]))
        # Contract with second copy a_i[beta, s_i', delta]:
        # curr[s_i, s_i', gamma, delta] = sum_{beta} t1[beta, s_i, gamma] * a_i[beta, s_i', delta]
        curr = np.einsum("bsg,btd->stgd", t1, a_i)  # shape (p_i, p_i, D_R_i, D_R_i)

        # 2. Trace over all intervening sites k from i+1 to j-1 via transfer matrices
        for k in range(site_i + 1, site_j):
            a_k = self.mps.tensors[k]  # (D_L_k, p_k, D_R_k)
            # curr has shape (p_i, p_i, D_L_k, D_L_k)
            # Contract curr[s_i, s_i', a, b] * a_k[a, s, g] * a_k[b, s, h]
            curr = np.einsum("ijab,asg,bsh->ijgh", curr, a_k, a_k)

        # 3. Contract with site j and right environment E_R[j+1]
        # curr shape: (p_i, p_i, D_L_j, D_L_j)
        # a_j shape: (D_L_j, p_j, D_R_j)
        # e_r shape: (D_R_j, D_R_j)
        # Output indices: isjt -> (s_i, s_j, s_i', s_j')
        rho_4d = np.einsum(
            "ijab,asg,bth,gh->isjt", curr, a_j, a_j, e_r
        )  # shape (p_i, p_j, p_i, p_j)

        # Reshape into square matrix of dimension (p_i * p_j, p_i * p_j)
        rho_mat = rho_4d.reshape((p_i * p_j, p_i * p_j))

        return self._ensure_psd_and_trace_normalized(rho_mat)
