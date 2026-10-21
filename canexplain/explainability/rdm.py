"""Reduced Density Matrix (RDM) extraction for Matrix Product States."""

from typing import List, Tuple
import numpy as np

from canexplain.models.mps.tensor_core import MatrixProductState


class RDMExtractor:
    """Extracts single-site and two-site Reduced Density Matrices from an MPS."""

    def __init__(self, mps: MatrixProductState):
        self.mps = mps
        self.num_sites = mps.num_sites
        self.left_envs: List[np.ndarray] = []
        self.right_envs: List[np.ndarray] = []
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

    def get_single_site_rdm(self, site_idx: int) -> np.ndarray:
        """Computes the single-site Reduced Density Matrix rho_k of shape (p, p).

        rho_k[s, s'] = Tr( E_L[k] * A_s * E_R[k+1] * A_{s'}^T ) / Z
        """
        if not (0 <= site_idx < self.num_sites):
            raise IndexError(f"Site index {site_idx} out of range [0, {self.num_sites}).")

        e_l = self.left_envs[site_idx]      # (D_L, D_L)
        e_r = self.right_envs[site_idx + 1]  # (D_R, D_R)
        a = self.mps.tensors[site_idx]      # (D_L, p, D_R)
        p = a.shape[1]

        # T = sum_i e_l[i, j] * a[i, s, a_r] -> (D_L, p, D_R)
        t = np.tensordot(e_l, a, axes=([0], [0]))
        # T2 = sum_b T[j, s, b] * e_r[b, c] -> (D_L, p, D_R)
        t2 = np.tensordot(t, e_r, axes=([2], [0]))
        # Contract with a[j, s', c] -> rho[s, s']
        rho = np.tensordot(t2, a, axes=([0, 2], [0, 2]))  # (p, p)

        # Normalize trace to 1.0
        tr = np.trace(rho)
        if tr > 1e-12:
            rho = rho / tr
        else:
            rho = np.eye(p, dtype=np.float64) / p

        # Ensure Hermitian symmetry
        return 0.5 * (rho + rho.T)

    def get_two_site_rdm(self, site_i: int, site_j: int) -> np.ndarray:
        """Computes the two-site Reduced Density Matrix rho_{i, j} of shape (p_i * p_j, p_i * p_j)."""
        if site_i > site_j:
            site_i, site_j = site_j, site_i
        if site_i == site_j:
            raise ValueError("Sites i and j must be distinct.")

        # For adjacent sites (i, i+1)
        if site_j == site_i + 1:
            e_l = self.left_envs[site_i]
            e_r = self.right_envs[site_j + 1]
            a_i = self.mps.tensors[site_i]     # (D_L, p1, D_M)
            a_j = self.mps.tensors[site_j]     # (D_M, p2, D_R)
            p1, p2 = a_i.shape[1], a_j.shape[1]

            # Merge two sites: B of shape (D_L, p1, p2, D_R)
            b = np.tensordot(a_i, a_j, axes=([2], [0]))
            t = np.tensordot(e_l, b, axes=([0], [0]))  # (D_L, p1, p2, D_R)
            t2 = np.tensordot(t, e_r, axes=([3], [0])) # (D_L, p1, p2, D_R)
            rho_4d = np.tensordot(t2, b, axes=([0, 3], [0, 3]))  # (p1, p2, p1, p2)

            rho_mat = rho_4d.reshape((p1 * p2, p1 * p2))
            tr = np.trace(rho_mat)
            if tr > 1e-12:
                rho_mat = rho_mat / tr
            else:
                rho_mat = np.eye(p1 * p2) / (p1 * p2)
            return 0.5 * (rho_mat + rho_mat.T)
        else:
            # Non-adjacent: product approximation via tensor product of single-site RDMs
            rho_i = self.get_single_site_rdm(site_i)
            rho_j = self.get_single_site_rdm(site_j)
            return np.kron(rho_i, rho_j)
