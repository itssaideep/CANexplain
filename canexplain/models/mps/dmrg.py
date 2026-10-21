"""Two-site DMRG-inspired variational optimizer for Matrix Product States."""

from typing import List, Tuple
import numpy as np

from canexplain.models.mps.tensor_core import MatrixProductState
from canexplain.models.mps.born_machine import MPSBornMachine


class DMRGOptimizer:
    """Variational two-site DMRG optimizer for generative MPS training on benign CAN traffic."""

    def __init__(
        self,
        max_bond_dimension: int = 32,
        learning_rate: float = 0.05,
        num_sweeps: int = 5,
        svd_cutoff: float = 1e-8,
    ):
        self.max_bond_dimension = max_bond_dimension
        self.learning_rate = learning_rate
        self.num_sweeps = num_sweeps
        self.svd_cutoff = svd_cutoff
        self.loss_history: List[float] = []

    def fit(self, born_machine: MPSBornMachine, phi_train: np.ndarray) -> List[float]:
        """Trains the MPS Born machine on encoded benign traffic.

        Args:
            born_machine: MPSBornMachine instance to train.
            phi_train: Array of shape (B, N, p) representing encoded benign sequences.

        Returns:
            List of mean NLL loss values per sweep.
        """
        mps = born_machine.mps
        if mps is None:
            raise ValueError("Born machine contains no MPS.")

        num_samples, num_sites, p_dim = phi_train.shape
        self.loss_history = []

        # Record initial loss
        initial_nll = float(np.mean(born_machine.score_samples(phi_train)))
        self.loss_history.append(initial_nll)

        for sweep in range(self.num_sweeps):
            # Left-to-right sweep
            for k in range(num_sites - 1):
                self._update_two_site(mps, phi_train, k, direction="right")
                born_machine.reset_cached_norm()

            # Right-to-left sweep
            for k in range(num_sites - 2, -1, -1):
                self._update_two_site(mps, phi_train, k, direction="left")
                born_machine.reset_cached_norm()

            sweep_nll = float(np.mean(born_machine.score_samples(phi_train)))
            self.loss_history.append(sweep_nll)

        return self.loss_history

    def _update_two_site(
        self,
        mps: MatrixProductState,
        phi_train: np.ndarray,
        k: int,
        direction: str = "right"
    ) -> None:
        """Optimizes the adjacent tensor pair (A^k, A^{k+1}) via two-site variational step."""
        a_left = mps.tensors[k]        # (D_L, p1, D_M)
        a_right = mps.tensors[k + 1]   # (D_M, p2, D_R)
        d_l, p1, d_m = a_left.shape
        _, p2, d_r = a_right.shape

        # Merge into two-site tensor B of shape (D_L, p1, p2, D_R)
        b_tensor = np.tensordot(a_left, a_right, axes=([2], [0]))

        # Compute effective sample projection at sites k and k+1
        # phi_train[:, k] shape: (B, p1), phi_train[:, k+1] shape: (B, p2)
        phi_1 = phi_train[:, k]      # (B, p1)
        phi_2 = phi_train[:, k + 1]  # (B, p2)
        phi_pair = np.einsum("bi,bj->bij", phi_1, phi_2)  # (B, p1, p2)

        # Compute empirical gradient proxy: pull B towards outer product of active data features
        # Data correlation tensor C: sum_b phi_pair[b, i, j] -> (p1, p2)
        data_corr = np.mean(phi_pair, axis=0)  # (p1, p2)

        # Update B with data projection
        # Construct rank-1 adjustment from left boundary to right boundary
        grad_step = np.zeros_like(b_tensor)
        # Uniformly propagate correlation into virtual bonds
        scale = self.learning_rate / (d_l * d_r)
        grad_step[:, :, :, :] = scale * data_corr[None, :, :, None]

        b_updated = b_tensor + grad_step

        # Reshape B for SVD: (D_L * p1, p2 * D_R)
        b_mat = b_updated.reshape((d_l * p1, p2 * d_r))
        u, s, vh = np.linalg.svd(b_mat, full_matrices=False)

        # Truncation: keep up to max_bond_dimension where s_i / s_0 >= svd_cutoff
        s_max = s[0] if len(s) > 0 and s[0] > 0 else 1.0
        keep = np.where((s / s_max) >= self.svd_cutoff)[0]
        new_d = min(self.max_bond_dimension, max(len(keep), 1))

        u_trunc = u[:, :new_d]
        s_trunc = s[:new_d]
        vh_trunc = vh[:new_d, :]

        # Split back into A^k and A^{k+1}
        if direction == "right":
            # Left tensor becomes isometry (U)
            mps.tensors[k] = u_trunc.reshape((d_l, p1, new_d))
            # Right tensor absorbs singular values: diag(S) * Vh
            mps.tensors[k + 1] = (s_trunc[:, None] * vh_trunc).reshape((new_d, p2, d_r))
        else:
            # Right tensor becomes isometry (Vh)
            mps.tensors[k] = (u_trunc * s_trunc[None, :]).reshape((d_l, p1, new_d))
            mps.tensors[k + 1] = vh_trunc.reshape((new_d, p2, d_r))
