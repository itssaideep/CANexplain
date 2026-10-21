"""Tensor network optimizers for generative MPS Born machines.

Includes:
1. HeuristicCorrelationOptimizer (DMRG-inspired heuristic local correlation update).
2. GradientBornMachineOptimizer (Exact analytical gradient descent on empirical NLL).
3. DMRGOptimizer (Backward-compatible alias for HeuristicCorrelationOptimizer).
"""

from typing import List, Optional, Tuple
import numpy as np

from canexplain.models.mps.tensor_core import MatrixProductState
from canexplain.models.mps.born_machine import MPSBornMachine


class HeuristicCorrelationOptimizer:
    """DMRG-inspired heuristic optimizer for generative Matrix Product States.

    Scientific Disclaimer:
        In quantum chemistry and physics, standard DMRG projects a Hamiltonian into the
        environment basis and solves an exact local generalized eigenvalue problem.
        In generative unsupervised modeling, exact variational optimization contracts sample
        environment tensors to compute exact gradients.
        This class implements a fast DMRG-inspired heuristic update:
        it sweeps through adjacent tensor pairs (A^k, A^{k+1}), merges them into a two-site
        tensor B, adds a scaled empirical feature correlation outer product, and truncates
        virtual bonds via SVD.
        It is provided as a fast heuristic baseline. For exact likelihood gradient updates,
        use GradientBornMachineOptimizer.
    """

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
        self.val_loss_history: List[float] = []

    def fit(
        self,
        born_machine: MPSBornMachine,
        phi_train: np.ndarray,
        phi_val: Optional[np.ndarray] = None,
    ) -> List[float]:
        """Trains the MPS Born machine on encoded benign traffic.

        Args:
            born_machine: MPSBornMachine instance to train.
            phi_train: Array of shape (B, N, p) representing encoded benign sequences.
            phi_val: Optional validation split of shape (B_val, N, p) for tracking generalization.

        Returns:
            List of mean training NLL loss values per sweep.
        """
        mps = born_machine.mps
        if mps is None:
            raise ValueError("Born machine contains no MPS.")

        num_samples, num_sites, p_dim = phi_train.shape
        self.loss_history = []
        self.val_loss_history = []

        # Record initial loss
        initial_nll = float(np.mean(born_machine.score_samples(phi_train)))
        self.loss_history.append(initial_nll)
        if phi_val is not None:
            self.val_loss_history.append(float(np.mean(born_machine.score_samples(phi_val))))

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
            if phi_val is not None:
                self.val_loss_history.append(float(np.mean(born_machine.score_samples(phi_val))))

        return self.loss_history

    def _update_two_site(
        self,
        mps: MatrixProductState,
        phi_train: np.ndarray,
        k: int,
        direction: str = "right",
    ) -> None:
        """Updates adjacent tensor pair (A^k, A^{k+1}) via heuristic correlation outer product."""
        a_left = mps.tensors[k]        # (D_L, p1, D_M)
        a_right = mps.tensors[k + 1]   # (D_M, p2, D_R)
        d_l, p1, d_m = a_left.shape
        _, p2, d_r = a_right.shape

        # Merge into two-site tensor B of shape (D_L, p1, p2, D_R)
        b_tensor = np.tensordot(a_left, a_right, axes=([2], [0]))

        # Empirical data outer product at sites k and k+1
        phi_1 = phi_train[:, k]      # (B, p1)
        phi_2 = phi_train[:, k + 1]  # (B, p2)
        phi_pair = np.einsum("bi,bj->bij", phi_1, phi_2)  # (B, p1, p2)
        data_corr = np.mean(phi_pair, axis=0)  # (p1, p2)

        # Broadcast correlation across virtual dimensions
        grad_step = np.zeros_like(b_tensor)
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

        if direction == "right":
            mps.tensors[k] = u_trunc.reshape((d_l, p1, new_d))
            mps.tensors[k + 1] = (s_trunc[:, None] * vh_trunc).reshape((new_d, p2, d_r))
        else:
            mps.tensors[k] = (u_trunc * s_trunc[None, :]).reshape((d_l, p1, new_d))
            mps.tensors[k + 1] = vh_trunc.reshape((new_d, p2, d_r))


class GradientBornMachineOptimizer:
    """Exact analytical gradient descent on Born machine Negative Log-Likelihood.

    Objective:
        L(Psi) = - (2/B) sum_{b=1}^B ln |<Psi | Phi^(b)>| + ln <Psi | Psi>

    Gradient for site tensor A^(k):
        grad_{A^(k)} = - (2/B) sum_{b=1}^B (1 / <Psi | Phi^(b)>) [ L^(b)_k (x) Phi^(b)_k (x) R^(b)_k ]
                       + (2 / Z) [ E_L^(k) * A^(k) * E_R^(k) ]
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        num_epochs: int = 10,
        batch_size: int = 64,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    def fit(
        self,
        born_machine: MPSBornMachine,
        phi_train: np.ndarray,
        phi_val: Optional[np.ndarray] = None,
    ) -> List[float]:
        mps = born_machine.mps
        if mps is None:
            raise ValueError("Born machine contains no MPS.")

        num_samples, num_sites, p_dim = phi_train.shape
        self.loss_history = []
        self.val_loss_history = []

        initial_loss = float(np.mean(born_machine.score_samples(phi_train)))
        self.loss_history.append(initial_loss)
        if phi_val is not None:
            self.val_loss_history.append(float(np.mean(born_machine.score_samples(phi_val))))

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(num_samples)
            num_batches = int(np.ceil(num_samples / self.batch_size))

            for b_i in range(num_batches):
                batch_idx = indices[b_i * self.batch_size : (b_i + 1) * self.batch_size]
                phi_b = phi_train[batch_idx]
                b_size = len(phi_b)
                if b_size == 0:
                    continue

                # 1. Compute left environments for all samples in batch: L[k] shape (B, D_k)
                left_envs = [np.ones((b_size, 1), dtype=np.float64)]
                v_curr = np.ones((b_size, 1, 1), dtype=np.float64)
                for k in range(num_sites - 1):
                    m_k = np.einsum("lpr,bp->blr", mps.tensors[k], phi_b[:, k, :])
                    v_curr = np.matmul(v_curr, m_k)
                    left_envs.append(v_curr[:, 0, :])

                # 2. Compute right environments for all samples in batch: R[k] shape (B, D_k)
                right_envs = [np.ones((b_size, 1), dtype=np.float64)]
                v_right = np.ones((b_size, 1, 1), dtype=np.float64)
                for k in range(num_sites - 1, 0, -1):
                    m_k = np.einsum("lpr,bp->blr", mps.tensors[k], phi_b[:, k, :])
                    v_right = np.matmul(m_k, v_right)
                    right_envs.append(v_right[:, :, 0])
                right_envs.reverse()

                # Sample amplitudes: <Psi | Phi^(b)>
                amplitudes = born_machine.mps.contract_batch(phi_b)
                # Safeguard against zero amplitudes
                safe_amps = np.where(np.abs(amplitudes) < 1e-12, 1e-12 * np.sign(amplitudes + 1e-15), amplitudes)

                # Partition function and norm environments
                z = max(mps.norm_squared(), 1e-12)
                # Left transfer matrices for norm
                e_l = [np.ones((1, 1), dtype=np.float64)]
                e_curr = np.ones((1, 1), dtype=np.float64)
                for k in range(num_sites - 1):
                    t = np.tensordot(e_curr, mps.tensors[k], axes=([0], [0]))
                    e_curr = np.tensordot(t, mps.tensors[k], axes=([0, 1], [0, 1]))
                    e_l.append(e_curr)

                # Right transfer matrices for norm
                e_r = [np.ones((1, 1), dtype=np.float64)]
                e_r_curr = np.ones((1, 1), dtype=np.float64)
                for k in range(num_sites - 1, 0, -1):
                    t = np.tensordot(mps.tensors[k], e_r_curr, axes=([2], [0]))
                    e_r_curr = np.tensordot(t, mps.tensors[k], axes=([1, 2], [1, 2]))
                    e_r.append(e_r_curr)
                e_r.reverse()

                # Site-by-site gradient descent update
                for k in range(num_sites):
                    l_k = left_envs[k]            # (B, D_L)
                    r_k = right_envs[k]           # (B, D_R)
                    phi_k = phi_b[:, k, :]        # (B, p)
                    weights = (1.0 / safe_amps)   # (B,)

                    # Sample gradient contribution: - (2/B) sum_b (1/amp) * l_b (x) phi_b (x) r_b
                    # Einsum: b, bl, bp, br -> lpr
                    grad_sample = -2.0 * np.einsum("b,bl,bp,br->lpr", weights, l_k, phi_k, r_k) / b_size

                    # Norm gradient contribution: (2/Z) * E_L * A * E_R
                    # E_L: (D_L, D_L), A: (D_L, p, D_R), E_R: (D_R, D_R)
                    t_norm = np.tensordot(e_l[k], mps.tensors[k], axes=([0], [0]))
                    grad_norm = 2.0 * np.tensordot(t_norm, e_r[k], axes=([2], [0])) / z

                    grad_k = grad_sample + grad_norm
                    # Gradient step with gradient clipping
                    grad_clipped = np.clip(grad_k, -5.0, 5.0)
                    mps.tensors[k] = mps.tensors[k] - self.learning_rate * grad_clipped

                born_machine.reset_cached_norm()

            # End of epoch loss
            epoch_loss = float(np.mean(born_machine.score_samples(phi_train)))
            self.loss_history.append(epoch_loss)
            if phi_val is not None:
                self.val_loss_history.append(float(np.mean(born_machine.score_samples(phi_val))))

        return self.loss_history


# Backward-compatible alias for existing code
DMRGOptimizer = HeuristicCorrelationOptimizer
