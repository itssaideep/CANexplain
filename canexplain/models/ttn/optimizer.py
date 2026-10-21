"""Hierarchical layer-sweep correlation optimizer for Tree Tensor Networks."""

import numpy as np

from canexplain.models.ttn.born_machine import TTNBornMachine


class TTNCorrelationOptimizer:
    """Hierarchical bottom-to-top correlation sweep optimizer for Binary TTNs.

    Updates node tensors layer-by-layer by computing local two-child correlation matrices
    and performing truncated Singular Value Decomposition (SVD) along the tree hierarchy.
    """

    def __init__(
        self,
        max_bond_dimension: int = 16,
        learning_rate: float = 0.1,
        num_sweeps: int = 3,
    ):
        self.max_bond_dimension = max_bond_dimension
        self.learning_rate = learning_rate
        self.num_sweeps = num_sweeps
        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    def fit(
        self,
        born_machine: TTNBornMachine,
        phi_train: np.ndarray,
        phi_val: np.ndarray | None = None,
    ) -> list[float]:
        """Fits TTN tensors using hierarchical correlation sweeps over training sequences.

        Args:
            born_machine: Target TTNBornMachine instance.
            phi_train: Training batch of shape (B, L, d).
            phi_val: Optional validation batch of shape (B_val, L, d).

        Returns:
            List of mean training NLL scores across sweeps.
        """
        ttn = born_machine.ttn
        if ttn is None:
            raise ValueError("Born machine contains no TTN.")

        batch_size = phi_train.shape[0]
        self.loss_history = []
        self.val_loss_history = []

        # Record initial loss
        initial_nll = float(np.mean(born_machine.score_samples(phi_train)))
        self.loss_history.append(initial_nll)
        if phi_val is not None:
            self.val_loss_history.append(float(np.mean(born_machine.score_samples(phi_val))))

        for _sweep in range(self.num_sweeps):
            # Propagate and update bottom-to-top
            curr_vectors = [phi_train[:, i, :] for i in range(phi_train.shape[1])]

            if len(curr_vectors) < ttn.num_leaves:
                pad_count = ttn.num_leaves - len(curr_vectors)
                neutral = np.zeros((batch_size, ttn.physical_dim), dtype=np.float64)
                neutral[:, 0] = 1.0
                for _ in range(pad_count):
                    curr_vectors.append(neutral)

            for layer_idx, layer in enumerate(ttn.layers):
                is_root = layer_idx == ttn.num_layers - 1
                next_vectors = []

                for node_idx, tensor in enumerate(layer):
                    v_l = curr_vectors[2 * node_idx]  # (B, d_l)
                    v_r = curr_vectors[2 * node_idx + 1]  # (B, d_r)
                    d_l = v_l.shape[1]
                    d_r = v_r.shape[1]

                    # Empirical two-child correlation: C[a, b] = (1/B) sum_k v_l[k, a] * v_r[k, b]
                    # Shape: (d_l, d_r)
                    corr_matrix = (v_l.T @ v_r) / float(batch_size)

                    if is_root:
                        # Root node parent dim is 1: tensor shape (d_l, d_r, 1)
                        # Blend existing tensor with empirical correlation
                        old_mat = tensor[:, :, 0]
                        new_mat = (
                            1.0 - self.learning_rate
                        ) * old_mat + self.learning_rate * corr_matrix
                        norm = np.linalg.norm(new_mat)
                        if norm > 1e-12:
                            new_mat /= norm
                        layer[node_idx] = new_mat[:, :, None]
                        # Contract root representation: (B, 1)
                        v_parent = np.einsum("lr,bl,br->b", new_mat, v_l, v_r)[:, None]
                    else:
                        # Truncated SVD on flattened correlation tensor
                        # Reshape tensor to matrix: (d_l * d_r, d_p)
                        d_p_curr = tensor.shape[2]
                        old_mat = tensor.reshape((d_l * d_r, d_p_curr))

                        # Target bond dimension
                        target_d = min(self.max_bond_dimension, d_l * d_r)

                        # Flatten correlation: (d_l * d_r, 1)
                        c_flat = corr_matrix.reshape((d_l * d_r, 1))
                        # Pad or match dimensions with old_mat
                        if old_mat.shape[1] < target_d:
                            pad = np.zeros(
                                (d_l * d_r, target_d - old_mat.shape[1]), dtype=np.float64
                            )
                            old_mat = np.hstack([old_mat, pad])
                        elif old_mat.shape[1] > target_d:
                            old_mat = old_mat[:, :target_d]

                        # Blend with outer product representation
                        update_term = np.zeros_like(old_mat)
                        update_term[:, 0:1] = c_flat
                        blended = (
                            1.0 - self.learning_rate
                        ) * old_mat + self.learning_rate * update_term

                        u, s, vh = np.linalg.svd(blended, full_matrices=False)
                        new_d = min(target_d, len(s))
                        new_tensor_mat = u[:, :new_d]  # Column-orthogonal isometry

                        layer[node_idx] = new_tensor_mat.reshape((d_l, d_r, new_d))

                        # Forward propagation: v_parent = einsum('lrp,bl,br->bp')
                        t_l = np.tensordot(layer[node_idx], v_l, axes=([0], [1]))
                        v_parent = np.einsum("rpb,br->bp", t_l, v_r)

                    next_vectors.append(v_parent)
                curr_vectors = next_vectors

            born_machine.reset_cached_norm()
            sweep_loss = float(np.mean(born_machine.score_samples(phi_train)))
            self.loss_history.append(sweep_loss)
            if phi_val is not None:
                self.val_loss_history.append(float(np.mean(born_machine.score_samples(phi_val))))

        return self.loss_history
