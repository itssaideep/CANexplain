"""Hierarchical Binary Tree Tensor Network (TTN) core for automotive sequence modeling.

Scientific Principle:
    In a 1D Matrix Product State (MPS), the virtual bond distance between distant sequence
    sites scales linearly as O(N). For extended temporal windows (e.g. W=32 frames, 320 sites),
    this linear topology induces severe gradient vanishing and correlation attenuation.

    A Binary Tree Tensor Network (TTN) organizes tensors into a balanced hierarchical tree of
    depth K = log2(L), where L is the number of leaf sites.
    - Path length between any two leaves is at most 2 * log2(L) = O(log L).
    - Lower layers capture fine-grained intra-frame byte correlations.
    - Upper layers capture multi-frame temporal driving dynamics and attack trends.
"""

import math
from typing import Any

import numpy as np


class BinaryTreeTensorNetwork:
    """Represents a balanced binary Tree Tensor Network with open boundaries.

    Attributes:
        num_leaves: Number of physical leaf sites (must be a power of 2).
        physical_dim: Dimension of local Hilbert space at each leaf (default 2).
        bond_dimension: Maximum virtual bond dimension connecting tree nodes.
        layers: List of tensor layers from leaves (layer 0) to root (layer K-1).
    """

    def __init__(
        self,
        layers: list[list[np.ndarray]],
        physical_dim: int = 2,
    ):
        self.layers = layers
        self.num_layers = len(layers)
        self.physical_dim = physical_dim
        self.num_leaves = 2**self.num_layers
        self._validate_tree_structure()

    def _validate_tree_structure(self) -> None:
        """Verifies tensor shapes and virtual bond compatibility across layers."""
        expected_nodes = self.num_leaves // 2
        for layer_idx, layer in enumerate(self.layers):
            if len(layer) != expected_nodes:
                raise ValueError(
                    f"Layer {layer_idx} expected {expected_nodes} nodes, got {len(layer)}."
                )
            expected_nodes //= 2

    @classmethod
    def initialize_random(
        cls,
        num_leaves: int,
        physical_dim: int = 2,
        bond_dimension: int = 16,
        seed: int | None = 42,
    ) -> "BinaryTreeTensorNetwork":
        """Initializes a random normalized Binary Tree Tensor Network.

        Args:
            num_leaves: Number of leaf sites (must be power of 2, e.g. 8, 16, 32, 64).
            physical_dim: Local physical site dimension (default 2 for qubits).
            bond_dimension: Maximum virtual bond dimension D.
            seed: RNG seed for reproducible initialization.
        """
        if num_leaves < 2 or (num_leaves & (num_leaves - 1)) != 0:
            # Round up to nearest power of 2
            power = int(math.ceil(math.log2(max(num_leaves, 2))))
            num_leaves = 2**power

        num_layers = int(math.log2(num_leaves))
        rng = np.random.default_rng(seed)
        layers: list[list[np.ndarray]] = []

        d_child = physical_dim
        for layer_idx in range(num_layers):
            num_nodes = num_leaves // (2 ** (layer_idx + 1))
            is_root = layer_idx == num_layers - 1
            d_parent = 1 if is_root else min(bond_dimension, d_child * d_child)

            layer_tensors = []
            for _ in range(num_nodes):
                # Shape: (d_left, d_right, d_parent)
                scale = 1.0 / np.sqrt(d_child * d_child)
                t = rng.normal(0.0, scale, size=(d_child, d_child, d_parent))
                # Normalize column Frobenius norm
                norm = np.linalg.norm(t)
                if norm > 1e-12:
                    t /= norm
                layer_tensors.append(t)

            layers.append(layer_tensors)
            d_child = d_parent

        ttn = cls(layers=layers, physical_dim=physical_dim)
        return ttn

    def contract_sample(self, phi_sample: np.ndarray) -> float:
        """Contracts TTN with a single sample state product |Phi(x)> of shape (L, d).

        Args:
            phi_sample: Array of shape (num_leaves, physical_dim).

        Returns:
            Scalar amplitude <Psi_TTN | Phi(x)>.
        """
        curr_vectors = [phi_sample[i] for i in range(len(phi_sample))]

        # Pad with |0> = [1, 0] states if sample length < num_leaves
        if len(curr_vectors) < self.num_leaves:
            pad_count = self.num_leaves - len(curr_vectors)
            neutral = np.zeros(self.physical_dim, dtype=np.float64)
            neutral[0] = 1.0
            for _ in range(pad_count):
                curr_vectors.append(neutral)

        # Bottom-to-top contraction through tree layers
        for layer in self.layers:
            next_vectors = []
            for node_idx, tensor in enumerate(layer):
                v_left = curr_vectors[2 * node_idx]
                v_right = curr_vectors[2 * node_idx + 1]

                # Contract left child: T_l = sum_s_l tensor[s_l, s_r, p] * v_left[s_l] -> (s_r, p)
                t_l = np.tensordot(tensor, v_left, axes=([0], [0]))
                # Contract right child: T_lr = sum_s_r T_l[s_r, p] * v_right[s_r] -> (p,)
                v_parent = np.tensordot(t_l, v_right, axes=([0], [0]))
                next_vectors.append(v_parent)
            curr_vectors = next_vectors

        # Root node has parent dimension 1
        root_val = float(curr_vectors[0][0]) if len(curr_vectors[0]) > 0 else 0.0
        return root_val

    def contract_batch(self, phi_batch: np.ndarray) -> np.ndarray:
        """Vectorized contraction for a batch of samples of shape (B, L, d).

        Args:
            phi_batch: Array of shape (B, L, d).

        Returns:
            1D array of amplitudes <Psi | Phi(x)> of shape (B,).
        """
        batch_size = phi_batch.shape[0]
        # curr_vectors shape: (num_leaves, B, d)
        curr_vectors = [phi_batch[:, i, :] for i in range(phi_batch.shape[1])]

        if len(curr_vectors) < self.num_leaves:
            pad_count = self.num_leaves - len(curr_vectors)
            neutral = np.zeros((batch_size, self.physical_dim), dtype=np.float64)
            neutral[:, 0] = 1.0
            for _ in range(pad_count):
                curr_vectors.append(neutral)

        for layer in self.layers:
            next_vectors = []
            for node_idx, tensor in enumerate(layer):
                v_left = curr_vectors[2 * node_idx]  # (B, d_l)
                v_right = curr_vectors[2 * node_idx + 1]  # (B, d_r)

                # tensor shape: (d_l, d_r, d_p)
                # Contract left: (d_r, d_p, B) -> axes=([0], [1])
                t_l = np.tensordot(tensor, v_left, axes=([0], [1]))
                # t_l shape: (d_r, d_p, B). Contract right with v_right:
                # sum_{d_r} t_l[d_r, d_p, b] * v_right[b, d_r] -> (d_p, B)
                # Using einsum for exact index alignment:
                v_parent = np.einsum("rpb,br->bp", t_l, v_right)
                next_vectors.append(v_parent)
            curr_vectors = next_vectors

        # Root output: shape (B, 1) -> (B,)
        return curr_vectors[0][:, 0].astype(np.float64)

    def compute_norm_squared(self) -> float:
        """Computes total state norm Z = <Psi_TTN | Psi_TTN> via tree transfer contraction."""
        # Leaves start with identity matrices I_d of shape (d, d)
        curr_densities = [
            np.eye(self.physical_dim, dtype=np.float64) for _ in range(self.num_leaves)
        ]

        for layer in self.layers:
            next_densities = []
            for node_idx, tensor in enumerate(layer):
                rho_l = curr_densities[2 * node_idx]  # (d_l, d_l)
                rho_r = curr_densities[2 * node_idx + 1]  # (d_r, d_r)

                # tensor: (d_l, d_r, d_p)
                # Contract with rho_l on left: T1 = sum_a tensor[a, r, p] * rho_l[a, a']
                t1 = np.tensordot(tensor, rho_l, axes=([0], [0]))
                # Contract with tensor on a': T2 = sum_a' T1[r, p, a'] * tensor[a', r', p']
                t2 = np.tensordot(t1, tensor, axes=([2], [0]))  # (r, p, r', p')
                # Contract with rho_r on (r, r'): T3 = sum_{r, r'} T2[r, p, r', p'] * rho_r[r, r']
                rho_p = np.einsum("rprq,rr->pq", t2, rho_r)  # (d_p, d_p)
                next_densities.append(rho_p)
            curr_densities = next_densities

        z = float(np.trace(curr_densities[0]))
        return max(1e-12, z)

    def to_dict(self) -> dict[str, Any]:
        """Serializes TTN architecture and weights to a dictionary."""
        serialized_layers = []
        for layer in self.layers:
            serialized_layers.append([t.tolist() for t in layer])
        return {
            "num_leaves": self.num_leaves,
            "physical_dim": self.physical_dim,
            "num_layers": self.num_layers,
            "layers": serialized_layers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BinaryTreeTensorNetwork":
        """Reconstructs TTN from dictionary."""
        layers = []
        for l_data in data["layers"]:
            layers.append([np.array(t, dtype=np.float64) for t in l_data])
        return cls(layers=layers, physical_dim=data["physical_dim"])
