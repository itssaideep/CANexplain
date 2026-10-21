"""Matrix Product State (MPS) core tensor operations and canonicalization."""

from typing import List, Optional, Tuple
import numpy as np


class MatrixProductState:
    """Open-boundary Matrix Product State (MPS) tensor chain.

    Represents a quantum-like wavefunction |Psi> decomposed into N local tensors:
    A^(k) of shape (D_{k-1}, p_k, D_k), with D_0 = D_N = 1.
    """

    def __init__(self, tensors: List[np.ndarray]):
        """Initializes MPS from a list of 3-way tensors."""
        self.num_sites = len(tensors)
        self.tensors = [np.array(t, dtype=np.float64) for t in tensors]
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.num_sites == 0:
            raise ValueError("MPS must have at least one site.")
        if self.tensors[0].shape[0] != 1:
            raise ValueError(f"Site 0 left virtual dimension must be 1; got {self.tensors[0].shape[0]}.")
        if self.tensors[-1].shape[2] != 1:
            raise ValueError(f"Site N-1 right virtual dimension must be 1; got {self.tensors[-1].shape[2]}.")

        for k in range(self.num_sites - 1):
            d_right = self.tensors[k].shape[2]
            d_left = self.tensors[k + 1].shape[0]
            if d_right != d_left:
                raise ValueError(
                    f"Virtual bond mismatch between site {k} (D_R={d_right}) and site {k+1} (D_L={d_left})."
                )

    @property
    def physical_dimensions(self) -> List[int]:
        return [t.shape[1] for t in self.tensors]

    @property
    def bond_dimensions(self) -> List[int]:
        return [t.shape[2] for t in self.tensors[:-1]]

    @property
    def max_bond_dimension(self) -> int:
        bonds = self.bond_dimensions
        return max(bonds) if bonds else 1

    @classmethod
    def initialize_random(
        cls,
        num_sites: int,
        physical_dim: int = 2,
        bond_dimension: int = 20,
        seed: Optional[int] = 42,
    ) -> "MatrixProductState":
        """Initializes a random normalized open-boundary MPS."""
        rng = np.random.default_rng(seed)
        tensors = []

        d_prev = 1
        for k in range(num_sites):
            # Bond dimension grows up to min(bond_dimension, p^k, p^(N-k))
            if k == num_sites - 1:
                d_next = 1
            else:
                d_next = min(bond_dimension, d_prev * physical_dim)

            # Draw random Gaussian tensor
            tensor = rng.normal(0.0, 1.0 / np.sqrt(d_prev * physical_dim), size=(d_prev, physical_dim, d_next))
            tensors.append(tensor)
            d_prev = d_next

        mps = cls(tensors)
        mps.canonicalize_left()
        return mps

    def contract_sample(self, phi_sample: np.ndarray) -> float:
        """Contracts MPS with a single sample state vector product.

        Args:
            phi_sample: Array of shape (num_sites, physical_dim) where each row is |phi(x_k)>.

        Returns:
            Scalar amplitude <Psi | Phi(v)>.
        """
        # Start with left boundary vector of shape (1, 1)
        v_curr = np.ones((1, 1), dtype=np.float64)

        for k in range(self.num_sites):
            # A^(k) shape: (D_L, p, D_R)
            # phi_k shape: (p,)
            # Contract over physical index p: M_k = sum_s A^(k)_{:, s, :} * phi_k[s] -> shape (D_L, D_R)
            phi_k = phi_sample[k]
            m_k = np.tensordot(self.tensors[k], phi_k, axes=([1], [0]))  # shape (D_L, D_R)
            v_curr = np.matmul(v_curr, m_k)

        return float(v_curr[0, 0])

    def contract_batch(self, phi_batch: np.ndarray) -> np.ndarray:
        """Contracts MPS with a batch of samples via vectorized BLAS tensor contractions.

        Args:
            phi_batch: Array of shape (B, num_sites, physical_dim).

        Returns:
            Array of shape (B,) containing amplitudes <Psi | Phi(v)>.
        """
        b = len(phi_batch)
        if b == 0:
            return np.empty((0,), dtype=np.float64)

        # Vectorized batch contraction: shape (B, 1, 1)
        v_curr = np.ones((b, 1, 1), dtype=np.float64)
        for k in range(self.num_sites):
            # A^(k) shape: (D_L, p, D_R), phi_batch[:, k, :] shape: (B, p)
            m_k = np.einsum("lpr,bp->blr", self.tensors[k], phi_batch[:, k, :])
            v_curr = np.matmul(v_curr, m_k)

        return np.asarray(v_curr[:, 0, 0], dtype=np.float64)

    def norm_squared(self) -> float:
        """Computes the norm squared <Psi | Psi> via transfer matrix contractions."""
        # Transfer matrix contraction from left to right
        # E_0 shape: (1, 1)
        e = np.ones((1, 1), dtype=np.float64)

        for k in range(self.num_sites):
            # A = self.tensors[k] shape: (D_L, p, D_R)
            # e shape: (D_L, D_L)
            # contract e with A: T = sum_{i} e[i, j] * A[i, s, a] -> shape (D_L, p, D_R)
            a = self.tensors[k]
            t = np.tensordot(e, a, axes=([0], [0]))  # shape (D_L, p, D_R)
            # contract with A conjugate: E_new[a, b] = sum_{j, s} T[j, s, a] * A[j, s, b]
            e = np.tensordot(t, a, axes=([0, 1], [0, 1]))  # shape (D_R, D_R)

        return float(e[0, 0])

    def canonicalize_left(self) -> "MatrixProductState":
        """Transforms MPS into left-canonical form via successive QR decompositions."""
        for k in range(self.num_sites - 1):
            d_l, p, d_r = self.tensors[k].shape
            # Reshape into matrix of shape (d_l * p, d_r)
            mat = self.tensors[k].reshape((d_l * p, d_r))
            q, r = np.linalg.qr(mat, mode="reduced")
            new_bond = q.shape[1]

            self.tensors[k] = q.reshape((d_l, p, new_bond))
            # Absorb R into next tensor A^(k+1)
            # next tensor shape: (d_r, p_{k+1}, d_{next})
            next_t = self.tensors[k + 1]
            self.tensors[k + 1] = np.tensordot(r, next_t, axes=([1], [0]))

        # Normalize last site
        last_t = self.tensors[-1]
        norm = np.linalg.norm(last_t)
        if norm > 1e-12:
            self.tensors[-1] = last_t / norm

        return self
