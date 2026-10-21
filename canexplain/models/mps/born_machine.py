"""MPS Born Machine for unsupervised generative anomaly detection."""

import numpy as np

from canexplain.models.mps.tensor_core import MatrixProductState


class MPSBornMachine:
    """Born Machine anomaly detector using Matrix Product States.

    Calculates anomaly score as the Negative Log-Likelihood (NLL):
    NLL(v) = -log P(v) = -2 * log |<Psi | Phi(v)>| + log <Psi | Psi>
    """

    def __init__(
        self,
        mps: MatrixProductState | None = None,
        num_sites: int | None = None,
        physical_dim: int = 2,
        bond_dimension: int = 32,
        min_prob_floor: float = 1e-15,
        seed: int | None = None,
    ):
        self.physical_dim = physical_dim
        self.bond_dimension = bond_dimension
        self.min_prob_floor = min_prob_floor

        if mps is not None:
            self.mps = mps
            self.num_sites = mps.num_sites
        elif num_sites is not None:
            self.num_sites = num_sites
            self.mps = MatrixProductState.initialize_random(
                num_sites=num_sites,
                physical_dim=physical_dim,
                bond_dimension=bond_dimension,
                seed=seed,
            )
        else:
            self.mps = None
            self.num_sites = 0

        self._norm_sq: float | None = None

    @property
    def partition_function(self) -> float:
        """Returns the partition function Z = <Psi | Psi>."""
        if self._norm_sq is None:
            if self.mps is None:
                return 1.0
            self._norm_sq = max(self.mps.norm_squared(), 1e-12)
        return self._norm_sq

    def reset_cached_norm(self) -> None:
        """Invalidates cached partition function after tensor updates."""
        self._norm_sq = None

    def probability(self, phi_sample: np.ndarray) -> float:
        """Computes Born rule probability P(v) = |<Psi|Phi(v)>|^2 / Z."""
        if self.mps is None:
            raise RuntimeError("MPSBornMachine has not been initialized.")
        amp = self.mps.contract_sample(phi_sample)
        z = self.partition_function
        prob = (amp**2) / z
        return float(np.clip(prob, self.min_prob_floor, 1.0))

    def score_samples(self, phi_batch: np.ndarray) -> np.ndarray:
        """Computes Negative Log-Likelihood (NLL) for a batch of encoded samples.

        Args:
            phi_batch: Array of shape (B, num_sites, physical_dim).

        Returns:
            Array of shape (B,) containing NLL scores. Higher score = more anomalous.
        """
        if self.mps is None:
            raise RuntimeError("MPSBornMachine has not been initialized.")

        amplitudes = self.mps.contract_batch(phi_batch)
        z = self.partition_function
        log_z = np.log(z)

        # Numerical underflow floor
        abs_amps = np.maximum(np.abs(amplitudes), 1e-15)
        nll_scores = -2.0 * np.log(abs_amps) + log_z
        return np.asarray(nll_scores, dtype=np.float64)

    def save(self, filepath: str) -> None:
        """Saves MPS tensors to a compressed .npz archive."""
        if self.mps is None:
            raise RuntimeError("Cannot save uninitialized MPSBornMachine.")
        tensor_dict = {f"tensor_{i}": t for i, t in enumerate(self.mps.tensors)}
        tensor_dict["num_sites"] = np.array(self.num_sites)
        tensor_dict["physical_dim"] = np.array(self.physical_dim)
        tensor_dict["bond_dimension"] = np.array(self.bond_dimension)
        np.savez_compressed(filepath, **tensor_dict)

    @classmethod
    def load(cls, filepath: str) -> "MPSBornMachine":
        """Loads an MPSBornMachine from a compressed .npz archive."""
        data = np.load(filepath)
        num_sites = int(data["num_sites"])
        physical_dim = int(data["physical_dim"])
        bond_dimension = int(data["bond_dimension"])

        tensors = [data[f"tensor_{i}"] for i in range(num_sites)]
        mps = MatrixProductState(tensors)

        bm = cls(
            mps=mps, num_sites=num_sites, physical_dim=physical_dim, bond_dimension=bond_dimension
        )
        return bm
