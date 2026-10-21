"""Tree Tensor Network (TTN) Born Machine for hierarchical CAN sequence anomaly detection."""

import json
import os
from typing import Any

import numpy as np

from canexplain.models.ttn.tree_tensor import BinaryTreeTensorNetwork


class TTNBornMachine:
    """Born Machine anomaly detector parameterized by a hierarchical Binary Tree Tensor Network.

    Assigns a generative probability to an automotive window sequence vector:
        P(x) = |<Psi_TTN | Phi(x)>|^2 / Z
    Anomaly score is the Negative Log-Likelihood (NLL):
        S(x) = -ln P(x) = -2 ln |<Psi_TTN | Phi(x)>| + ln Z
    """

    def __init__(
        self,
        num_leaves: int = 32,
        physical_dim: int = 2,
        bond_dimension: int = 16,
        seed: int | None = 42,
    ):
        self.num_leaves = num_leaves
        self.physical_dim = physical_dim
        self.bond_dimension = bond_dimension
        self.seed = seed
        self.ttn: BinaryTreeTensorNetwork | None = None
        self._cached_norm_sq: float | None = None

        if num_leaves > 0:
            self.ttn = BinaryTreeTensorNetwork.initialize_random(
                num_leaves=num_leaves,
                physical_dim=physical_dim,
                bond_dimension=bond_dimension,
                seed=seed,
            )

    @property
    def norm_squared(self) -> float:
        """Computes or retrieves cached state norm Z = <Psi | Psi>."""
        if self._cached_norm_sq is None:
            if self.ttn is None:
                raise ValueError("TTN is uninitialized.")
            self._cached_norm_sq = self.ttn.compute_norm_squared()
        return self._cached_norm_sq

    def reset_cached_norm(self) -> None:
        """Invalidates cached norm when tensor parameters are updated."""
        self._cached_norm_sq = None

    def score_samples(self, phi_batch: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        """Scores a batch of embedded sequence vectors via Negative Log-Likelihood (NLL).

        Args:
            phi_batch: Array of shape (B, L, d).
            eps: Numerical floor to prevent log(0).

        Returns:
            1D array of scalar NLL anomaly scores of shape (B,).
        """
        if self.ttn is None:
            raise ValueError("TTN Born Machine is uninitialized.")

        # Batch contraction: <Psi_TTN | Phi(x)>
        amplitudes = self.ttn.contract_batch(phi_batch)
        abs_amps = np.abs(amplitudes)
        clamped_amps = np.maximum(abs_amps, eps)

        z = self.norm_squared
        nll = -2.0 * np.log(clamped_amps) + np.log(max(z, eps))
        return nll.astype(np.float64)

    def save(self, filepath: str) -> None:
        """Saves TTN Born Machine configuration and weights to a JSON file."""
        if self.ttn is None:
            raise ValueError("Cannot save uninitialized TTN Born Machine.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "num_leaves": self.num_leaves,
            "physical_dim": self.physical_dim,
            "bond_dimension": self.bond_dimension,
            "seed": self.seed,
            "ttn_state": self.ttn.to_dict(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "TTNBornMachine":
        """Loads a persisted TTN Born Machine from a JSON file."""
        with open(filepath, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        machine = cls(
            num_leaves=data["num_leaves"],
            physical_dim=data["physical_dim"],
            bond_dimension=data["bond_dimension"],
            seed=data["seed"],
        )
        machine.ttn = BinaryTreeTensorNetwork.from_dict(data["ttn_state"])
        return machine
