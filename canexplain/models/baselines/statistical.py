"""Statistical deviation baseline anomaly detector based on CAN timing and frequency."""

from typing import Dict, Optional, Tuple
import numpy as np


class StatisticalDeviationDetector:
    """Unsupervised statistical baseline tracking inter-arrival times and message frequencies."""

    def __init__(self, dt_feature_idx: int = 1, id_feature_idx: int = 2):
        self.dt_feature_idx = dt_feature_idx
        self.id_feature_idx = id_feature_idx
        self.mean_dt: float = 0.0
        self.std_dt: float = 1.0
        self._is_fitted = False

    def fit(self, windows: np.ndarray) -> "StatisticalDeviationDetector":
        """Fits baseline statistics on normal traffic windows.

        Args:
            windows: Tensor of shape (N_windows, W, F).
        """
        # Extract all delta_t values from windows
        dts = windows[:, :, self.dt_feature_idx].flatten()
        self.mean_dt = float(np.mean(dts))
        self.std_dt = float(np.std(dts))
        if self.std_dt <= 0:
            self.std_dt = 1e-4

        self._is_fitted = True
        return self

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        """Computes statistical deviation score for each window.

        Higher score = greater anomaly.
        """
        if not self._is_fitted:
            raise RuntimeError("StatisticalDeviationDetector must be fitted before scoring.")

        # Window score: max z-score deviation of delta_t plus variance of dts in window
        dts = windows[:, :, self.dt_feature_idx]
        z_scores = np.abs(dts - self.mean_dt) / self.std_dt

        # Combine maximum deviation and mean deviation
        max_dev = np.max(z_scores, axis=1)
        mean_dev = np.mean(z_scores, axis=1)

        scores = 0.7 * max_dev + 0.3 * mean_dev
        return scores
