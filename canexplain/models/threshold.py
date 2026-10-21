"""Threshold calibration utilities for anomaly detection scores."""

from typing import Tuple
import numpy as np


class ThresholdCalibrator:
    """Calibrates anomaly detection thresholds strictly on normal validation traffic."""

    def __init__(self, method: str = "percentile", percentile: float = 99.0, k_std: float = 3.0):
        self.method = method
        self.percentile = percentile
        self.k_std = k_std
        self.threshold: float = 0.0
        self._is_calibrated = False

    def fit(self, normal_scores: np.ndarray) -> "ThresholdCalibrator":
        """Calibrates threshold using scores computed on normal validation data."""
        if len(normal_scores) == 0:
            raise ValueError("Cannot calibrate threshold on empty normal score array.")

        scores = np.asarray(normal_scores, dtype=np.float64)

        if self.method == "percentile":
            self.threshold = float(np.percentile(scores, self.percentile))
        elif self.method == "std":
            mean = np.mean(scores)
            std = np.std(scores)
            self.threshold = float(mean + self.k_std * std)
        else:
            raise ValueError(f"Unknown threshold calibration method: {self.method}")

        self._is_calibrated = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Assigns binary anomaly labels (1 = anomaly, 0 = normal) based on threshold."""
        if not self._is_calibrated:
            raise RuntimeError("ThresholdCalibrator must be fitted before predicting.")
        return (np.asarray(scores, dtype=np.float64) > self.threshold).astype(int)
