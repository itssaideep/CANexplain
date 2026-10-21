"""Isolation Forest unsupervised anomaly detector baseline."""

from typing import Optional
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """Wrapper around scikit-learn Isolation Forest for windowed CAN tensors."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.01,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    def _extract_summary_features(self, windows: np.ndarray) -> np.ndarray:
        """Extracts summary statistics (mean, std, min, max) across window timesteps."""
        # windows shape: (N, W, F)
        mean_feat = np.mean(windows, axis=1)
        std_feat = np.std(windows, axis=1)
        min_feat = np.min(windows, axis=1)
        max_feat = np.max(windows, axis=1)
        return np.concatenate([mean_feat, std_feat, min_feat, max_feat], axis=1)

    def fit(self, normal_windows: np.ndarray) -> "IsolationForestDetector":
        """Fits Isolation Forest on normal traffic windows."""
        x_summary = self._extract_summary_features(normal_windows)
        self.model.fit(x_summary)
        self._is_fitted = True
        return self

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        """Scores windows. Returns anomaly scores where higher indicates greater anomaly."""
        if not self._is_fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before scoring.")
        x_summary = self._extract_summary_features(windows)
        # scikit-learn score_samples returns negative anomaly score (lower is more anomalous)
        # Invert so higher score = more anomalous
        return -self.model.score_samples(x_summary)
