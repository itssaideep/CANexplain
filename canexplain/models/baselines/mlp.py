"""Multi-Layer Perceptron (MLP) supervised neural reference baseline."""

from typing import Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier


class MLPDetector:
    """Supervised feed-forward neural baseline representing DNN automotive IDSs."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
        )
        self._is_fitted = False

    def _extract_summary_features(self, windows: np.ndarray) -> np.ndarray:
        mean_feat = np.mean(windows, axis=1)
        std_feat = np.std(windows, axis=1)
        min_feat = np.min(windows, axis=1)
        max_feat = np.max(windows, axis=1)
        return np.concatenate([mean_feat, std_feat, min_feat, max_feat], axis=1)

    def fit(self, windows: np.ndarray, labels: np.ndarray) -> "MLPDetector":
        """Fits supervised MLP on labeled windows."""
        x_summary = self._extract_summary_features(windows)
        self.model.fit(x_summary, labels)
        self._is_fitted = True
        return self

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        """Returns predicted probability of attack for each window."""
        if not self._is_fitted:
            raise RuntimeError("MLPDetector must be fitted before scoring.")
        x_summary = self._extract_summary_features(windows)
        if len(self.model.classes_) == 1:
            # Only single class observed during training
            return np.zeros(len(windows))
        probs = self.model.predict_proba(x_summary)
        # Class 1 is attack
        return probs[:, 1]
