"""Zero-leakage normalization pipeline for CAN traffic features."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class CANFeatureScaler:
    """Standardizes feature columns with min-max or robust scaling.

    Strictly fitted on training benign traffic to prevent train-test data leakage.
    """

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.min_vals: Dict[str, float] = {}
        self.max_vals: Dict[str, float] = {}
        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "CANFeatureScaler":
        """Fits scaler parameters on training DataFrame (benign traffic only)."""
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in ("label", "subclass")]

        self.feature_names = list(feature_cols)
        self.min_vals = {}
        self.max_vals = {}

        for col in self.feature_names:
            c_min = float(df[col].min())
            c_max = float(df[col].max())
            if c_max <= c_min:
                c_max = c_min + 1.0  # Prevent division by zero
            self.min_vals[col] = c_min
            self.max_vals[col] = c_max

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms a DataFrame into a normalized numpy array of shape (N, F)."""
        if not self._is_fitted:
            raise RuntimeError("CANFeatureScaler must be fitted before transforming data.")

        scaled = np.zeros((len(df), len(self.feature_names)), dtype=np.float64)
        for i, col in enumerate(self.feature_names):
            vals = df[col].to_numpy(dtype=np.float64)
            c_min = self.min_vals[col]
            c_max = self.max_vals[col]
            normed = (vals - c_min) / (c_max - c_min)
            scaled[:, i] = np.clip(normed, self.clip_min, self.clip_max)

        return scaled

    def fit_transform(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Fits and transforms in a single call."""
        return self.fit(df, feature_cols=feature_cols).transform(df)
