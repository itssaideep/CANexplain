"""Quantum-inspired local feature maps for tensor network representations."""

import numpy as np


def trigonometric_feature_map(x: np.ndarray) -> np.ndarray:
    """Maps normalized continuous values in [0, 1] to 2D state vectors.

    |phi(x)> = [cos(pi/2 * x), sin(pi/2 * x)]^T

    Ensures unit norm: <phi(x)|phi(x)> = cos^2(pi/2 * x) + sin^2(pi/2 * x) = 1.0.

    Args:
        x: Array of arbitrary shape with elements in [0, 1].

    Returns:
        Array of shape (*x.shape, 2) representing the mapped quantum-like state vectors.
    """
    x_clamped = np.clip(x, 0.0, 1.0)
    theta = 0.5 * np.pi * x_clamped

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Stack along last axis
    return np.stack([cos_theta, sin_theta], axis=-1)


def encode_feature_matrix(
    normalized_features: np.ndarray,
    encoding_type: str = "trigonometric"
) -> np.ndarray:
    """Encodes a normalized feature matrix into local tensor state vectors.

    Args:
        normalized_features: Array of shape (N_samples, N_features) in [0, 1].
        encoding_type: 'trigonometric' (p=2).

    Returns:
        Array of shape (N_samples, N_features, 2).
    """
    if encoding_type == "trigonometric":
        return trigonometric_feature_map(normalized_features)
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")


class MeanPooledFeatureMap:
    """Explicitly named pooled-feature baseline encoder.

    Averages features across the temporal window dimension W before trigonometric mapping:
    (N_windows, W, F) -> mean over W -> (N_windows, F) -> (N_windows, F, 2).

    Note:
        Temporal order is collapsed by the mean; single-frame anomalies are diluted by 1/W.
        Retained as an explicit comparative baseline.
    """

    def __init__(self, encoding_type: str = "trigonometric"):
        self.encoding_type = encoding_type

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Encodes windows via temporal mean-pooling.

        Args:
            windows: Array of shape (N_windows, W, F) or (N_samples, F).

        Returns:
            Array of shape (N_windows, F, 2).
        """
        if windows.ndim == 3:
            pooled = np.mean(windows, axis=1)
        elif windows.ndim == 2:
            pooled = windows
        else:
            raise ValueError(f"Expected 2D or 3D array; got ndim={windows.ndim}")
        return encode_feature_matrix(pooled, encoding_type=self.encoding_type)


class SequencePreservingEncoder:
    """Sequence-preserving quantum encoder for CAN temporal windows.

    Preserves sequential frame ordering within temporal windows:
    - Mode 'flatten': Reshapes (N_windows, W, F) -> (N_windows, W * F) and encodes each site,
      preserving temporal order along an expanded 1D MPS chain.
    - Mode 'per_frame': Returns (N_windows, W, F, 2) for individual frame-level scoring
      without temporal pooling dilution.
    """

    def __init__(self, mode: str = "flatten", encoding_type: str = "trigonometric"):
        if mode not in ("flatten", "per_frame"):
            raise ValueError(f"mode must be 'flatten' or 'per_frame'; got {mode}")
        self.mode = mode
        self.encoding_type = encoding_type

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Encodes windows into sequence-preserving quantum state tensors.

        Args:
            windows: Array of shape (N_windows, W, F).

        Returns:
            If mode == 'flatten': Array of shape (N_windows, W * F, 2).
            If mode == 'per_frame': Array of shape (N_windows, W, F, 2).
        """
        if windows.ndim != 3:
            raise ValueError(f"SequencePreservingEncoder requires 3D (N, W, F); got {windows.shape}")
        n_win, w_len, f_dim = windows.shape
        if self.mode == "flatten":
            flattened = windows.reshape(n_win, w_len * f_dim)
            return encode_feature_matrix(flattened, encoding_type=self.encoding_type)
        else:
            return encode_feature_matrix(windows, encoding_type=self.encoding_type)
