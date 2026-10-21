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
