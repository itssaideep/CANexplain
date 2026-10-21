"""Temporal sliding window construction and forensic provenance tracking."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class WindowProvenance:
    """Forensic provenance metadata for a single temporal window."""
    window_idx: int
    start_frame_idx: int
    end_frame_idx: int
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    frame_count: int


def construct_sliding_windows(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    window_size: int = 64,
    stride: int = 32,
) -> Tuple[np.ndarray, np.ndarray, List[WindowProvenance]]:
    """Constructs temporal sliding windows from sequential CAN frames.

    Args:
        feature_matrix: Array of shape (T, F) containing normalized features.
        labels: Array of shape (T,) containing binary frame-level labels.
        timestamps: Array of shape (T,) containing frame timestamps.
        window_size: Number of consecutive frames per window (W).
        stride: Step size between consecutive windows (S).

    Returns:
        windows: Array of shape (N_windows, W, F).
        window_labels: Array of shape (N_windows,) where window is 1 if any frame is 1.
        provenance_list: List of WindowProvenance records for forensic traceability.
    """
    total_frames = len(feature_matrix)
    if total_frames < window_size:
        raise ValueError(
            f"Total frames ({total_frames}) is less than window size ({window_size})."
        )

    # Number of valid windows
    num_windows = (total_frames - window_size) // stride + 1

    feature_windows = []
    window_labels = []
    provenance_list = []

    for w_i in range(num_windows):
        start_idx = w_i * stride
        end_idx = start_idx + window_size

        sub_feats = feature_matrix[start_idx:end_idx]
        sub_labels = labels[start_idx:end_idx]
        sub_times = timestamps[start_idx:end_idx]

        feature_windows.append(sub_feats)
        # Window-level label: positive if any frame is attack
        window_labels.append(1 if np.any(sub_labels == 1) else 0)

        prov = WindowProvenance(
            window_idx=w_i,
            start_frame_idx=start_idx,
            end_frame_idx=end_idx - 1,
            start_timestamp=float(sub_times[0]),
            end_timestamp=float(sub_times[-1]),
            duration_seconds=float(sub_times[-1] - sub_times[0]),
            frame_count=window_size
        )
        provenance_list.append(prov)

    return (
        np.array(feature_windows, dtype=np.float64),
        np.array(window_labels, dtype=np.int32),
        provenance_list
    )
