"""Temporal sliding window construction and forensic provenance tracking."""

from dataclasses import dataclass

import numpy as np


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
    attack_frame_count: int = 0
    attack_frame_ratio: float = 0.0
    subclass: str | None = None
    dataset_name: str | None = None
    trace_name: str | None = None


def construct_sliding_windows(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    window_size: int = 64,
    stride: int = 32,
    label_threshold: float = 0.0,
    subclasses: np.ndarray | list[str] | None = None,
    dataset_name: str | None = None,
    trace_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[WindowProvenance]]:
    """Constructs temporal sliding windows from sequential CAN frames.

    Supports configurable window labeling:
    - label_threshold = 0.0 (default): Any attack frame labels the window as attack.
    - label_threshold > 0.0: Attack frame ratio (count / W) must strictly exceed threshold.
    - Stores exact attack_frame_count and attack_frame_ratio in provenance for transparent audit.

    Args:
        feature_matrix: Array of shape (T, F) containing normalized features.
        labels: Array of shape (T,) containing binary frame-level labels.
        timestamps: Array of shape (T,) containing frame timestamps.
        window_size: Number of consecutive frames per window (W).
        stride: Step size between consecutive windows (S).
        label_threshold: Attack frame fraction threshold for binary window label.
        subclasses: Optional array of frame-level attack subclass strings.
        dataset_name: Optional dataset provenance name.
        trace_name: Optional capture trace provenance name.

    Returns:
        windows: Array of shape (N_windows, W, F).
        window_labels: Array of shape (N_windows,) binary labels.
        provenance_list: List of WindowProvenance records for forensic traceability.
    """
    total_frames = len(feature_matrix)
    if total_frames < window_size:
        raise ValueError(f"Total frames ({total_frames}) is less than window size ({window_size}).")

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

        # Quantitative attack metrics per window
        attack_count = int(np.sum(sub_labels == 1))
        attack_ratio = float(attack_count / window_size)
        is_attack = 1 if attack_ratio > label_threshold else 0
        window_labels.append(is_attack)

        # Detect primary attack subclass in window if provided
        window_subclass: str | None = None
        if subclasses is not None and len(subclasses) > 0:
            sub_classes = subclasses[start_idx:end_idx]
            attack_subclasses = [
                c for c, lbl in zip(sub_classes, sub_labels, strict=True) if lbl == 1
            ]
            if attack_subclasses:
                # Most common attack subclass
                window_subclass = max(set(attack_subclasses), key=attack_subclasses.count)

        prov = WindowProvenance(
            window_idx=w_i,
            start_frame_idx=start_idx,
            end_frame_idx=end_idx - 1,
            start_timestamp=float(sub_times[0]),
            end_timestamp=float(sub_times[-1]),
            duration_seconds=float(sub_times[-1] - sub_times[0]),
            frame_count=window_size,
            attack_frame_count=attack_count,
            attack_frame_ratio=attack_ratio,
            subclass=window_subclass,
            dataset_name=dataset_name,
            trace_name=trace_name,
        )
        provenance_list.append(prov)

    return (
        np.array(feature_windows, dtype=np.float64),
        np.array(window_labels, dtype=np.int32),
        provenance_list,
    )
