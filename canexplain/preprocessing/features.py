"""Feature engineering for automotive CAN-bus frames grounded in can-sleuth."""

import math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_inter_arrival_times(df: pd.DataFrame) -> pd.Series:
    """Computes inter-arrival time (delta t) per arbitration ID.

    For each arbitration ID, delta_t_i = t_i - t_{i-1}.
    Initial frame for an ID receives the median delta t of the trace.
    """
    df_sorted = df.sort_values("timestamp")
    delta_t = df_sorted.groupby("arbitration_id")["timestamp"].diff()
    median_dt = delta_t.median()
    if pd.isna(median_dt) or median_dt <= 0:
        median_dt = 0.020  # default 20ms fallback
    return delta_t.fillna(median_dt)


def compute_byte_entropy(payload_bytes_series: pd.Series) -> pd.Series:
    """Computes the Shannon entropy across the 8 payload bytes for each frame."""
    def _entropy(b: bytes) -> float:
        if not b:
            return 0.0
        counts = {}
        for byte_val in b:
            counts[byte_val] = counts.get(byte_val, 0) + 1
        n = len(b)
        ent = 0.0
        for count in counts.values():
            p = count / n
            ent -= p * math.log2(p)
        return ent

    return payload_bytes_series.apply(_entropy)


CANONICAL_10_FEATURES: List[str] = [
    "delta_t",
    "arbitration_id",
    "byte_0",
    "byte_1",
    "byte_2",
    "byte_3",
    "byte_4",
    "byte_5",
    "byte_6",
    "byte_7",
]

LEGACY_11_FEATURES: List[str] = [
    "timestamp_offset",
    "delta_t",
    "arbitration_id",
    "byte_0",
    "byte_1",
    "byte_2",
    "byte_3",
    "byte_4",
    "byte_5",
    "byte_6",
    "byte_7",
]


def extract_canonical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts exactly the 10 stationary features grounded in CAN-bus IDS literature:
    [delta_t, arbitration_id, byte_0, ..., byte_7].

    Excludes monotonic global timestamp offsets to guarantee stationary distributions
    and prevent out-of-distribution quantum phase oscillations.

    Returns:
        DataFrame containing the 10 numeric feature columns plus label and subclass.
    """
    return extract_10_features(df, stationary_only=True)


def extract_10_features(
    df: pd.DataFrame,
    stationary_only: bool = True,
    include_timestamp_offset: bool = False,
) -> pd.DataFrame:
    """Extracts features from CAN traffic DataFrame.

    Resolves the 10 vs 11 feature discrepancy:
    - When stationary_only=True (default) and include_timestamp_offset=False:
      Extracts EXACTLY 10 stationary features:
      [delta_t, arbitration_id, byte_0, ..., byte_7].
    - When include_timestamp_offset=True or stationary_only=False:
      Extracts the legacy 11 features:
      [timestamp_offset, delta_t, arbitration_id, byte_0, ..., byte_7].

    Args:
        df: Input DataFrame containing CAN frames.
        stationary_only: If True and include_timestamp_offset is False, returns 10 stationary features.
        include_timestamp_offset: If True, prepends monotonic timestamp offset (11 features).

    Returns:
        DataFrame with feature columns plus label and subclass.
    """
    feat_df = pd.DataFrame(index=df.index)

    include_offset = include_timestamp_offset or (not stationary_only)

    if include_offset:
        # Monotonic timestamp offset (legacy 11-feature variant)
        t_start = df["timestamp"].iloc[0] if len(df) > 0 else 0.0
        feat_df["timestamp_offset"] = df["timestamp"] - t_start

    # 1. Inter-arrival time delta_t per arbitration ID
    feat_df["delta_t"] = compute_inter_arrival_times(df)

    # 2. Arbitration ID
    feat_df["arbitration_id"] = df["arbitration_id"].astype(float)

    # 3-10. Subdivided 8 payload bytes
    for i in range(8):
        feat_df[f"byte_{i}"] = df["payload_bytes"].apply(
            lambda p: float(p[i]) if len(p) > i else 0.0
        )

    # Carry forward labels
    if "label" in df.columns:
        feat_df["label"] = df["label"].astype(int)
    else:
        feat_df["label"] = 0

    if "subclass" in df.columns:
        feat_df["subclass"] = df["subclass"]
    else:
        feat_df["subclass"] = "benign"

    return feat_df
