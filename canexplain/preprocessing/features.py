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


def extract_10_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the 10-feature subdivided byte representation verified in can-sleuth.

    Features:
        1. timestamp_offset: Elapsed seconds from start of trace.
        2. delta_t: Inter-arrival time for the same arbitration ID.
        3. arbitration_id: Raw arbitration ID value.
        4-11. byte_0 through byte_7: Unsigned byte values (0-255).

    Returns:
        DataFrame with the 10 numeric feature columns plus label and subclass.
    """
    feat_df = pd.DataFrame(index=df.index)

    # 1. Timestamp offset
    t_start = df["timestamp"].iloc[0] if len(df) > 0 else 0.0
    feat_df["timestamp_offset"] = df["timestamp"] - t_start

    # 2. Inter-arrival time delta_t
    feat_df["delta_t"] = compute_inter_arrival_times(df)

    # 3. Arbitration ID
    feat_df["arbitration_id"] = df["arbitration_id"].astype(float)

    # 4-11. Subdivided 8 payload bytes
    for i in range(8):
        feat_df[f"byte_{i}"] = df["payload_bytes"].apply(lambda p: float(p[i]) if len(p) > i else 0.0)

    # Carry forward labels
    feat_df["label"] = df["label"].astype(int)
    feat_df["subclass"] = df.get("subclass", "benign")

    return feat_df
