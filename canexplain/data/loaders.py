"""Dataset loaders for automotive CAN-bus intrusion detection datasets."""

import os
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from canexplain.data.schema import CANFrame, parse_can_id, normalize_payload_bytes


# Standard base epoch timestamp: 2023-01-01 00:00:00.000 UTC
STANDARD_BASE_TIMESTAMP: float = 1672531200.0


def load_can_csv(
    filepath: str,
    standardize_timestamps: bool = True,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Loads and standardizes a CAN trace from CSV.

    Supports both can-train-and-test-v1.5 and HCRL format conventions.

    Args:
        filepath: Path to the CSV file.
        standardize_timestamps: If True, standardizes the trace starting timestamp
            to STANDARD_BASE_TIMESTAMP while preserving exact delta t.
        sample_limit: Optional row limit for rapid inspection/testing.

    Returns:
        Structured pandas DataFrame with canonical columns:
        ['timestamp', 'arbitration_id', 'id_hex', 'dlc', 'data_hex', 'payload_bytes', 'label', 'subclass']
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CAN trace file not found: {filepath}")

    # Inspect delimiter and header
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        has_comma = "," in first_line
        delim = "," if has_comma else r"\s+"

    df = pd.read_csv(filepath, sep=delim, nrows=sample_limit, engine="python")

    # Column name normalization
    col_map = {}
    for col in df.columns:
        c_lower = str(col).lower().strip()
        if "time" in c_lower or c_lower == "t":
            col_map[col] = "timestamp"
        elif "id" in c_lower or c_lower == "can_id":
            col_map[col] = "arbitration_id"
        elif "dlc" in c_lower:
            col_map[col] = "dlc"
        elif "data" in c_lower or "payload" in c_lower:
            col_map[col] = "data"
        elif "class" in c_lower or "flag" in c_lower or "label" in c_lower:
            col_map[col] = "label"
        elif "sub" in c_lower:
            col_map[col] = "subclass"

    # If header was missing or numeric
    if len(col_map) < 3:
        # Standard HCRL / raw layout: timestamp, id, dlc, byte0..byte7, flag
        if df.shape[1] >= 11:
            df.columns = ["timestamp", "arbitration_id", "dlc"] + [f"b{i}" for i in range(8)] + ["label"] + list(df.columns[12:])
            # Merge 8 bytes into hex string
            df["data"] = df[[f"b{i}" for i in range(8)]].astype(str).agg("".join, axis=1)
        elif df.shape[1] >= 4:
            cols = ["timestamp", "arbitration_id", "dlc", "data"]
            if df.shape[1] >= 5:
                cols.append("label")
            df = df.iloc[:, :len(cols)]
            df.columns = cols
    else:
        df = df.rename(columns=col_map)

    # Clean and parse types
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if standardize_timestamps and len(df) > 0:
        first_t = df["timestamp"].iloc[0]
        # Offset so first timestamp is STANDARD_BASE_TIMESTAMP
        df["timestamp"] = STANDARD_BASE_TIMESTAMP + (df["timestamp"] - first_t)

    # Standardize arbitration ID
    df["arbitration_id"] = df["arbitration_id"].apply(parse_can_id)
    df["id_hex"] = df["arbitration_id"].apply(lambda x: f"{x:03X}")

    # Standardize DLC
    if "dlc" in df.columns:
        df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(8).astype(int)
    else:
        df["dlc"] = 8

    # Standardize Payload
    if "data" in df.columns:
        df["payload_bytes"] = df["data"].apply(lambda p: normalize_payload_bytes(p))
    else:
        df["payload_bytes"] = [b"\x00" * 8] * len(df)
    df["data_hex"] = df["payload_bytes"].apply(lambda p: p.hex().upper())

    # Standardize Labels
    if "label" in df.columns:
        # Map R/T or text labels to 0/1
        def map_label(val):
            s = str(val).strip().upper()
            if s in ("0", "0.0", "R", "BENIGN", "NORMAL"):
                return 0
            return 1
        df["label"] = df["label"].apply(map_label)
    else:
        df["label"] = 0

    if "subclass" not in df.columns:
        df["subclass"] = "benign"

    return df[["timestamp", "arbitration_id", "id_hex", "dlc", "data_hex", "payload_bytes", "label", "subclass"]]
