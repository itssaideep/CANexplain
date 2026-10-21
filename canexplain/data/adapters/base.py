"""Base adapter interface and validation for automotive CAN-bus datasets."""

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

CANONICAL_DATAFRAME_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "id_hex",
    "dlc",
    "data_hex",
    "payload_bytes",
    "label",
    "subclass",
    "frame_index",
    "is_extended_id",
    "raw_payload_len",
    "dataset_name",
    "vehicle_id",
    "trace_name",
]


class BaseCANAdapter(ABC):
    """Abstract base adapter for normalizing heterogeneous CAN dataset logs into canonical format.

    Preserves original sequential frame order and timestamps without uncontrolled sorting.
    """

    def __init__(self, dataset_name: str, vehicle_id: str = "unknown"):
        self.dataset_name = dataset_name
        self.vehicle_id = vehicle_id

    @abstractmethod
    def parse_raw_dataframe(
        self,
        df_raw: pd.DataFrame,
        trace_name: str | None = None,
    ) -> pd.DataFrame:
        """Parses a dataset-specific raw DataFrame into canonical schema.

        Args:
            df_raw: Raw DataFrame as loaded from disk.
            trace_name: Optional trace or file identifier.

        Returns:
            Normalized DataFrame conforming to CANONICAL_DATAFRAME_COLUMNS.
        """
        pass

    def load_trace(
        self,
        filepath: str,
        sample_limit: int | None = None,
        offset: int = 0,
        trace_name: str | None = None,
    ) -> pd.DataFrame:
        """Loads and normalizes a trace file.

        Args:
            filepath: Path to the raw trace file (CSV / text).
            sample_limit: Maximum number of rows to load.
            offset: Number of header/skip rows to offset.
            trace_name: Trace identifier (defaults to filename if None).

        Returns:
            Canonical DataFrame.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trace file not found: {filepath}")

        if trace_name is None:
            trace_name = os.path.splitext(os.path.basename(filepath))[0]

        # Inspect delimiter
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
            delim = "," if "," in first_line else r"\s+"

        skiprows = range(1, offset + 1) if offset > 0 else None
        df_raw = pd.read_csv(
            filepath,
            sep=delim,
            skiprows=skiprows,
            nrows=sample_limit,
            engine="python",
        )

        canonical_df = self.parse_raw_dataframe(df_raw, trace_name=trace_name)
        self.validate_schema(canonical_df)
        return canonical_df

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validates that the DataFrame strictly adheres to the canonical CAN schema."""
        missing = [c for c in CANONICAL_DATAFRAME_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Canonical DataFrame is missing required columns: {missing}")

        # Check types
        if not np.issubdtype(df["timestamp"].dtype, np.number):
            raise TypeError("timestamp column must be numeric.")
        if not np.issubdtype(df["arbitration_id"].dtype, np.integer):
            raise TypeError("arbitration_id column must be integer.")
        if not np.issubdtype(df["label"].dtype, np.integer):
            raise TypeError("label column must be integer.")

        # Ensure no NaN values in essential fields
        if df["timestamp"].isna().any():
            raise ValueError("NaN values found in timestamp column.")
        if df["arbitration_id"].isna().any():
            raise ValueError("NaN values found in arbitration_id column.")
        if df["label"].isna().any():
            raise ValueError("NaN values found in label column.")
