"""Adapter for can-train-and-test-v1.5 dataset format."""

from typing import Optional
import numpy as np
import pandas as pd

from canexplain.data.adapters.base import BaseCANAdapter, CANONICAL_DATAFRAME_COLUMNS
from canexplain.data.schema import normalize_payload_bytes, parse_can_id


class SharedCANAdapter(BaseCANAdapter):
    """Adapter for the can-train-and-test-v1.5 benchmark dataset.

    Format:
        timestamp,arbitration_id,data_field,attack
    """

    def __init__(self, vehicle_id: str = "vehicle_set_01"):
        super().__init__(dataset_name="can-train-and-test-v1.5", vehicle_id=vehicle_id)

    def parse_raw_dataframe(
        self,
        df_raw: pd.DataFrame,
        trace_name: Optional[str] = None,
    ) -> pd.DataFrame:
        df = df_raw.copy()

        # Map column names
        col_map = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if "time" in cl or cl == "t":
                col_map[c] = "timestamp"
            elif "id" in cl:
                col_map[c] = "arbitration_id"
            elif "data" in cl or "payload" in cl:
                col_map[c] = "data_hex"
            elif "attack" in cl or "label" in cl:
                col_map[c] = "label"

        df = df.rename(columns=col_map)

        # Retain original frame order
        df["frame_index"] = np.arange(len(df), dtype=np.int64)

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

        # Parse IDs (standard 11-bit)
        df["arbitration_id"] = df["arbitration_id"].apply(lambda v: parse_can_id(v, allow_extended=False))
        df["id_hex"] = df["arbitration_id"].apply(lambda v: f"{v:03X}")
        df["is_extended_id"] = False

        # Parse payloads
        if "data_hex" in df.columns:
            def _parse_payload(d):
                s = str(d).strip()
                b = normalize_payload_bytes(s)
                raw_len = min(len(s) // 2, 8)
                return b, raw_len

            parsed = df["data_hex"].apply(_parse_payload)
            df["payload_bytes"] = [p[0] for p in parsed]
            df["raw_payload_len"] = [p[1] for p in parsed]
            df["dlc"] = df["raw_payload_len"]
            df["data_hex"] = df["payload_bytes"].apply(lambda b: b.hex().upper())
        else:
            df["payload_bytes"] = [b"\x00" * 8] * len(df)
            df["raw_payload_len"] = 0
            df["dlc"] = 0
            df["data_hex"] = "0000000000000000"

        # Binary label
        if "label" in df.columns:
            df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        else:
            df["label"] = 0

        # Subclass determination from trace name or label
        if trace_name and "dos" in trace_name.lower():
            df["subclass"] = np.where(df["label"] == 1, "DoS", "benign")
        elif trace_name and "standstill" in trace_name.lower():
            df["subclass"] = np.where(df["label"] == 1, "Standstill_Spoof", "benign")
        elif trace_name and "fuzz" in trace_name.lower():
            df["subclass"] = np.where(df["label"] == 1, "Fuzzing", "benign")
        else:
            df["subclass"] = np.where(df["label"] == 1, "attack", "benign")

        df["dataset_name"] = self.dataset_name
        df["vehicle_id"] = self.vehicle_id
        df["trace_name"] = trace_name or "unknown_trace"

        return df[CANONICAL_DATAFRAME_COLUMNS]
