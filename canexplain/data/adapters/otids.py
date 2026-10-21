"""Adapter for OTIDS (Oakland / ETRI CAN Intrusion Dataset)."""

import numpy as np
import pandas as pd

from canexplain.data.adapters.base import CANONICAL_DATAFRAME_COLUMNS, BaseCANAdapter
from canexplain.data.schema import normalize_payload_bytes, parse_can_id


class OTIDSAdapter(BaseCANAdapter):
    """Adapter for OTIDS (CAN Intrusion Dataset, Kang et al. / ETRI).

    Vehicle: KIA Soul.
    Attacks:
        - DoS Attack
        - Fuzzy Attack
        - Impersonation Attack (spoofing legitimate arbitration IDs, e.g., 0x164)
    """

    def __init__(self, vehicle_id: str = "kia_soul"):
        super().__init__(dataset_name="OTIDS", vehicle_id=vehicle_id)

    def parse_raw_dataframe(
        self,
        df_raw: pd.DataFrame,
        trace_name: str | None = None,
    ) -> pd.DataFrame:
        df = df_raw.copy()

        # If columns match OTIDS layout: Timestamp, ID, DLC, Data (or Data[0..7]), Flag
        col_map = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if "time" in cl or cl == "t":
                col_map[c] = "timestamp"
            elif "id" in cl or "can_id" in cl:
                col_map[c] = "arbitration_id"
            elif "dlc" in cl:
                col_map[c] = "dlc"
            elif "data" in cl or "payload" in cl:
                col_map[c] = "data_hex"
            elif "flag" in cl or "label" in cl or "attack" in cl:
                col_map[c] = "label"

        df = df.rename(columns=col_map)

        df["frame_index"] = np.arange(len(df), dtype=np.int64)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

        # Allow extended IDs if OTIDS has 29-bit CAN IDs
        def _parse_id(v):
            cleaned = str(v).strip()
            is_ext = len(cleaned.replace("0x", "")) > 3 or (isinstance(v, int) and v > 0x7FF)
            parsed = parse_can_id(v, allow_extended=True)
            return parsed, is_ext

        id_info = df["arbitration_id"].apply(_parse_id)
        df["arbitration_id"] = [i[0] for i in id_info]
        df["is_extended_id"] = [i[1] for i in id_info]
        df["id_hex"] = [
            f"{aid:08X}" if ext else f"{aid:03X}"
            for aid, ext in zip(df["arbitration_id"], df["is_extended_id"], strict=True)
        ]

        # DLC
        if "dlc" in df.columns:
            df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(8).astype(int)
        else:
            df["dlc"] = 8

        # Payloads
        if "data_hex" in df.columns:

            def _parse_payload(d):
                s = str(d).strip()
                b = normalize_payload_bytes(s)
                raw_len = min(len(s) // 2, 8)
                return b, raw_len

            parsed = df["data_hex"].apply(_parse_payload)
            df["payload_bytes"] = [p[0] for p in parsed]
            df["raw_payload_len"] = [p[1] for p in parsed]
            df["data_hex"] = df["payload_bytes"].apply(lambda b: b.hex().upper())
        else:
            df["payload_bytes"] = [b"\x00" * 8] * len(df)
            df["raw_payload_len"] = 8
            df["data_hex"] = "0000000000000000"

        # Labels
        if "label" in df.columns:

            def _map_otids_label(val):
                s = str(val).strip().upper()
                if s in ("0", "0.0", "R", "BENIGN", "NORMAL"):
                    return 0
                return 1

            df["label"] = df["label"].apply(_map_otids_label)
        else:
            df["label"] = 0

        # Subclass
        subclass_name = "attack"
        if trace_name:
            tn = trace_name.lower()
            if "dos" in tn:
                subclass_name = "DoS"
            elif "fuzzy" in tn:
                subclass_name = "Fuzzy"
            elif "impersonation" in tn:
                subclass_name = "Impersonation"

        df["subclass"] = np.where(df["label"] == 1, subclass_name, "benign")
        df["dataset_name"] = self.dataset_name
        df["vehicle_id"] = self.vehicle_id
        df["trace_name"] = trace_name or "otids_trace"

        return df[CANONICAL_DATAFRAME_COLUMNS]
