"""Adapter for Korea University HCRL Car-Hacking Dataset."""

from typing import Optional
import numpy as np
import pandas as pd

from canexplain.data.adapters.base import BaseCANAdapter, CANONICAL_DATAFRAME_COLUMNS
from canexplain.data.schema import normalize_payload_bytes, parse_can_id


class HCRLCarHackingAdapter(BaseCANAdapter):
    """Adapter for HCRL Car-Hacking Dataset (Song et al. / Korea University HCRL).

    Vehicle: Hyundai YF Sonata.
    Layout:
        Timestamp, CAN ID, DLC, DATA[0], DATA[1], DATA[2], DATA[3], DATA[4], DATA[5], DATA[6], DATA[7], Flag
        Where Flag == 'R' (Regular/Benign) or 'T' (Transmitted/Attack).
    Attacks:
        - DoS (0x000 message injection)
        - Fuzzy (random ID and payload injection)
        - RPM Spoofing (CAN ID 0x316)
        - Gear Spoofing (CAN ID 0x43F)
    """

    def __init__(self, vehicle_id: str = "hyundai_yf_sonata"):
        super().__init__(dataset_name="HCRL_CarHacking", vehicle_id=vehicle_id)

    def parse_raw_dataframe(
        self,
        df_raw: pd.DataFrame,
        trace_name: Optional[str] = None,
    ) -> pd.DataFrame:
        df = df_raw.copy()

        # If columns match standard HCRL layout by position or header
        if df.shape[1] >= 11:
            first_col_name = str(df.columns[0]).lower()
            # If no named header, rename positional columns
            if not ("time" in first_col_name or "timestamp" in first_col_name):
                cols = ["timestamp", "arbitration_id", "dlc"] + [f"b_{i}" for i in range(8)] + ["flag"]
                if df.shape[1] > len(cols):
                    cols += [f"extra_{i}" for i in range(df.shape[1] - len(cols))]
                df.columns = cols[:df.shape[1]]
            else:
                # Rename named header
                col_map = {}
                for c in df.columns:
                    cl = str(c).lower().strip()
                    if "time" in cl:
                        col_map[c] = "timestamp"
                    elif "id" in cl:
                        col_map[c] = "arbitration_id"
                    elif "dlc" in cl:
                        col_map[c] = "dlc"
                    elif "flag" in cl or "label" in cl or "class" in cl:
                        col_map[c] = "flag"
                df = df.rename(columns=col_map)
        else:
            raise ValueError(f"HCRL trace requires at least 11 columns, got {df.shape[1]}")

        df["frame_index"] = np.arange(len(df), dtype=np.int64)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

        # Standard 11-bit CAN ID
        df["arbitration_id"] = df["arbitration_id"].apply(lambda v: parse_can_id(v, allow_extended=False))
        df["id_hex"] = df["arbitration_id"].apply(lambda v: f"{v:03X}")
        df["is_extended_id"] = False

        # DLC
        if "dlc" in df.columns:
            df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(8).astype(int)
        else:
            df["dlc"] = 8

        # Combine byte columns b_0 .. b_7 or DATA[0..7] into payload bytes
        byte_cols = [c for c in df.columns if any(p in str(c).lower() for p in ("b_", "data", "byte"))]
        if len(byte_cols) >= 8:
            selected_byte_cols = byte_cols[:8]
            # Each byte is typically 2 hex digits like '00', 'fe', or integer
            def _extract_bytes(row):
                raw_list = []
                for col in selected_byte_cols:
                    val = str(row[col]).strip()
                    try:
                        raw_list.append(int(val, 16))
                    except ValueError:
                        try:
                            raw_list.append(int(val, 10))
                        except ValueError:
                            raw_list.append(0)
                b = bytes(raw_list[:8])
                return b

            df["payload_bytes"] = df.apply(_extract_bytes, axis=1)
            df["raw_payload_len"] = df["dlc"].clip(0, 8)
            df["data_hex"] = df["payload_bytes"].apply(lambda b: b.hex().upper())
        else:
            # Fallback if hex payload string column exists
            df["payload_bytes"] = [b"\x00" * 8] * len(df)
            df["raw_payload_len"] = 8
            df["data_hex"] = "0000000000000000"

        # Flag mapping: 'R' -> 0 (benign), 'T' -> 1 (attack)
        flag_col = "flag" if "flag" in df.columns else ("label" if "label" in df.columns else None)
        if flag_col:
            def _map_flag(val):
                s = str(val).strip().upper()
                if s in ("R", "0", "BENIGN", "NORMAL"):
                    return 0
                return 1

            df["label"] = df[flag_col].apply(_map_flag)
        else:
            df["label"] = 0

        # Subclass determination from trace name or attack ID
        subclass_name = "attack"
        if trace_name:
            tn = trace_name.lower()
            if "dos" in tn:
                subclass_name = "DoS"
            elif "fuzzy" in tn:
                subclass_name = "Fuzzy"
            elif "rpm" in tn:
                subclass_name = "RPM_Spoof"
            elif "gear" in tn:
                subclass_name = "Gear_Spoof"

        df["subclass"] = np.where(df["label"] == 1, subclass_name, "benign")
        df["dataset_name"] = self.dataset_name
        df["vehicle_id"] = self.vehicle_id
        df["trace_name"] = trace_name or "hcrl_trace"

        return df[CANONICAL_DATAFRAME_COLUMNS]
