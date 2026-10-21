"""Adapter for HCRL Survival Analysis CAN IDS dataset."""

import numpy as np
import pandas as pd

from canexplain.data.adapters.base import CANONICAL_DATAFRAME_COLUMNS, BaseCANAdapter
from canexplain.data.schema import normalize_payload_bytes, parse_can_id


class HCRLSurvivalAdapter(BaseCANAdapter):
    """Adapter for HCRL Survival Analysis Dataset (Han et al. / Korea University).

    Vehicle: Hyundai YF Sonata.
    Features sequential CAN message timing intervals and attack insertion sequences.
    """

    def __init__(self, vehicle_id: str = "hyundai_yf_sonata"):
        super().__init__(dataset_name="HCRL_Survival", vehicle_id=vehicle_id)

    def parse_raw_dataframe(
        self,
        df_raw: pd.DataFrame,
        trace_name: str | None = None,
    ) -> pd.DataFrame:
        df = df_raw.copy()

        # Handle headerless HCRL format: timestamp, can_id, dlc, b0..b7, label
        if df.shape[1] >= 11 and not any("time" in str(c).lower() for c in df.columns):
            cols = ["timestamp", "arbitration_id", "dlc"] + [f"b_{i}" for i in range(8)] + ["label"]
            if df.shape[1] > len(cols):
                cols += [f"extra_{i}" for i in range(df.shape[1] - len(cols))]
            df.columns = cols[: df.shape[1]]
        else:
            col_map = {}
            for c in df.columns:
                cl = str(c).lower().strip()
                if "time" in cl or cl == "t":
                    col_map[c] = "timestamp"
                elif "id" in cl:
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

        df["arbitration_id"] = df["arbitration_id"].apply(
            lambda v: parse_can_id(v, allow_extended=False)
        )
        df["id_hex"] = df["arbitration_id"].apply(lambda v: f"{v:03X}")
        df["is_extended_id"] = False

        if "dlc" in df.columns:
            df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").fillna(8).astype(int)
        else:
            df["dlc"] = 8

        byte_cols = [
            c for c in df.columns if any(p in str(c).lower() for p in ("b_", "data", "byte"))
        ]
        if len(byte_cols) >= 8:
            selected_byte_cols = byte_cols[:8]

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
                return bytes(raw_list[:8])

            df["payload_bytes"] = df.apply(_extract_bytes, axis=1)
            df["raw_payload_len"] = df["dlc"].clip(0, 8)
            df["data_hex"] = df["payload_bytes"].apply(lambda b: b.hex().upper())
        elif "data_hex" in df.columns:
            df["payload_bytes"] = df["data_hex"].apply(normalize_payload_bytes)
            df["raw_payload_len"] = df["dlc"].clip(0, 8)
            df["data_hex"] = df["payload_bytes"].apply(lambda b: b.hex().upper())
        else:
            df["payload_bytes"] = [b"\x00" * 8] * len(df)
            df["raw_payload_len"] = 8
            df["data_hex"] = "0000000000000000"

        if "label" in df.columns:

            def _map_label(val):
                s = str(val).strip().upper()
                if s in ("0", "0.0", "R", "BENIGN", "NORMAL"):
                    return 0
                return 1

            df["label"] = df["label"].apply(_map_label)
        else:
            df["label"] = 0

        df["subclass"] = np.where(df["label"] == 1, "injection_attack", "benign")
        df["dataset_name"] = self.dataset_name
        df["vehicle_id"] = self.vehicle_id
        df["trace_name"] = trace_name or "survival_trace"

        return df[CANONICAL_DATAFRAME_COLUMNS]
