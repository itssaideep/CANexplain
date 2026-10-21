"""Adapter for the ROAD (Real, Oak Ridge Automotive Dataset) CAN IDS benchmark."""

import numpy as np
import pandas as pd

from canexplain.data.adapters.base import CANONICAL_DATAFRAME_COLUMNS, BaseCANAdapter
from canexplain.data.schema import normalize_payload_bytes, parse_can_id


class ROADDatasetAdapter(BaseCANAdapter):
    """Adapter for ROAD dataset (Oak Ridge National Laboratory, Zenodo record 10462796).

    Vehicle: Kia Soul (dynamometer & road driving).
    Attacks:
        - correlated_signal_attack (fabrication)
        - max_engine_coolant_temp_attack (falsification)
        - max_speedometer_attack
        - reverse_light_off_attack
        - reverse_light_on_attack
    """

    def __init__(self, vehicle_id: str = "kia_soul"):
        super().__init__(dataset_name="ROAD", vehicle_id=vehicle_id)

    def parse_raw_dataframe(
        self,
        df_raw: pd.DataFrame,
        trace_name: str | None = None,
    ) -> pd.DataFrame:
        df = df_raw.copy()

        col_map = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if "time" in cl or cl == "t":
                col_map[c] = "timestamp"
            elif "id" in cl or "can_id" in cl:
                col_map[c] = "arbitration_id"
            elif "payload" in cl or "data" in cl:
                col_map[c] = "data_hex"
            elif "label" in cl or "class" in cl:
                col_map[c] = "subclass"

        df = df.rename(columns=col_map)

        df["frame_index"] = np.arange(len(df), dtype=np.int64)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

        # ROAD CAN IDs are standard 11-bit hex strings or ints
        df["arbitration_id"] = df["arbitration_id"].apply(
            lambda v: parse_can_id(v, allow_extended=False)
        )
        df["id_hex"] = df["arbitration_id"].apply(lambda v: f"{v:03X}")
        df["is_extended_id"] = False

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
            df["dlc"] = df["raw_payload_len"]
            df["data_hex"] = df["payload_bytes"].apply(lambda b: b.hex().upper())
        else:
            df["payload_bytes"] = [b"\x00" * 8] * len(df)
            df["raw_payload_len"] = 0
            df["dlc"] = 0
            df["data_hex"] = "0000000000000000"

        # Binary label and subclass
        if "subclass" in df.columns:

            def _map_road_label(val):
                s = str(val).lower().strip()
                if s in ("0", "normal", "benign", "false", "none"):
                    return 0, "benign"
                return 1, str(val).strip()

            mapped = df["subclass"].apply(_map_road_label)
            df["label"] = [m[0] for m in mapped]
            df["subclass"] = [m[1] for m in mapped]
        else:
            df["label"] = 0
            df["subclass"] = "benign"

        df["dataset_name"] = self.dataset_name
        df["vehicle_id"] = self.vehicle_id
        df["trace_name"] = trace_name or "road_trace"

        return df[CANONICAL_DATAFRAME_COLUMNS]
