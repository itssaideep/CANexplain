"""Semantic CAN signal tokenizer and cross-vehicle state normalization."""

from typing import Any

import numpy as np
import pandas as pd

from canexplain.data.dbc.opendbc_adapter import UNIVERSAL_DYNAMICS_SIGNALS
from canexplain.data.dbc.signal_schema import DBCDatabase


class SemanticCANTokenizer:
    """Decodes raw CAN frames via a vehicle DBC and constructs normalized physical state vectors.

    Scientific Principle:
        Raw arbitration IDs and byte offsets differ across automotive manufacturers.
        However, physical kinematic signals (speed, steering, RPM, throttle) obey identical
        automotive engineering laws. By mapping OEM bitfields into a standardized semantic
        space, models trained on Vehicle A can evaluate Vehicle B without architectural shift.
    """

    def __init__(
        self,
        dbc: DBCDatabase,
        target_signals: list[str] | None = None,
    ):
        self.dbc = dbc
        self.target_signals = (
            target_signals if target_signals is not None else list(UNIVERSAL_DYNAMICS_SIGNALS)
        )
        # Signal normalization bounds: {name: (min_phys, max_phys)}
        self.bounds: dict[str, tuple[float, float]] = self._extract_signal_bounds()

    def _extract_signal_bounds(self) -> dict[str, tuple[float, float]]:
        bounds = {}
        for msg in self.dbc.messages.values():
            for sig in msg.signals:
                bounds[sig.name] = (sig.min_physical, sig.max_physical)
        return bounds

    def tokenize_stream(self, df_frames: pd.DataFrame) -> pd.DataFrame:
        """Processes a sequential CAN frame DataFrame into normalized semantic states.

        Args:
            df_frames: DataFrame with columns ['timestamp', 'arbitration_id', 'payload_bytes']
                       and optionally ['label', 'subclass'].

        Returns:
            DataFrame where columns are normalized signals in [0, 1] plus labels.
        """
        # Current rolling state of physical values (Sample & Hold)
        current_state: dict[str, float] = {
            s: self.bounds.get(s, (0.0, 100.0))[0] for s in self.target_signals
        }

        output_rows: list[dict[str, Any]] = []

        for _, row in df_frames.iterrows():
            aid = int(row["arbitration_id"])
            payload = row["payload_bytes"]
            if isinstance(payload, (list, np.ndarray)):
                payload_bytes = bytes(payload)
            elif isinstance(payload, bytes):
                payload_bytes = payload
            else:
                payload_bytes = b"\x00" * 8

            # Decode any signals in this message
            decoded = self.dbc.decode_frame(aid, payload_bytes)
            for s_name, val in decoded.items():
                if s_name in current_state:
                    current_state[s_name] = val

            # Normalize rolling state to [0.0, 1.0]
            norm_vector: dict[str, Any] = {}
            for s_name in self.target_signals:
                min_p, max_p = self.bounds.get(s_name, (0.0, 100.0))
                raw_val = current_state[s_name]
                rng = max_p - min_p if max_p > min_p else 1.0
                norm_val = float(np.clip((raw_val - min_p) / rng, 0.0, 1.0))
                norm_vector[s_name] = norm_val

            norm_vector["timestamp"] = float(row.get("timestamp", 0.0))
            norm_vector["label"] = int(row.get("label", 0))
            if "subclass" in row:
                norm_vector["subclass"] = row["subclass"]

            output_rows.append(norm_vector)

        return pd.DataFrame(output_rows)
