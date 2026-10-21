"""Forensic provenance tracing linking anomaly alerts back to raw CAN frames."""

import pandas as pd

from canexplain.preprocessing.windows import WindowProvenance


def trace_window_to_raw_traffic(
    provenance: WindowProvenance,
    raw_df: pd.DataFrame,
) -> dict[str, object]:
    """Extracts the exact raw CAN frames and summary statistics for an alert window.

    Args:
        provenance: WindowProvenance object for the flagged window.
        raw_df: Original raw DataFrame from which the window was constructed.

    Returns:
        Dictionary containing granular forensic provenance information.
    """
    sub_df = raw_df.iloc[provenance.start_frame_idx : provenance.end_frame_idx + 1]

    id_counts = sub_df["arbitration_id"].value_counts().to_dict()
    id_hex_counts = {f"0x{int(k):03X}": int(v) for k, v in id_counts.items()}

    primary_id = sub_df["arbitration_id"].mode().iloc[0] if len(sub_df) > 0 else 0

    return {
        "window_index": provenance.window_idx,
        "start_frame_index": provenance.start_frame_idx,
        "end_frame_index": provenance.end_frame_idx,
        "frame_count": provenance.frame_count,
        "start_timestamp": provenance.start_timestamp,
        "end_timestamp": provenance.end_timestamp,
        "duration_seconds": provenance.duration_seconds,
        "primary_arbitration_id": f"0x{int(primary_id):03X}",
        "arbitration_id_counts": id_hex_counts,
        "unique_id_count": len(id_hex_counts),
        "mean_inter_arrival_time": float(sub_df["timestamp"].diff().mean())
        if len(sub_df) > 1
        else 0.0,
    }
