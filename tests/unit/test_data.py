"""Unit tests for CAN data schema, loaders, and synthetic stream generator."""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np

from canexplain.data.schema import CANFrame, parse_can_id, normalize_payload_bytes
from canexplain.data.loaders import load_can_csv, STANDARD_BASE_TIMESTAMP
from canexplain.data.synthetic import generate_synthetic_can_stream


def test_parse_can_id():
    assert parse_can_id("0x120") == 0x120
    assert parse_can_id("120") == 0x120
    assert parse_can_id(0x288) == 0x288
    assert parse_can_id("000") == 0x000

    with pytest.raises(ValueError):
        parse_can_id(0x800)  # Exceeds 11-bit


def test_normalize_payload_bytes():
    # 8-byte hex string
    p = normalize_payload_bytes("008000807FFA5589")
    assert len(p) == 8
    assert p == bytes.fromhex("008000807FFA5589")

    # Short payload padded with zeros
    p_short = normalize_payload_bytes("010203")
    assert len(p_short) == 8
    assert p_short[:3] == bytes([1, 2, 3])
    assert p_short[3:] == bytes([0, 0, 0, 0, 0])

    # Spaces or hyphens stripped
    p_spaced = normalize_payload_bytes("00 80 00 80 7F FA 55 89")
    assert len(p_spaced) == 8


def test_can_frame_dataclass():
    frame = CANFrame(
        timestamp=100.0,
        arbitration_id=0x120,
        dlc=8,
        payload=bytes([0] * 8),
        label=0
    )
    assert frame.id_hex == "120"
    assert frame.payload_hex == "0000000000000000"


def test_synthetic_stream_generation():
    # Benign stream
    df_benign = generate_synthetic_can_stream(num_frames=500, attack_type="none")
    assert len(df_benign) == 500
    assert (df_benign["label"] == 0).all()

    # DoS attack stream
    df_dos = generate_synthetic_can_stream(
        num_frames=500,
        attack_type="dos",
        attack_start_idx=100,
        attack_duration=50
    )
    assert len(df_dos) == 500
    assert (df_dos["label"] == 1).sum() == 50
    assert (df_dos.loc[df_dos["label"] == 1, "arbitration_id"] == 0x000).all()


def test_load_can_csv():
    df_syn = generate_synthetic_can_stream(num_frames=100, attack_type="fuzzing", attack_start_idx=50, attack_duration=20)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df_syn[["timestamp", "arbitration_id", "dlc", "data_hex", "label"]].to_csv(f, index=False)
        temp_path = f.name

    try:
        loaded = load_can_csv(temp_path, standardize_timestamps=True)
        assert len(loaded) == 100
        assert "timestamp" in loaded.columns
        assert "payload_bytes" in loaded.columns
        assert loaded["timestamp"].iloc[0] == STANDARD_BASE_TIMESTAMP
        assert (loaded["label"] == 1).sum() == 20
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
