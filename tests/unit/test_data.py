"""Unit tests for CAN data schema, loaders, and synthetic stream generator."""

import os
import tempfile

import pytest

from canexplain.data.loaders import STANDARD_BASE_TIMESTAMP, load_can_csv
from canexplain.data.schema import CANFrame, normalize_payload_bytes, parse_can_id
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
        label=0,
        frame_index=42,
        dataset_name="can-train-and-test-v1.5",
        vehicle_id="vehicle_A",
        trace_name="accessory-1",
        split="train",
    )
    assert frame.id_hex == "120"
    assert frame.payload_hex == "0000000000000000"
    assert frame.frame_index == 42
    assert frame.dataset_name == "can-train-and-test-v1.5"

    # Test extended 29-bit CAN ID
    frame_ext = CANFrame(
        timestamp=101.5,
        arbitration_id=0x18DAF110,
        is_extended_id=True,
        dlc=8,
        payload=bytes([1] * 8),
        label=1,
        subclass="DoS",
        raw_payload_len=8,
    )
    assert frame_ext.id_hex == "18DAF110"
    assert frame_ext.is_extended_id is True
    assert frame_ext.subclass == "DoS"

    # Test out of bounds ID rejection
    with pytest.raises(ValueError):
        CANFrame(timestamp=0.0, arbitration_id=0x800, is_extended_id=False)  # 0x800 > 0x7FF

    # Test out of bounds DLC rejection
    with pytest.raises(ValueError):
        CANFrame(timestamp=0.0, arbitration_id=0x100, dlc=9)


def test_parse_can_id_extended():
    # Standard 11-bit
    assert parse_can_id("0x7FF", allow_extended=False) == 0x7FF
    with pytest.raises(ValueError):
        parse_can_id("0x800", allow_extended=False)

    # Extended 29-bit
    assert parse_can_id("0x18DAF110", allow_extended=True) == 0x18DAF110
    assert parse_can_id(0x1FFFFFFF, allow_extended=True) == 0x1FFFFFFF
    with pytest.raises(ValueError):
        parse_can_id(0x20000000, allow_extended=True)


def test_synthetic_stream_generation():
    # Benign stream
    df_benign = generate_synthetic_can_stream(num_frames=500, attack_type="none")
    assert len(df_benign) == 500
    assert (df_benign["label"] == 0).all()

    # DoS attack stream
    df_dos = generate_synthetic_can_stream(
        num_frames=500, attack_type="dos", attack_start_idx=100, attack_duration=50
    )
    assert len(df_dos) == 500
    assert (df_dos["label"] == 1).sum() == 50
    assert (df_dos.loc[df_dos["label"] == 1, "arbitration_id"] == 0x000).all()


def test_load_can_csv():
    df_syn = generate_synthetic_can_stream(
        num_frames=100, attack_type="fuzzing", attack_start_idx=50, attack_duration=20
    )
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
