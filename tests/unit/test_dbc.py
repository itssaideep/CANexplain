"""Unit tests for DBC signal decoding, OpenDBC profiles, and semantic tokenization."""

import numpy as np
import pandas as pd

from canexplain.data.dbc.opendbc_adapter import (
    create_chevrolet_impala_dbc,
    create_hyundai_sonata_dbc,
)
from canexplain.data.dbc.semantic_tokenizer import SemanticCANTokenizer
from canexplain.data.dbc.signal_schema import CANSignalDefinition
from canexplain.models.mps.born_machine import MPSBornMachine
from canexplain.preprocessing.encoding import trigonometric_feature_map


def test_can_signal_decoding_unsigned():
    """Tests little-endian unsigned integer extraction and scaling."""
    sig = CANSignalDefinition(
        name="engine_rpm",
        start_bit=16,
        bit_length=16,
        scale=0.25,
        offset=0.0,
        min_physical=0.0,
        max_physical=8000.0,
    )
    # Raw value 4000 at bits 16..31: 4000 in hex is 0x0FA0 -> in little-endian [0xA0, 0x0F]
    payload = bytes([0x00, 0x00, 0xA0, 0x0F, 0x00, 0x00, 0x00, 0x00])
    val = sig.decode(payload)
    # 4000 * 0.25 = 1000.0 RPM
    assert np.isclose(val, 1000.0)


def test_can_signal_decoding_signed_negative():
    """Tests two's complement signed signal extraction (e.g. steering angle)."""
    sig = CANSignalDefinition(
        name="steering_angle",
        start_bit=0,
        bit_length=16,
        is_signed=True,
        scale=0.1,
        offset=0.0,
        min_physical=-500.0,
        max_physical=500.0,
    )
    # Negative angle: -250.0 degrees -> raw = -2500
    # 16-bit two's complement: 65536 - 2500 = 63036 = 0xF63C -> little-endian: [0x3C, 0xF6]
    payload = bytes([0x3C, 0xF6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    val = sig.decode(payload)
    assert np.isclose(val, -250.0)


def test_hyundai_sonata_dbc_message_decoding():
    """Tests decoding of multiple signals from Hyundai engine and steering messages."""
    dbc = create_hyundai_sonata_dbc()

    # Message 0x316: engine RPM (bits 16..31) and throttle (bits 32..39)
    # RPM = 8000 raw -> 2000 RPM (0x1F40 -> bytes [0x40, 0x1F])
    # Throttle = 128 raw -> 128 * 0.392 = 50.176% (0x80)
    p_engine = bytes([0x00, 0x00, 0x40, 0x1F, 0x80, 0x00, 0x00, 0x00])
    decoded_engine = dbc.decode_frame(0x316, p_engine)
    assert "engine_rpm" in decoded_engine
    assert np.isclose(decoded_engine["engine_rpm"], 2000.0)
    assert "throttle_pct" in decoded_engine
    assert np.isclose(decoded_engine["throttle_pct"], 50.176)


def test_semantic_can_tokenizer_sample_and_hold():
    """Validates that SemanticCANTokenizer maintains rolling state and normalizes to [0, 1]."""
    dbc = create_hyundai_sonata_dbc()
    tokenizer = SemanticCANTokenizer(dbc)

    frames = [
        # Frame 0: Steering message 0x164 (angle = 0 deg)
        {
            "timestamp": 1.0,
            "arbitration_id": 0x164,
            "payload_bytes": bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            "label": 0,
        },
        # Frame 1: Engine message 0x316 (RPM = 4000 -> 1000 RPM)
        {
            "timestamp": 1.02,
            "arbitration_id": 0x316,
            "payload_bytes": bytes([0x00, 0x00, 0xA0, 0x0F, 0x00, 0x00, 0x00, 0x00]),
            "label": 0,
        },
        # Frame 2: Speed message 0x220 (speed = 3200 raw * 0.03125 = 100 km/h)
        # 3200 in hex = 0x0C80 -> [0x80, 0x0C]
        {
            "timestamp": 1.04,
            "arbitration_id": 0x220,
            "payload_bytes": bytes([0x80, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            "label": 0,
        },
    ]
    df_raw = pd.DataFrame(frames)
    df_tokenized = tokenizer.tokenize_stream(df_raw)

    assert len(df_tokenized) == 3
    # Check that universal physical signals are present
    for s in ["vehicle_speed", "steering_angle", "engine_rpm", "throttle_pct"]:
        assert s in df_tokenized.columns
        # All normalized values must be strictly in [0.0, 1.0]
        assert np.all(df_tokenized[s] >= 0.0)
        assert np.all(df_tokenized[s] <= 1.0)

    # Frame 2 should retain engine_rpm from Frame 1 (sample and hold)
    # 1000 RPM / 8000 max = 0.125
    assert np.isclose(df_tokenized["engine_rpm"].iloc[2], 0.125)
    # Frame 2 vehicle speed: 100 km/h / 250 max = 0.4
    assert np.isclose(df_tokenized["vehicle_speed"].iloc[2], 0.4)


def test_cross_vehicle_semantic_compatibility():
    """Validates that Hyundai and Chevy streams yield identical semantic schemas for MPS."""
    dbc_hyundai = create_hyundai_sonata_dbc()
    dbc_chevy = create_chevrolet_impala_dbc()

    tok_hyundai = SemanticCANTokenizer(dbc_hyundai)
    tok_chevy = SemanticCANTokenizer(dbc_chevy)

    # Synthetic Hyundai stream
    hyundai_frames = pd.DataFrame(
        [
            {
                "timestamp": 0.1,
                "arbitration_id": 0x316,
                "payload_bytes": bytes([0, 0, 0xA0, 0x0F, 0x50, 0, 0, 0]),
            },
            {
                "timestamp": 0.2,
                "arbitration_id": 0x220,
                "payload_bytes": bytes([0x80, 0x0C, 0, 0, 0, 0, 0, 0]),
            },
        ]
    )

    # Synthetic Chevy stream (different IDs: 0x0C9 for engine, 0x1F1 for speed)
    chevy_frames = pd.DataFrame(
        [
            {
                "timestamp": 0.1,
                "arbitration_id": 0x0C9,
                "payload_bytes": bytes([0xA0, 0x0F, 0, 0x50, 0, 0, 0, 0]),
            },
            {
                "timestamp": 0.2,
                "arbitration_id": 0x1F1,
                "payload_bytes": bytes([0x00, 0x19, 0, 0, 0, 0, 0, 0]),
            },
        ]
    )

    df_h_sem = tok_hyundai.tokenize_stream(hyundai_frames)
    df_c_sem = tok_chevy.tokenize_stream(chevy_frames)

    # Signal feature columns must match exactly
    sig_cols = ["vehicle_speed", "steering_angle", "engine_rpm", "throttle_pct"]
    feats_h = df_h_sem[sig_cols].to_numpy()
    feats_c = df_c_sem[sig_cols].to_numpy()

    assert feats_h.shape == feats_c.shape

    # An MPS Born Machine trained on 4 semantic sites can score BOTH vehicles directly
    bm = MPSBornMachine(num_sites=4, physical_dim=2, bond_dimension=4, seed=42)
    phi_h = np.array([trigonometric_feature_map(row) for row in feats_h])
    phi_c = np.array([trigonometric_feature_map(row) for row in feats_c])

    scores_h = bm.score_samples(phi_h)
    scores_c = bm.score_samples(phi_c)

    assert len(scores_h) == 2
    assert len(scores_c) == 2
    assert np.all(np.isfinite(scores_h))
    assert np.all(np.isfinite(scores_c))
