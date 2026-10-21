"""Unit tests for feature extraction, zero-leakage scaling, quantum encoding, and sliding windows."""

import pytest
import numpy as np
import pandas as pd

from canexplain.data.synthetic import generate_synthetic_can_stream
from canexplain.preprocessing.features import extract_10_features, compute_byte_entropy
from canexplain.preprocessing.normalization import CANFeatureScaler
from canexplain.preprocessing.encoding import trigonometric_feature_map, encode_feature_matrix
from canexplain.preprocessing.windows import construct_sliding_windows


def test_extract_10_features():
    df = generate_synthetic_can_stream(num_frames=200, attack_type="none")
    feats = extract_10_features(df)

    assert "timestamp_offset" in feats.columns
    assert "delta_t" in feats.columns
    assert "arbitration_id" in feats.columns
    for i in range(8):
        assert f"byte_{i}" in feats.columns
    assert "label" in feats.columns

    # Check non-negative delta_t
    assert (feats["delta_t"] >= 0).all()


def test_zero_leakage_scaler():
    df_train = generate_synthetic_can_stream(num_frames=200, attack_type="none")
    df_test = generate_synthetic_can_stream(num_frames=200, attack_type="dos", attack_start_idx=50, attack_duration=50)

    feats_train = extract_10_features(df_train)
    feats_test = extract_10_features(df_test)

    scaler = CANFeatureScaler()
    train_scaled = scaler.fit_transform(feats_train)

    assert train_scaled.shape == (200, 11)  # 11 feature columns
    assert np.all(train_scaled >= 0.0)
    assert np.all(train_scaled <= 1.0)

    # Transform test without re-fitting
    test_scaled = scaler.transform(feats_test)
    assert test_scaled.shape == (200, 11)
    assert np.all(test_scaled >= 0.0)
    assert np.all(test_scaled <= 1.0)


def test_trigonometric_feature_map_properties():
    # Test values in [0, 1]
    vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    encoded = trigonometric_feature_map(vals)

    assert encoded.shape == (5, 2)

    # Verify unit norm: cos^2 + sin^2 = 1.0 for all vectors
    norms = np.sum(encoded ** 2, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    # Boundary values:
    # x=0 -> [cos(0), sin(0)] = [1, 0]
    np.testing.assert_allclose(encoded[0], [1.0, 0.0], atol=1e-12)
    # x=1 -> [cos(pi/2), sin(pi/2)] = [0, 1]
    np.testing.assert_allclose(encoded[-1], [0.0, 1.0], atol=1e-12)


def test_construct_sliding_windows():
    df = generate_synthetic_can_stream(num_frames=200, attack_type="dos", attack_start_idx=100, attack_duration=40)
    feats = extract_10_features(df)
    scaler = CANFeatureScaler().fit(feats)
    normed = scaler.transform(feats)

    timestamps = df["timestamp"].to_numpy()
    labels = df["label"].to_numpy()

    windows, w_labels, prov = construct_sliding_windows(
        feature_matrix=normed,
        labels=labels,
        timestamps=timestamps,
        window_size=64,
        stride=32
    )

    # Total frames = 200, window=64, stride=32:
    # Windows start at 0, 32, 64, 96, 128 -> 5 windows
    assert windows.shape == (5, 64, 11)
    assert len(w_labels) == 5
    assert len(prov) == 5

    # Check that windows overlapping with attack [100, 140) have label 1
    # Window 0: [0, 64) -> 0
    # Window 1: [32, 96) -> 0
    # Window 2: [64, 128) -> overlaps with [100, 128) -> 1
    # Window 3: [96, 160) -> overlaps with [100, 140) -> 1
    # Window 4: [128, 192) -> overlaps with [128, 140) -> 1
    assert w_labels[0] == 0
    assert w_labels[1] == 0
    assert w_labels[2] == 1
    assert w_labels[3] == 1
    assert w_labels[4] == 1

    # Verify provenance timestamps and indices
    assert prov[0].start_frame_idx == 0
    assert prov[0].end_frame_idx == 63
    assert prov[0].frame_count == 64
