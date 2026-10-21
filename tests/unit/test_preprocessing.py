"""Unit tests for feature extraction, zero-leakage scaling, quantum encoding, and sliding windows."""

import numpy as np

from canexplain.data.synthetic import generate_synthetic_can_stream
from canexplain.preprocessing.encoding import (
    MeanPooledFeatureMap,
    SequencePreservingEncoder,
    encode_feature_matrix,
    trigonometric_feature_map,
)
from canexplain.preprocessing.features import (
    CANONICAL_10_FEATURES,
    LEGACY_11_FEATURES,
    extract_10_features,
    extract_canonical_features,
)
from canexplain.preprocessing.normalization import CANFeatureScaler
from canexplain.preprocessing.windows import construct_sliding_windows


def test_extract_10_features():
    df = generate_synthetic_can_stream(num_frames=200, attack_type="none")

    # Canonical 10-feature stationary mode
    feats_canonical = extract_canonical_features(df)
    assert "timestamp_offset" not in feats_canonical.columns
    assert "delta_t" in feats_canonical.columns
    assert "arbitration_id" in feats_canonical.columns
    for i in range(8):
        assert f"byte_{i}" in feats_canonical.columns
    assert "label" in feats_canonical.columns
    assert "subclass" in feats_canonical.columns
    # Exactly 10 numeric features
    numeric_cols = [c for c in feats_canonical.columns if c not in ("label", "subclass")]
    assert numeric_cols == CANONICAL_10_FEATURES
    assert len(numeric_cols) == 10
    assert (feats_canonical["delta_t"] >= 0).all()

    # Legacy 11-feature mode
    feats_legacy = extract_10_features(df, stationary_only=False)
    assert "timestamp_offset" in feats_legacy.columns
    legacy_cols = [c for c in feats_legacy.columns if c not in ("label", "subclass")]
    assert legacy_cols == LEGACY_11_FEATURES
    assert len(legacy_cols) == 11


def test_zero_leakage_scaler():
    df_train = generate_synthetic_can_stream(num_frames=200, attack_type="none")
    df_test = generate_synthetic_can_stream(
        num_frames=200, attack_type="dos", attack_start_idx=50, attack_duration=50
    )

    feats_train = extract_canonical_features(df_train)
    feats_test = extract_canonical_features(df_test)

    scaler = CANFeatureScaler()
    train_scaled = scaler.fit_transform(feats_train)

    assert train_scaled.shape == (200, 10)  # 10 canonical feature columns
    assert np.all(train_scaled >= 0.0)
    assert np.all(train_scaled <= 1.0)

    # Transform test without re-fitting
    test_scaled = scaler.transform(feats_test)
    assert test_scaled.shape == (200, 10)
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
    df = generate_synthetic_can_stream(
        num_frames=200, attack_type="dos", attack_start_idx=100, attack_duration=40
    )
    feats = extract_canonical_features(df)
    scaler = CANFeatureScaler().fit(feats)
    normed = scaler.transform(feats)

    timestamps = df["timestamp"].to_numpy()
    labels = df["label"].to_numpy()

    windows, w_labels, prov = construct_sliding_windows(
        feature_matrix=normed,
        labels=labels,
        timestamps=timestamps,
        window_size=64,
        stride=32,
        dataset_name="synthetic_test",
        trace_name="trace_dos_1",
    )

    # Total frames = 200, window=64, stride=32:
    # Windows start at 0, 32, 64, 96, 128 -> 5 windows
    assert windows.shape == (5, 64, 10)
    assert len(w_labels) == 5
    assert len(prov) == 5

    # Check window 2 overlaps with attack [100, 128) -> attack frames present
    assert w_labels[0] == 0
    assert w_labels[1] == 0
    assert w_labels[2] == 1
    assert w_labels[3] == 1
    assert w_labels[4] == 1

    # Verify provenance timestamps and indices
    assert prov[0].start_frame_idx == 0
    assert prov[0].end_frame_idx == 63
    assert prov[0].frame_count == 64
    assert prov[2].attack_frame_count > 0
    assert prov[2].attack_frame_ratio > 0.0
    assert prov[0].dataset_name == "synthetic_test"
    assert prov[0].trace_name == "trace_dos_1"

    # Test majority label threshold (threshold = 0.5)
    _, maj_labels, _ = construct_sliding_windows(
        feature_matrix=normed,
        labels=labels,
        timestamps=timestamps,
        window_size=64,
        stride=32,
        label_threshold=0.5,
    )
    # Attack duration is 40 frames out of 64 frames (40/64 = 0.625 > 0.5 in window 3: [96, 160))
    # Window 2 has only 28 attack frames (28/64 = 0.4375 <= 0.5)
    assert maj_labels[2] == 0  # Under majority threshold
    assert maj_labels[3] == 1  # Over majority threshold


def test_encoders_pooled_and_sequence_preserving():
    windows = np.random.uniform(0.0, 1.0, size=(10, 16, 10))

    # 1. Pooled window baseline
    pooled_encoder = MeanPooledFeatureMap()
    phi_pooled = pooled_encoder.encode(windows)
    assert phi_pooled.shape == (10, 10, 2)
    # Check unit norm per site
    norms = np.sum(phi_pooled ** 2, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    # 2. Sequence-preserving encoder (flatten mode)
    seq_encoder_flat = SequencePreservingEncoder(mode="flatten")
    phi_seq_flat = seq_encoder_flat.encode(windows)
    assert phi_seq_flat.shape == (10, 160, 2)
    norms_flat = np.sum(phi_seq_flat ** 2, axis=-1)
    np.testing.assert_allclose(norms_flat, 1.0, atol=1e-12)

    # 3. Sequence-preserving encoder (per-frame mode)
    seq_encoder_frame = SequencePreservingEncoder(mode="per_frame")
    phi_seq_frame = seq_encoder_frame.encode(windows)
    assert phi_seq_frame.shape == (10, 16, 10, 2)
