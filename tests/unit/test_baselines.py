"""Unit tests for baseline detectors and threshold calibration."""

import numpy as np
import pytest

from canexplain.data.synthetic import generate_synthetic_can_stream
from canexplain.models.baselines.iforest import IsolationForestDetector
from canexplain.models.baselines.mlp import MLPDetector
from canexplain.models.baselines.statistical import StatisticalDeviationDetector
from canexplain.models.threshold import ThresholdCalibrator
from canexplain.preprocessing.features import extract_10_features
from canexplain.preprocessing.normalization import CANFeatureScaler
from canexplain.preprocessing.windows import construct_sliding_windows


@pytest.fixture
def sample_windows_data():
    """Provides benign and attack windows for testing."""
    df_benign = generate_synthetic_can_stream(num_frames=300, attack_type="none", seed=42)
    df_attack = generate_synthetic_can_stream(
        num_frames=300, attack_type="dos", attack_start_idx=100, attack_duration=100, seed=42
    )

    feats_benign = extract_10_features(df_benign)
    feats_attack = extract_10_features(df_attack)

    scaler = CANFeatureScaler().fit(feats_benign)
    normed_benign = scaler.transform(feats_benign)
    normed_attack = scaler.transform(feats_attack)

    win_benign, y_benign, _ = construct_sliding_windows(
        normed_benign,
        df_benign["label"].to_numpy(),
        df_benign["timestamp"].to_numpy(),
        window_size=32,
        stride=16,
    )
    win_attack, y_attack, _ = construct_sliding_windows(
        normed_attack,
        df_attack["label"].to_numpy(),
        df_attack["timestamp"].to_numpy(),
        window_size=32,
        stride=16,
    )

    return win_benign, y_benign, win_attack, y_attack


def test_threshold_calibrator():
    normal_scores = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22])
    calibrator = ThresholdCalibrator(method="percentile", percentile=90.0)
    calibrator.fit(normal_scores)

    assert calibrator.threshold > 0.2
    test_scores = np.array([0.1, 0.5])
    preds = calibrator.predict(test_scores)
    assert preds[0] == 0
    assert preds[1] == 1


def test_statistical_deviation_detector(sample_windows_data):
    win_benign, _, win_attack, y_attack = sample_windows_data

    # Canonical 10-feature layout: delta_t is at index 0 (auto-detected)
    detector = StatisticalDeviationDetector()
    detector.fit(win_benign)

    scores_benign = detector.score_windows(win_benign)
    scores_attack = detector.score_windows(win_attack)

    assert len(scores_benign) == len(win_benign)
    assert len(scores_attack) == len(win_attack)

    # Attack windows (which contain DoS floods with compressed delta t) should have high deviation
    calibrator = ThresholdCalibrator(method="percentile", percentile=95.0).fit(scores_benign)
    preds = calibrator.predict(scores_attack)
    # At least some attack windows must be detected
    assert np.any(preds[y_attack == 1] == 1)


def test_isolation_forest_detector(sample_windows_data):
    win_benign, _, win_attack, y_attack = sample_windows_data

    iforest = IsolationForestDetector(n_estimators=30, random_state=42)
    iforest.fit(win_benign)

    scores_benign = iforest.score_windows(win_benign)
    scores_attack = iforest.score_windows(win_attack)

    assert len(scores_attack) == len(win_attack)
    calibrator = ThresholdCalibrator(method="percentile", percentile=90.0).fit(scores_benign)
    preds = calibrator.predict(scores_attack)
    assert np.any(preds == 1)


def test_mlp_detector(sample_windows_data):
    win_benign, y_benign, win_attack, y_attack = sample_windows_data

    # Combine for supervised training
    win_train = np.concatenate([win_benign[:8], win_attack[:8]], axis=0)
    y_train = np.concatenate([y_benign[:8], y_attack[:8]], axis=0)

    mlp = MLPDetector(hidden_layer_sizes=(16,), max_iter=50, random_state=42)
    mlp.fit(win_train, y_train)

    probs = mlp.score_windows(win_attack)
    assert len(probs) == len(win_attack)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
