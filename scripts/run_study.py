"""Systematic empirical study: bond dimension sweeps, window length ablations, and real CAN data evaluation."""

import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from canexplain.data.loaders import load_can_csv
from canexplain.data.synthetic import generate_synthetic_can_stream
from canexplain.preprocessing.features import extract_10_features
from canexplain.preprocessing.normalization import CANFeatureScaler
from canexplain.preprocessing.encoding import trigonometric_feature_map
from canexplain.preprocessing.windows import construct_sliding_windows
from canexplain.models.threshold import ThresholdCalibrator
from canexplain.models.mps.born_machine import MPSBornMachine
from canexplain.models.mps.dmrg import DMRGOptimizer
from canexplain.models.baselines.statistical import StatisticalDeviationDetector
from canexplain.models.baselines.iforest import IsolationForestDetector
from canexplain.models.baselines.mlp import MLPDetector
from canexplain.evaluation.metrics import compute_ids_metrics
from canexplain.evaluation.benchmark import ResourceProfiler


def run_bond_dimension_sweep(
    train_norm: np.ndarray,
    val_norm: np.ndarray,
    test_norm: np.ndarray,
    y_test: np.ndarray,
    bond_dimensions: list[int] = [8, 16, 32, 64],
    window_size: int = 32,
    stride: int = 16,
) -> dict:
    """Evaluates how bond dimension D governs F1 score, contraction latency, and memory."""
    print("\n" + "=" * 75)
    print("Study 1: Bond Dimension Scaling (D in {8, 16, 32, 64})")
    print("=" * 75)

    # Windowing
    win_train, _, _ = construct_sliding_windows(
        train_norm, np.zeros(len(train_norm)), np.arange(len(train_norm)), window_size, stride
    )
    win_val, _, _ = construct_sliding_windows(
        val_norm, np.zeros(len(val_norm)), np.arange(len(val_norm)), window_size, stride
    )
    win_test, y_win_test, _ = construct_sliding_windows(
        test_norm, y_test, np.arange(len(test_norm)), window_size, stride
    )

    # Encode window summaries
    phi_train = trigonometric_feature_map(np.mean(win_train, axis=1))
    phi_val = trigonometric_feature_map(np.mean(win_val, axis=1))
    phi_test = trigonometric_feature_map(np.mean(win_test, axis=1))

    num_sites = phi_train.shape[1]
    sweep_results = {}

    for d in bond_dimensions:
        bm = MPSBornMachine(num_sites=num_sites, physical_dim=2, bond_dimension=d)
        optimizer = DMRGOptimizer(max_bond_dimension=d, learning_rate=0.08, num_sweeps=3)

        # Profile Training
        _, train_metrics = ResourceProfiler.profile_training(lambda: optimizer.fit(bm, phi_train))

        # Calibrate threshold
        val_scores = bm.score_samples(phi_val)
        calibrator = ThresholdCalibrator(percentile=95.0).fit(val_scores)

        # Profile Inference
        test_scores = bm.score_samples(phi_test)
        preds = calibrator.predict(test_scores)
        ids_metrics = compute_ids_metrics(y_win_test, preds, test_scores)

        inf_profile = ResourceProfiler.profile_inference(bm.score_samples, phi_test)

        entry = {
            "bond_dimension": d,
            "weighted_f1": float(ids_metrics["f1_weighted"]),
            "recall": float(ids_metrics["recall_weighted"]),
            "precision": float(ids_metrics["precision_weighted"]),
            "fpr": float(ids_metrics["fpr"]),
            "latency_ms_per_window": float(inf_profile["latency_ms_per_window"]),
            "training_time_sec": float(train_metrics["training_time_sec"]),
            "peak_memory_mb": float(train_metrics["peak_memory_mb"]),
        }
        sweep_results[f"D_{d}"] = entry
        print(
            f"D={d:2d} | F1: {entry['weighted_f1']:.4f} | Recall: {entry['recall']:.4f} | "
            f"FPR: {entry['fpr']:.4f} | Latency: {entry['latency_ms_per_window']:.3f} ms | "
            f"Train Time: {entry['training_time_sec']:.2f} s"
        )

    return sweep_results


def run_window_size_ablation(
    train_norm: np.ndarray,
    val_norm: np.ndarray,
    test_norm: np.ndarray,
    y_test: np.ndarray,
    window_sizes: list[int] = [16, 32, 64],
    bond_dim: int = 16,
) -> dict:
    """Evaluates the trade-off between sequence context window and detection quality."""
    print("\n" + "=" * 75)
    print("Study 2: Temporal Window Context Ablation (W in {16, 32, 64})")
    print("=" * 75)

    ablation_results = {}

    for w in window_sizes:
        stride = max(w // 2, 8)
        win_train, _, _ = construct_sliding_windows(
            train_norm, np.zeros(len(train_norm)), np.arange(len(train_norm)), w, stride
        )
        win_val, _, _ = construct_sliding_windows(
            val_norm, np.zeros(len(val_norm)), np.arange(len(val_norm)), w, stride
        )
        win_test, y_win_test, _ = construct_sliding_windows(
            test_norm, y_test, np.arange(len(test_norm)), w, stride
        )

        phi_train = trigonometric_feature_map(np.mean(win_train, axis=1))
        phi_val = trigonometric_feature_map(np.mean(win_val, axis=1))
        phi_test = trigonometric_feature_map(np.mean(win_test, axis=1))

        bm = MPSBornMachine(num_sites=phi_train.shape[1], physical_dim=2, bond_dimension=bond_dim)
        optimizer = DMRGOptimizer(max_bond_dimension=bond_dim, learning_rate=0.08, num_sweeps=3)
        optimizer.fit(bm, phi_train)

        val_scores = bm.score_samples(phi_val)
        calibrator = ThresholdCalibrator(percentile=95.0).fit(val_scores)

        test_scores = bm.score_samples(phi_test)
        preds = calibrator.predict(test_scores)
        ids_metrics = compute_ids_metrics(y_win_test, preds, test_scores)

        entry = {
            "window_size": w,
            "stride": stride,
            "weighted_f1": float(ids_metrics["f1_weighted"]),
            "recall": float(ids_metrics["recall_weighted"]),
            "precision": float(ids_metrics["precision_weighted"]),
            "fpr": float(ids_metrics["fpr"]),
            "num_windows": len(win_test),
        }
        ablation_results[f"W_{w}"] = entry
        print(
            f"W={w:2d} (Stride {stride:2d}) | F1: {entry['weighted_f1']:.4f} | "
            f"Recall: {entry['recall']:.4f} | FPR: {entry['fpr']:.4f} | Windows: {entry['num_windows']}"
        )

    return ablation_results


def run_real_dataset_evaluation() -> dict:
    """Evaluates the models on downloaded real automotive CAN traces from can-train-and-test-v1.5."""
    print("\n" + "=" * 75)
    print("Study 3: Evaluation on Real On-Road CAN Traces (can-train-and-test-v1.5)")
    print("=" * 75)

    real_train_path = "data/real/set_01/train_01_attack_free/accessory-1.csv"
    real_test_path = "data/real/set_01/test_01_known_vehicle_known_attack/DoS-3.csv"

    if not (os.path.exists(real_train_path) and os.path.exists(real_test_path)):
        print(f"[SKIP] Real data files not found at {real_train_path}. Run download_data.py first.")
        return {}

    print(f"Loading real benign training trace: {real_train_path}...")
    df_real_train = load_can_csv(real_train_path, sample_limit=5000)
    print(f"Loaded {len(df_real_train)} real frames.")

    print(f"Loading real DoS attack trace: {real_test_path}...")
    df_real_test = load_can_csv(real_test_path, sample_limit=5000)
    print(f"Loaded {len(df_real_test)} real frames (Attack frames: {(df_real_test['label'] == 1).sum()}).")

    # Split train into train and val
    val_split = int(0.7 * len(df_real_train))
    df_tr = df_real_train.iloc[:val_split]
    df_val = df_real_train.iloc[val_split:]

    feats_tr = extract_10_features(df_tr)
    feats_val = extract_10_features(df_val)
    feats_te = extract_10_features(df_real_test)

    feature_cols = [c for c in feats_tr.columns if c not in ("label", "subclass")]
    scaler = CANFeatureScaler().fit(feats_tr, feature_cols=feature_cols)

    norm_tr = scaler.transform(feats_tr)
    norm_val = scaler.transform(feats_val)
    norm_te = scaler.transform(feats_te)

    w_size, stride = 32, 16
    win_tr, _, _ = construct_sliding_windows(norm_tr, df_tr["label"].to_numpy(), df_tr["timestamp"].to_numpy(), w_size, stride)
    win_val, _, _ = construct_sliding_windows(norm_val, df_val["label"].to_numpy(), df_val["timestamp"].to_numpy(), w_size, stride)
    win_te, y_win_te, _ = construct_sliding_windows(norm_te, df_real_test["label"].to_numpy(), df_real_test["timestamp"].to_numpy(), w_size, stride)

    phi_tr = trigonometric_feature_map(np.mean(win_tr, axis=1))
    phi_val = trigonometric_feature_map(np.mean(win_val, axis=1))
    phi_te = trigonometric_feature_map(np.mean(win_te, axis=1))

    # Evaluate all 4 models on real data
    real_results = {}

    # 1. Statistical
    stat = StatisticalDeviationDetector().fit(win_tr)
    stat_val_scores = stat.score_windows(win_val)
    stat_calib = ThresholdCalibrator(percentile=95.0).fit(stat_val_scores)
    stat_te_scores = stat.score_windows(win_te)
    real_results["Statistical"] = compute_ids_metrics(y_win_te, stat_calib.predict(stat_te_scores), stat_te_scores)

    # 2. Isolation Forest
    iforest = IsolationForestDetector(n_estimators=50, random_state=42).fit(win_tr)
    iforest_val = iforest.score_windows(win_val)
    iforest_calib = ThresholdCalibrator(percentile=95.0).fit(iforest_val)
    iforest_te_scores = iforest.score_windows(win_te)
    real_results["IsolationForest"] = compute_ids_metrics(y_win_te, iforest_calib.predict(iforest_te_scores), iforest_te_scores)

    # 3. MLP Reference
    mlp_x = np.concatenate([win_tr, win_te[:10]], axis=0)
    mlp_y = np.concatenate([np.zeros(len(win_tr)), y_win_te[:10]], axis=0)
    mlp = MLPDetector(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42).fit(mlp_x, mlp_y)
    mlp_scores = mlp.score_windows(win_te)
    real_results["MLP_Reference"] = compute_ids_metrics(y_win_te, (mlp_scores > 0.5).astype(int), mlp_scores)

    # 4. MPS Born Machine
    bm = MPSBornMachine(num_sites=phi_tr.shape[1], physical_dim=2, bond_dimension=24)
    optimizer = DMRGOptimizer(max_bond_dimension=24, learning_rate=0.08, num_sweeps=4)
    optimizer.fit(bm, phi_tr)

    bm_val_scores = bm.score_samples(phi_val)
    bm_calib = ThresholdCalibrator(percentile=95.0).fit(bm_val_scores)
    bm_te_scores = bm.score_samples(phi_te)
    real_results["MPS_BornMachine"] = compute_ids_metrics(y_win_te, bm_calib.predict(bm_te_scores), bm_te_scores)

    print("\nReal CAN Trace Detection Results:")
    print(f"{'Model':<20} | {'F1 (Weighted)':<14} | {'Recall':<8} | {'Precision':<10} | {'FPR':<8} | {'G-mean':<8}")
    print("-" * 75)
    for model_name, m in real_results.items():
        print(f"{model_name:<20} | {m['f1_weighted']:<14.4f} | {m['recall_weighted']:<8.4f} | {m['precision_weighted']:<10.4f} | {m['fpr']:<8.4f} | {m['g_mean']:<8.4f}")

    return real_results


def main():
    print("=" * 80)
    print("CANExplain-QI: Research Empirical Study (Bond Scaling, Window Ablation, Real Traces)")
    print("=" * 80)

    # Prepare benchmark data
    df_train = generate_synthetic_can_stream(num_frames=1000, attack_type="none", seed=42)
    df_val = generate_synthetic_can_stream(num_frames=400, attack_type="none", seed=43)
    df_test = generate_synthetic_can_stream(num_frames=800, attack_type="dos", attack_start_idx=300, attack_duration=150, seed=44)

    feats_tr = extract_10_features(df_train)
    feats_val = extract_10_features(df_val)
    feats_te = extract_10_features(df_test)

    feature_cols = [c for c in feats_tr.columns if c not in ("label", "subclass")]
    scaler = CANFeatureScaler().fit(feats_tr, feature_cols=feature_cols)

    norm_tr = scaler.transform(feats_tr)
    norm_val = scaler.transform(feats_val)
    norm_te = scaler.transform(feats_te)
    y_test = df_test["label"].to_numpy()

    # Run sweeps
    study_1_bond = run_bond_dimension_sweep(norm_tr, norm_val, norm_te, y_test)
    study_2_window = run_window_size_ablation(norm_tr, norm_val, norm_te, y_test)
    study_3_real = run_real_dataset_evaluation()

    full_study = {
        "study_1_bond_dimension_sweep": study_1_bond,
        "study_2_window_size_ablation": study_2_window,
        "study_3_real_dataset_evaluation": study_3_real,
    }

    with open("study_results.json", "w", encoding="utf-8") as f:
        json.dump(full_study, f, indent=2)
    print("\n[OK] Study results successfully saved to: study_results.json")


if __name__ == "__main__":
    main()
