"""End-to-end experimental evaluation runner for CANExplain-QI."""

import json
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from canexplain.data.synthetic import generate_synthetic_can_stream
from canexplain.preprocessing.features import extract_10_features
from canexplain.preprocessing.normalization import CANFeatureScaler
from canexplain.preprocessing.encoding import trigonometric_feature_map
from canexplain.preprocessing.windows import construct_sliding_windows
from canexplain.models.threshold import ThresholdCalibrator
from canexplain.models.baselines.statistical import StatisticalDeviationDetector
from canexplain.models.baselines.iforest import IsolationForestDetector
from canexplain.models.baselines.mlp import MLPDetector
from canexplain.models.mps.born_machine import MPSBornMachine
from canexplain.models.mps.dmrg import DMRGOptimizer
from canexplain.explainability.rdm import RDMExtractor
from canexplain.explainability.entropy import von_neumann_entropy, compute_all_site_entropies
from canexplain.explainability.feature_prob import compute_feature_marginal_probabilities, isolate_anomalous_features
from canexplain.forensics.provenance import trace_window_to_raw_traffic
from canexplain.forensics.report import build_iso_21434_incident_report
from canexplain.evaluation.metrics import compute_ids_metrics
from canexplain.evaluation.benchmark import ResourceProfiler


def run_experiment_suite(output_json: str = "experiment_results.json") -> dict:
    """Executes comparative evaluation across all 4 detector models."""
    print("=" * 80)
    print("CANExplain-QI: Quantum-Inspired Automotive CAN-Bus Intrusion Detection")
    print("=" * 80)

    # 1. Generate Synthetic Datasets simulating can-train-and-test-v1.5 scenarios
    print("\n[Step 1/5] Synthesizing multi-scenario CAN traffic stream...")
    # Training: Normal attack-free driving traffic
    df_train = generate_synthetic_can_stream(num_frames=1200, attack_type="none", seed=101)
    # Validation: Normal traffic for threshold calibration
    df_val = generate_synthetic_can_stream(num_frames=600, attack_type="none", seed=202)
    # Test 1: Known vehicle, Known attack (DoS flood)
    df_test_dos = generate_synthetic_can_stream(
        num_frames=800, attack_type="dos", attack_start_idx=300, attack_duration=150, seed=303
    )
    # Test 2: Known vehicle, Fuzzing attack
    df_test_fuzz = generate_synthetic_can_stream(
        num_frames=800, attack_type="fuzzing", attack_start_idx=300, attack_duration=150, seed=404
    )
    # Test 3: Sensor/Payload Spoofing attack
    df_test_spoof = generate_synthetic_can_stream(
        num_frames=800, attack_type="spoofing", attack_start_idx=300, attack_duration=150, seed=505
    )

    # 2. Preprocessing & Feature Extraction
    print("[Step 2/5] Extracting 10-feature subdivided byte representations & sliding windows...")
    feats_train = extract_10_features(df_train)
    feats_val = extract_10_features(df_val)
    feats_dos = extract_10_features(df_test_dos)
    feats_fuzz = extract_10_features(df_test_fuzz)
    feats_spoof = extract_10_features(df_test_spoof)

    # Zero-leakage scaler: fit STRICTLY on train
    feature_cols = [c for c in feats_train.columns if c not in ("label", "subclass")]
    scaler = CANFeatureScaler().fit(feats_train, feature_cols=feature_cols)

    norm_train = scaler.transform(feats_train)
    norm_val = scaler.transform(feats_val)
    norm_dos = scaler.transform(feats_dos)
    norm_fuzz = scaler.transform(feats_fuzz)
    norm_spoof = scaler.transform(feats_spoof)

    window_size = 32
    stride = 16

    win_train, y_train, _ = construct_sliding_windows(
        norm_train, df_train["label"].to_numpy(), df_train["timestamp"].to_numpy(),
        window_size=window_size, stride=stride
    )
    win_val, y_val, _ = construct_sliding_windows(
        norm_val, df_val["label"].to_numpy(), df_val["timestamp"].to_numpy(),
        window_size=window_size, stride=stride
    )
    win_dos, y_dos, prov_dos = construct_sliding_windows(
        norm_dos, df_test_dos["label"].to_numpy(), df_test_dos["timestamp"].to_numpy(),
        window_size=window_size, stride=stride
    )
    win_fuzz, y_fuzz, prov_fuzz = construct_sliding_windows(
        norm_fuzz, df_test_fuzz["label"].to_numpy(), df_test_fuzz["timestamp"].to_numpy(),
        window_size=window_size, stride=stride
    )
    win_spoof, y_spoof, prov_spoof = construct_sliding_windows(
        norm_spoof, df_test_spoof["label"].to_numpy(), df_test_spoof["timestamp"].to_numpy(),
        window_size=window_size, stride=stride
    )

    # Summary feature for window-level models: mean vector per window (N_win, F)
    win_mean_train = np.mean(win_train, axis=1)
    win_mean_val = np.mean(win_val, axis=1)
    win_mean_dos = np.mean(win_dos, axis=1)
    win_mean_fuzz = np.mean(win_fuzz, axis=1)
    win_mean_spoof = np.mean(win_spoof, axis=1)

    # Quantum state mapping for MPS: encode window summary features
    phi_train = trigonometric_feature_map(win_mean_train)
    phi_val = trigonometric_feature_map(win_mean_val)
    phi_dos = trigonometric_feature_map(win_mean_dos)
    phi_fuzz = trigonometric_feature_map(win_mean_fuzz)
    phi_spoof = trigonometric_feature_map(win_mean_spoof)

    # 3. Model Training & Validation Calibration
    print("[Step 3/5] Fitting models on benign traffic & calibrating thresholds...")
    results = {}

    # Model A: Statistical Deviation Detector
    stat_detector = StatisticalDeviationDetector(dt_feature_idx=1).fit(win_train)
    stat_val_scores = stat_detector.score_windows(win_val)
    stat_calib = ThresholdCalibrator(percentile=95.0).fit(stat_val_scores)

    # Model B: Isolation Forest
    iforest_detector = IsolationForestDetector(n_estimators=50, random_state=42).fit(win_train)
    iforest_val_scores = iforest_detector.score_windows(win_val)
    iforest_calib = ThresholdCalibrator(percentile=95.0).fit(iforest_val_scores)

    # Model C: Supervised Multi-Layer Perceptron (Reference)
    mlp_train_x = np.concatenate([win_train, win_dos[:10]], axis=0)
    mlp_train_y = np.concatenate([y_train, y_dos[:10]], axis=0)
    mlp_detector = MLPDetector(hidden_layer_sizes=(32, 16), max_iter=100, random_state=42).fit(mlp_train_x, mlp_train_y)

    # Model D: MPS Born Machine (Generative Quantum-Inspired)
    num_features = phi_train.shape[1]
    mps_detector = MPSBornMachine(num_sites=num_features, physical_dim=2, bond_dimension=16)
    dmrg_opt = DMRGOptimizer(max_bond_dimension=16, learning_rate=0.08, num_sweeps=4)

    def _train_mps():
        return dmrg_opt.fit(mps_detector, phi_train)

    _, mps_train_perf = ResourceProfiler.profile_training(_train_mps)
    mps_val_scores = mps_detector.score_samples(phi_val)
    mps_calib = ThresholdCalibrator(percentile=95.0).fit(mps_val_scores)

    # 4. Comparative Evaluation on Attack Scenarios
    print("[Step 4/5] Evaluating detection metrics across attack categories...")
    test_scenarios = [
        ("DoS_Flood", win_dos, phi_dos, y_dos, prov_dos, df_test_dos),
        ("Fuzzing", win_fuzz, phi_fuzz, y_fuzz, prov_fuzz, df_test_fuzz),
        ("Payload_Spoofing", win_spoof, phi_spoof, y_spoof, prov_spoof, df_test_spoof),
    ]

    for scenario_name, w_test, p_test, y_test, prov_test, raw_df in test_scenarios:
        print(f"\n--- Scenario: {scenario_name} (Class Imbalance: {np.sum(y_test==0)}:{np.sum(y_test==1)}) ---")
        scenario_results = {}

        # Evaluate Statistical
        scores_stat = stat_detector.score_windows(w_test)
        preds_stat = stat_calib.predict(scores_stat)
        m_stat = compute_ids_metrics(y_test, preds_stat, scores_stat)
        scenario_results["Statistical"] = m_stat

        # Evaluate Isolation Forest
        scores_if = iforest_detector.score_windows(w_test)
        preds_if = iforest_calib.predict(scores_if)
        m_if = compute_ids_metrics(y_test, preds_if, scores_if)
        scenario_results["IsolationForest"] = m_if

        # Evaluate MLP
        scores_mlp = mlp_detector.score_windows(w_test)
        preds_mlp = (scores_mlp > 0.5).astype(int)
        m_mlp = compute_ids_metrics(y_test, preds_mlp, scores_mlp)
        scenario_results["MLP_Reference"] = m_mlp

        # Evaluate MPS Born Machine
        scores_mps = mps_detector.score_samples(p_test)
        preds_mps = mps_calib.predict(scores_mps)
        m_mps = compute_ids_metrics(y_test, preds_mps, scores_mps)
        scenario_results["MPS_BornMachine"] = m_mps

        results[scenario_name] = scenario_results

        print(f"{'Model':<20} | {'F1 (Weighted)':<14} | {'Recall':<8} | {'Precision':<10} | {'FPR':<8} | {'G-mean':<8}")
        print("-" * 75)
        for model_name, m in scenario_results.items():
            print(f"{model_name:<20} | {m['f1_weighted']:<14.4f} | {m['recall_weighted']:<8.4f} | {m['precision_weighted']:<10.4f} | {m['fpr']:<8.4f} | {m['g_mean']:<8.4f}")

    # 5. Native Quantum Explainability & ISO/SAE 21434 Forensics
    print("\n[Step 5/5] Generating ISO/SAE 21434 forensic incident report via MPS RDMs...")
    rdm_extractor = RDMExtractor(mps_detector.mps)
    rdms = [rdm_extractor.get_single_site_rdm(k) for k in range(num_features)]
    entropies = compute_all_site_entropies(rdms)

    # Find the top anomaly in DoS scenario
    dos_scores = mps_detector.score_samples(phi_dos)
    top_anom_idx = int(np.argmax(dos_scores))
    top_window_phi = phi_dos[top_anom_idx]

    feature_probs = compute_feature_marginal_probabilities(rdms, top_window_phi)
    baseline_probs = compute_feature_marginal_probabilities(rdms, phi_val[0])

    anom_feats_list = []
    for f_idx, feat_name in enumerate(feature_cols):
        anom_feats_list.append({
            "feature": feat_name,
            "observed_marginal_p": round(float(feature_probs[f_idx]), 6),
            "baseline_p": round(float(baseline_probs[f_idx]), 6),
            "von_neumann_entropy_bits": round(float(entropies[f_idx]), 4),
        })

    prov_info = trace_window_to_raw_traffic(prov_dos[top_anom_idx], df_test_dos)
    meta = {
        "architecture": "Matrix Product State (Born Machine)",
        "bond_dimension": mps_detector.bond_dimension,
        "physical_dimension": mps_detector.physical_dim,
        "features": feature_cols,
        "training_time_sec": mps_train_perf["training_time_sec"],
        "peak_memory_mb": mps_train_perf["peak_memory_mb"],
    }

    report_yaml = build_iso_21434_incident_report(
        alert_id="CANQI-2026-EXP001",
        anomaly_nll=float(dos_scores[top_anom_idx]),
        threshold=float(mps_calib.threshold),
        provenance_info=prov_info,
        anomalous_features=anom_feats_list,
        model_metadata=meta
    )

    with open("forensic_incident_report_sample.yaml", "w", encoding="utf-8") as f:
        f.write(report_yaml)
    print("[OK] Forensic report written to: forensic_incident_report_sample.yaml")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Full benchmark results exported to: {output_json}")

    print("\n" + "=" * 80)
    print("Experimental evaluation completed successfully.")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_experiment_suite()
