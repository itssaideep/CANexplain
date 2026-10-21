"""End-to-end experimental evaluation runner for CANExplain-QI.

Implements the experimental protocol specified in docs/EXPERIMENT_PROTOCOL.md:
- Experiment A: Within-dataset multi-trial evaluation
  (Statistical, Isolation Forest, MLP, MPS Born Machine).
- Experiment B: Cross-dataset evaluation (ROAD, HCRL fixtures).
- Experiment D: Sequence-preserving representation vs. pooled-window baseline ablation.
- Experiment E: Bond dimension scaling sweep (D in [8, 16, 24, 32]).
- Experiment F: Multi-seed replication reporting mean +/- sample standard deviation.
- Forensic support: Generates and verifies SHA-256 hashed incident reports.
"""

import argparse
import json
import os
import sys
import time
from typing import Any

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from canexplain.data.adapters import HCRLCarHackingAdapter, ROADDatasetAdapter
from canexplain.data.synthetic import generate_synthetic_can_stream
from canexplain.evaluation.benchmark import ResourceProfiler
from canexplain.evaluation.metrics import compute_ids_metrics
from canexplain.explainability.entropy import compute_all_site_entropies
from canexplain.explainability.feature_prob import compute_feature_marginal_probabilities
from canexplain.explainability.rdm import RDMExtractor
from canexplain.forensics.provenance import trace_window_to_raw_traffic
from canexplain.forensics.report import build_forensic_incident_report, verify_report_integrity
from canexplain.models.baselines.iforest import IsolationForestDetector
from canexplain.models.baselines.mlp import MLPDetector
from canexplain.models.baselines.statistical import StatisticalDeviationDetector
from canexplain.models.mps.born_machine import MPSBornMachine
from canexplain.models.mps.dmrg import HeuristicCorrelationOptimizer
from canexplain.models.threshold import ThresholdCalibrator
from canexplain.preprocessing.encoding import (
    MeanPooledFeatureMap,
    SequencePreservingEncoder,
    trigonometric_feature_map,
)
from canexplain.preprocessing.features import extract_canonical_features
from canexplain.preprocessing.normalization import CANFeatureScaler
from canexplain.preprocessing.windows import construct_sliding_windows

FIXTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures"))


def aggregate_metric_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes mean and sample standard deviation across multiple experimental runs."""
    if not runs:
        return {}
    keys = runs[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float, np.floating, np.integer))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg[f"{k}_all"] = [float(v) for v in vals]
        else:
            agg[k] = runs[0][k]
    return agg


def run_experiment_suite(
    output_json: str = "experiment_results.json",
    num_trials: int = 3,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Executes the full suite of reproducible experiments."""
    if seeds is None:
        seeds = [42, 101, 202]
    print("=" * 80)
    print("CANExplain-QI: Quantum-Inspired Automotive CAN-Bus Intrusion Detection")
    print(f"Experimental Evaluation Suite ({num_trials} Trials per Model | Seeds: {seeds})")
    print("=" * 80)

    # 1. Generate Synthetic Datasets
    print(
        "\n[Step 1/6] Synthesizing multi-scenario CAN traffic streams (10 stationary features)..."
    )
    df_train = generate_synthetic_can_stream(num_frames=1200, attack_type="none", seed=101)
    df_val = generate_synthetic_can_stream(num_frames=600, attack_type="none", seed=202)
    df_dos = generate_synthetic_can_stream(
        num_frames=800, attack_type="dos", attack_start_idx=300, attack_duration=150, seed=303
    )
    df_fuzz = generate_synthetic_can_stream(
        num_frames=800, attack_type="fuzzing", attack_start_idx=300, attack_duration=150, seed=404
    )
    df_spoof = generate_synthetic_can_stream(
        num_frames=800, attack_type="spoofing", attack_start_idx=300, attack_duration=150, seed=505
    )

    # Extract 10 stationary canonical features
    feats_train = extract_canonical_features(df_train)
    feats_val = extract_canonical_features(df_val)
    feats_dos = extract_canonical_features(df_dos)
    feats_fuzz = extract_canonical_features(df_fuzz)
    feats_spoof = extract_canonical_features(df_spoof)

    feature_cols = [c for c in feats_train.columns if c not in ("label", "subclass")]
    scaler = CANFeatureScaler().fit(feats_train, feature_cols=feature_cols)

    norm_train = scaler.transform(feats_train)
    norm_val = scaler.transform(feats_val)
    norm_dos = scaler.transform(feats_dos)
    norm_fuzz = scaler.transform(feats_fuzz)
    norm_spoof = scaler.transform(feats_spoof)

    window_size, stride = 32, 16

    win_train, y_train, _ = construct_sliding_windows(
        norm_train,
        df_train["label"].to_numpy(),
        df_train["timestamp"].to_numpy(),
        window_size,
        stride,
    )
    win_val, y_val, _ = construct_sliding_windows(
        norm_val, df_val["label"].to_numpy(), df_val["timestamp"].to_numpy(), window_size, stride
    )
    win_dos, y_dos, prov_dos = construct_sliding_windows(
        norm_dos, df_dos["label"].to_numpy(), df_dos["timestamp"].to_numpy(), window_size, stride
    )
    win_fuzz, y_fuzz, prov_fuzz = construct_sliding_windows(
        norm_fuzz, df_fuzz["label"].to_numpy(), df_fuzz["timestamp"].to_numpy(), window_size, stride
    )
    win_spoof, y_spoof, prov_spoof = construct_sliding_windows(
        norm_spoof,
        df_spoof["label"].to_numpy(),
        df_spoof["timestamp"].to_numpy(),
        window_size,
        stride,
    )

    pooled_encoder = MeanPooledFeatureMap()
    phi_train = pooled_encoder.encode(win_train)
    phi_val = pooled_encoder.encode(win_val)
    phi_dos = pooled_encoder.encode(win_dos)
    phi_fuzz = pooled_encoder.encode(win_fuzz)
    phi_spoof = pooled_encoder.encode(win_spoof)

    results: dict[str, Any] = {
        "metadata": {
            "num_trials": num_trials,
            "seeds": seeds,
            "feature_representation": "canonical_10_stationary",
            "window_size": window_size,
            "stride": stride,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "experiments": {},
    }

    # -------------------------------------------------------------------------
    # Experiment A: Within-Dataset Multi-Trial Benchmark
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Executing Experiment A: Within-Dataset Multi-Trial Benchmark...")
    test_scenarios = [
        ("DoS_Flood", win_dos, phi_dos, y_dos, prov_dos, df_dos),
        ("Fuzzing", win_fuzz, phi_fuzz, y_fuzz, prov_fuzz, df_fuzz),
        ("Payload_Spoofing", win_spoof, phi_spoof, y_spoof, prov_spoof, df_spoof),
    ]

    exp_a_results: dict[str, Any] = {}

    for scenario_name, w_test, p_test, y_test, _prov_test, _raw_df in test_scenarios:
        benign_cnt = int(np.sum(y_test == 0))
        attack_cnt = int(np.sum(y_test == 1))
        print(f"\n--- Scenario: {scenario_name} (Class Imbalance: {benign_cnt}:{attack_cnt}) ---")
        models = ["Statistical", "IsolationForest", "MLP_Reference", "MPS_BornMachine"]
        runs_by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in models}

        for _trial_idx, seed in enumerate(seeds[:num_trials]):
            np.random.seed(seed)

            # Model 1: Statistical Deviation Detector
            stat = StatisticalDeviationDetector().fit(win_train)
            stat_val_scores = stat.score_windows(win_val)
            stat_calib = ThresholdCalibrator(percentile=95.0).fit(stat_val_scores)
            scores_stat = stat.score_windows(w_test)
            preds_stat = stat_calib.predict(scores_stat)
            m_stat = compute_ids_metrics(y_test, preds_stat, scores_stat, eval_unit="window")
            runs_by_model["Statistical"].append(m_stat)

            # Model 2: Isolation Forest
            iforest = IsolationForestDetector(n_estimators=50, random_state=seed).fit(win_train)
            iforest_val_scores = iforest.score_windows(win_val)
            iforest_calib = ThresholdCalibrator(percentile=95.0).fit(iforest_val_scores)
            scores_if = iforest.score_windows(w_test)
            preds_if = iforest_calib.predict(scores_if)
            m_if = compute_ids_metrics(y_test, preds_if, scores_if, eval_unit="window")
            runs_by_model["IsolationForest"].append(m_if)

            # Model 3: Supervised MLP (Zero-Leakage: trained on training set + synthetic anomalies)
            rng = np.random.default_rng(seed)
            num_syn_anom = min(len(win_train), 30)
            syn_anom_windows = win_train[:num_syn_anom].copy()
            for s_i in range(num_syn_anom):
                if s_i % 2 == 0:
                    syn_anom_windows[s_i, :, 0] *= 0.1  # Fast flood
                else:
                    syn_anom_windows[s_i, :, 2:] = rng.uniform(
                        0.0, 1.0, size=syn_anom_windows[s_i, :, 2:].shape
                    )

            mlp_x = np.concatenate([win_train, syn_anom_windows], axis=0)
            mlp_y = np.concatenate([np.zeros(len(win_train)), np.ones(num_syn_anom)], axis=0)
            mlp = MLPDetector(hidden_layer_sizes=(32, 16), max_iter=80, random_state=seed).fit(
                mlp_x, mlp_y
            )
            scores_mlp = mlp.score_windows(w_test)
            preds_mlp = (scores_mlp > 0.5).astype(int)
            m_mlp = compute_ids_metrics(y_test, preds_mlp, scores_mlp, eval_unit="window")
            runs_by_model["MLP_Reference"].append(m_mlp)

            # Model 4: MPS Born Machine
            bm = MPSBornMachine(
                num_sites=phi_train.shape[1], physical_dim=2, bond_dimension=16, seed=seed
            )
            opt = HeuristicCorrelationOptimizer(
                max_bond_dimension=16, learning_rate=0.08, num_sweeps=3
            )
            opt.fit(bm, phi_train)
            bm_val_scores = bm.score_samples(phi_val)
            bm_calib = ThresholdCalibrator(percentile=95.0).fit(bm_val_scores)
            scores_bm = bm.score_samples(p_test)
            preds_bm = bm_calib.predict(scores_bm)
            m_bm = compute_ids_metrics(y_test, preds_bm, scores_bm, eval_unit="window")
            runs_by_model["MPS_BornMachine"].append(m_bm)

        # Aggregate across trials
        scenario_summary = {}
        hdr = (
            f"{'Model':<18} | {'F1_Binary (Mean +/- Std)':<24} | "
            f"{'Recall':<16} | {'Precision':<16} | {'FPR':<16}"
        )
        print(hdr)
        print("-" * 100)
        for model_name, run_list in runs_by_model.items():
            agg = aggregate_metric_runs(run_list)
            scenario_summary[model_name] = agg
            f1_str = f"{agg['f1_binary_mean']:.4f} +/- {agg['f1_binary_std']:.4f}"
            rec_str = f"{agg['recall_binary_mean']:.4f}"
            prec_str = f"{agg['precision_binary_mean']:.4f}"
            fpr_str = f"{agg['fpr_mean']:.4f}"
            print(
                f"{model_name:<18} | {f1_str:<24} | {rec_str:<16} | {prec_str:<16} | {fpr_str:<16}"
            )

        exp_a_results[scenario_name] = scenario_summary

    results["experiments"]["Experiment_A_Within_Dataset"] = exp_a_results

    # -------------------------------------------------------------------------
    # Experiment B: Cross-Dataset Evaluation (Zero Test Leakage)
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Executing Experiment B: Cross-Dataset Transferability...")
    cross_results: dict[str, Any] = {}

    # Train MPS on can-train-and-test-v1.5, evaluate on ROAD and HCRL fixtures
    bm_cross = MPSBornMachine(num_sites=10, physical_dim=2, bond_dimension=16, seed=42)
    opt_cross = HeuristicCorrelationOptimizer(
        max_bond_dimension=16, learning_rate=0.08, num_sweeps=3
    )
    opt_cross.fit(bm_cross, phi_train)
    val_calib = ThresholdCalibrator(percentile=95.0).fit(bm_cross.score_samples(phi_val))

    # Evaluate on ROAD fixture
    road_fixture = os.path.join(FIXTURES_DIR, "road_sample.csv")
    if os.path.exists(road_fixture):
        road_adapter = ROADDatasetAdapter(vehicle_id="kia_soul")
        df_road = road_adapter.load_trace(road_fixture, trace_name="road_sample")
        feats_road = extract_canonical_features(df_road)
        norm_road = scaler.transform(feats_road)  # Fitted on training split only
        phi_road = trigonometric_feature_map(norm_road)
        scores_road = bm_cross.score_samples(phi_road)
        preds_road = val_calib.predict(scores_road)
        m_road = compute_ids_metrics(
            df_road["label"].to_numpy(), preds_road, scores_road, eval_unit="frame"
        )
        cross_results["ROAD_Transfer"] = m_road
        print(
            f"ROAD Transfer    : Accuracy={m_road['accuracy']:.4f}, "
            f"TPR={m_road['tpr']:.4f}, FPR={m_road['fpr']:.4f}"
        )

    # Evaluate on HCRL fixture
    hcrl_fixture = os.path.join(FIXTURES_DIR, "hcrl_sample.csv")
    if os.path.exists(hcrl_fixture):
        hcrl_adapter = HCRLCarHackingAdapter(vehicle_id="hyundai_yf_sonata")
        df_hcrl = hcrl_adapter.load_trace(hcrl_fixture, trace_name="hcrl_sample")
        feats_hcrl = extract_canonical_features(df_hcrl)
        norm_hcrl = scaler.transform(feats_hcrl)
        phi_hcrl = trigonometric_feature_map(norm_hcrl)
        scores_hcrl = bm_cross.score_samples(phi_hcrl)
        preds_hcrl = val_calib.predict(scores_hcrl)
        m_hcrl = compute_ids_metrics(
            df_hcrl["label"].to_numpy(), preds_hcrl, scores_hcrl, eval_unit="frame"
        )
        cross_results["HCRL_Transfer"] = m_hcrl
        print(
            f"HCRL Transfer    : Accuracy={m_hcrl['accuracy']:.4f}, "
            f"TPR={m_hcrl['tpr']:.4f}, FPR={m_hcrl['fpr']:.4f}"
        )

    results["experiments"]["Experiment_B_Cross_Dataset"] = cross_results

    # -------------------------------------------------------------------------
    # Experiment D: Representation Ablation (Sequence-Preserving vs. Mean-Pooled)
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Executing Experiment D: Sequence-Preserving vs. Pooled-Window Ablation...")
    seq_encoder_flat = SequencePreservingEncoder(mode="flatten")
    # For sequence representation, windows of size W=8, F=10 -> 80 sites
    w_small, s_small = 8, 4
    win_tr_sm, _, _ = construct_sliding_windows(
        norm_train, df_train["label"].to_numpy(), df_train["timestamp"].to_numpy(), w_small, s_small
    )
    win_val_sm, _, _ = construct_sliding_windows(
        norm_val, df_val["label"].to_numpy(), df_val["timestamp"].to_numpy(), w_small, s_small
    )
    win_spoof_sm, y_spoof_sm, _ = construct_sliding_windows(
        norm_spoof, df_spoof["label"].to_numpy(), df_spoof["timestamp"].to_numpy(), w_small, s_small
    )

    phi_tr_seq = seq_encoder_flat.encode(win_tr_sm)
    phi_val_seq = seq_encoder_flat.encode(win_val_sm)
    phi_sp_seq = seq_encoder_flat.encode(win_spoof_sm)

    phi_tr_pool = pooled_encoder.encode(win_tr_sm)
    phi_val_pool = pooled_encoder.encode(win_val_sm)
    phi_sp_pool = pooled_encoder.encode(win_spoof_sm)

    # 1. Pooled MPS (10 sites)
    bm_pool = MPSBornMachine(num_sites=10, physical_dim=2, bond_dimension=12, seed=42)
    opt_pool = HeuristicCorrelationOptimizer(
        max_bond_dimension=12, learning_rate=0.08, num_sweeps=2
    )
    opt_pool.fit(bm_pool, phi_tr_pool)
    calib_pool = ThresholdCalibrator(percentile=95.0).fit(bm_pool.score_samples(phi_val_pool))
    scores_pool = bm_pool.score_samples(phi_sp_pool)
    m_pool = compute_ids_metrics(
        y_spoof_sm, calib_pool.predict(scores_pool), scores_pool, eval_unit="window"
    )

    # 2. Sequence-Preserving MPS (80 sites)
    bm_seq = MPSBornMachine(num_sites=80, physical_dim=2, bond_dimension=6, seed=42)
    opt_seq = HeuristicCorrelationOptimizer(max_bond_dimension=6, learning_rate=0.05, num_sweeps=2)
    opt_seq.fit(bm_seq, phi_tr_seq)
    calib_seq = ThresholdCalibrator(percentile=95.0).fit(bm_seq.score_samples(phi_val_seq))
    scores_seq = bm_seq.score_samples(phi_sp_seq)
    m_seq = compute_ids_metrics(
        y_spoof_sm, calib_seq.predict(scores_seq), scores_seq, eval_unit="window"
    )

    rep_ablation = {
        "MeanPooled_Baseline": m_pool,
        "SequencePreserving": m_seq,
    }
    print(
        f"MeanPooled F1_Binary   : {m_pool['f1_binary']:.4f} "
        f"(Recall: {m_pool['recall_binary']:.4f})"
    )
    print(
        f"SeqPreserve F1_Binary  : {m_seq['f1_binary']:.4f} "
        f"(Recall: {m_seq['recall_binary']:.4f})"
    )
    results["experiments"]["Experiment_D_Representation_Ablation"] = rep_ablation

    # -------------------------------------------------------------------------
    # Experiment E: Bond Dimension Scaling Sweep
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] Executing Experiment E: Bond Dimension Scaling (D in [8, 16, 24, 32])...")
    bond_sweep_results = {}
    for d in [8, 16, 24, 32]:
        bm_d = MPSBornMachine(num_sites=10, physical_dim=2, bond_dimension=d, seed=42)
        opt_d = HeuristicCorrelationOptimizer(
            max_bond_dimension=d, learning_rate=0.08, num_sweeps=2
        )

        _, train_perf = ResourceProfiler.profile_training(
            lambda o=opt_d, b=bm_d: o.fit(b, phi_train)
        )
        calib_d = ThresholdCalibrator(percentile=95.0).fit(bm_d.score_samples(phi_val))

        t0 = time.perf_counter()
        scores_d = bm_d.score_samples(phi_dos)
        latency_us = ((time.perf_counter() - t0) / len(phi_dos)) * 1e6

        m_d = compute_ids_metrics(y_dos, calib_d.predict(scores_d), scores_d, eval_unit="window")
        m_d["training_time_sec"] = train_perf["training_time_sec"]
        m_d["peak_memory_mb"] = train_perf["peak_memory_mb"]
        m_d["inference_latency_us_per_window"] = float(latency_us)
        bond_sweep_results[f"D_{d}"] = m_d
        print(
            f"D={d:2d} | F1_Binary={m_d['f1_binary']:.4f} | "
            f"Latency={latency_us:.1f} us/win | Train={train_perf['training_time_sec']:.2f}s"
        )

    results["experiments"]["Experiment_E_Bond_Dimension_Scaling"] = bond_sweep_results

    # -------------------------------------------------------------------------
    # Forensic Evidence Report Generation & SHA-256 Integrity Verification
    # -------------------------------------------------------------------------
    print("\n[Step 6/6] Generating and Verifying Forensic Incident Report...")
    bm_best = MPSBornMachine(num_sites=10, physical_dim=2, bond_dimension=16, seed=42)
    HeuristicCorrelationOptimizer(max_bond_dimension=16, learning_rate=0.08, num_sweeps=2).fit(
        bm_best, phi_train
    )
    calib_best = ThresholdCalibrator(percentile=95.0).fit(bm_best.score_samples(phi_val))

    scores_dos = bm_best.score_samples(phi_dos)
    top_idx = int(np.argmax(scores_dos))
    top_phi = phi_dos[top_idx]

    extractor = RDMExtractor(bm_best.mps)
    rdms = [extractor.get_single_site_rdm(k) for k in range(10)]
    entropies = compute_all_site_entropies(rdms)
    feature_probs = compute_feature_marginal_probabilities(rdms, top_phi)
    base_probs = compute_feature_marginal_probabilities(rdms, phi_val[0])

    anom_feats = []
    for i, col in enumerate(feature_cols):
        anom_feats.append(
            {
                "feature": col,
                "observed_marginal_p": round(float(feature_probs[i]), 6),
                "baseline_p": round(float(base_probs[i]), 6),
                "von_neumann_entropy_bits": round(float(entropies[i]), 4),
            }
        )

    prov_record = trace_window_to_raw_traffic(prov_dos[top_idx], df_dos)
    meta = {
        "model": "MPS_BornMachine",
        "bond_dimension": 16,
        "physical_dimension": 2,
        "features": feature_cols,
        "num_sites": 10,
    }

    report_yaml = build_forensic_incident_report(
        alert_id="CANQI-EXP-VALIDATED-001",
        anomaly_nll=float(scores_dos[top_idx]),
        threshold=float(calib_best.threshold),
        provenance_info=prov_record,
        anomalous_features=anom_feats,
        model_metadata=meta,
    )

    with open("forensic_incident_report_sample.yaml", "w", encoding="utf-8") as f:
        f.write(report_yaml)

    is_valid = verify_report_integrity(report_yaml)
    assert is_valid is True, "Report integrity check failed!"
    print("[OK] Forensic incident report validated with SHA-256 integrity check (Status: VERIFIED)")

    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Full experiment results exported to {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CANExplain-QI experimental evaluation.")
    parser.add_argument("--output", type=str, default="experiment_results.json")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    run_experiment_suite(output_json=args.output, num_trials=args.trials)
