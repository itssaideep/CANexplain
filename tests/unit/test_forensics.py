"""Unit tests for forensic provenance tracing and ISO/SAE 21434 report builder."""

import pytest
import yaml
import pandas as pd
import numpy as np

from canexplain.data.synthetic import generate_synthetic_can_stream
from canexplain.preprocessing.windows import WindowProvenance
from canexplain.forensics.provenance import trace_window_to_raw_traffic
from canexplain.forensics.report import build_iso_21434_incident_report


def test_forensic_provenance_tracing():
    df_raw = generate_synthetic_can_stream(num_frames=100, attack_type="none")
    prov = WindowProvenance(
        window_idx=0,
        start_frame_idx=10,
        end_frame_idx=29,
        start_timestamp=float(df_raw["timestamp"].iloc[10]),
        end_timestamp=float(df_raw["timestamp"].iloc[29]),
        duration_seconds=float(df_raw["timestamp"].iloc[29] - df_raw["timestamp"].iloc[10]),
        frame_count=20
    )

    trace_info = trace_window_to_raw_traffic(prov, df_raw)

    assert trace_info["window_index"] == 0
    assert trace_info["frame_count"] == 20
    assert "primary_arbitration_id" in trace_info
    assert "arbitration_id_counts" in trace_info
    assert trace_info["unique_id_count"] > 0


def test_build_iso_21434_incident_report():
    prov_info = {
        "window_index": 42,
        "start_frame_index": 200,
        "end_frame_index": 263,
        "primary_arbitration_id": "0x000",
        "frame_count": 64
    }
    anom_feats = [
        {"feature": "delta_t", "observed_p": 0.001, "baseline_p": 0.45},
        {"feature": "byte_0", "observed_p": 0.002, "baseline_p": 0.50}
    ]
    meta = {
        "architecture": "Matrix Product State (Born Machine)",
        "bond_dimension": 64,
        "training_data": "can-train-and-test-v1.5 set_01"
    }

    report_yaml = build_iso_21434_incident_report(
        alert_id="CANQI-2026-TEST",
        anomaly_nll=45.2,
        threshold=25.0,
        provenance_info=prov_info,
        anomalous_features=anom_feats,
        model_metadata=meta
    )

    # Validate valid YAML
    parsed = yaml.safe_load(report_yaml)
    assert "forensic_incident_report" in parsed
    inc = parsed["forensic_incident_report"]
    assert inc["report_id"] == "CANQI-2026-TEST"
    assert inc["incident_summary"]["severity"] == "CRITICAL"
    assert len(inc["anomalous_feature_attributions"]) == 2
    assert "ISO/SAE 21434" in inc["iso_sae_21434_clause"]
    assert "evidence_integrity_sha256" in inc
    assert len(inc["evidence_integrity_sha256"]) == 64

    # Verify report integrity
    from canexplain.forensics.report import verify_report_integrity
    assert verify_report_integrity(report_yaml) is True

    # Test tampering detection: modify anomaly NLL in the YAML
    tampered_yaml = report_yaml.replace("anomaly_nll_score: 45.2", "anomaly_nll_score: 12.0")
    assert verify_report_integrity(tampered_yaml) is False

    # Test tampering detection: modify arbitration ID
    tampered_yaml_id = report_yaml.replace("primary_arbitration_id: '0x000'", "primary_arbitration_id: '0x120'")
    assert verify_report_integrity(tampered_yaml_id) is False
