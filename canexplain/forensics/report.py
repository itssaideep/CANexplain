"""Forensic incident report builder with deterministic SHA-256 evidence integrity hashing.

Scientific & Regulatory Notice:
    This module generates structured incident reports designed to assist human forensic
    analysts by linking statistical tensor-network anomaly alerts back to raw CAN frames
    and provenance metadata. While conceptually aligned with automotive cybersecurity
    monitoring workflows (such as ISO/SAE 21434 Clause 13 and UNECE R155), this software
    is a RESEARCH PROTOTYPE. It does NOT constitute formal, certified ISO/SAE 21434 compliance
    or guaranteed legal admissibility in judicial proceedings.
"""

import hashlib
import json
from datetime import datetime
from typing import Any

import yaml


def compute_evidence_sha256(evidence_payload: dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 digest over canonicalized evidence metadata.

    Canonicalization Rules:
        1. Keys sorted recursively in ascending alphabetical order.
        2. Strict JSON formatting with separators (',', ':') to eliminate
           non-deterministic whitespace.
        3. UTF-8 encoded byte stream.

    Args:
        evidence_payload: Dictionary containing forensic evidence data.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    canonical_json = json.dumps(
        evidence_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def verify_report_integrity(report_input: str | dict[str, Any]) -> bool:
    """Verifies the SHA-256 evidence integrity hash of an incident report.

    Args:
        report_input: YAML string or parsed dictionary of the forensic report.

    Returns:
        True if the evidence payload matches the recorded SHA-256 digest; False if tampered.
    """
    if isinstance(report_input, str):
        try:
            report_dict = yaml.safe_load(report_input)
        except Exception:
            return False
    else:
        report_dict = report_input

    if not isinstance(report_dict, dict) or "forensic_incident_report" not in report_dict:
        return False

    root = report_dict["forensic_incident_report"]
    recorded_hash = root.get("evidence_integrity_sha256")
    if not recorded_hash:
        return False

    # Reconstruct exact evidence payload used to generate hash
    evidence_payload = {
        "report_id": root.get("report_id"),
        "incident_summary": root.get("incident_summary"),
        "provenance_trace": root.get("provenance_trace"),
        "anomalous_feature_attributions": root.get("anomalous_feature_attributions"),
        "model_metadata": root.get("model_metadata"),
    }

    computed_hash = compute_evidence_sha256(evidence_payload)
    return computed_hash == recorded_hash


def build_forensic_incident_report(
    alert_id: str,
    anomaly_nll: float,
    threshold: float,
    provenance_info: dict[str, Any],
    anomalous_features: list[dict[str, Any]],
    model_metadata: dict[str, Any],
    investigative_guidance: list[str] | None = None,
) -> str:
    """Builds a structured forensic incident report with deterministic SHA-256 evidence hashing.

    Args:
        alert_id: Unique identifier for the alert.
        anomaly_nll: Negative Log-Likelihood score of the anomalous window.
        threshold: Calibrated anomaly decision threshold.
        provenance_info: Dictionary containing window provenance and raw frame data.
        anomalous_features: List of dictionaries detailing feature probability collapses.
        model_metadata: Model architecture, bond dimension, and training metadata.
        investigative_guidance: Recommended action items for human forensic analysts.

    Returns:
        Structured YAML document string.
    """
    if investigative_guidance is None:
        investigative_guidance = [
            "Inspect physical bus wiring and OBD-II port for aftermarket rogue devices.",
            "Verify whether relevant ECUs logged Diagnostic Trouble Codes (DTCs).",
            "Review gateway firewall rules and inspect for buffer overflow or bus-off states.",
        ]

    confidence_margin = float(anomaly_nll - threshold)
    severity = (
        "CRITICAL"
        if confidence_margin > 20.0
        else ("HIGH" if confidence_margin > 5.0 else "MEDIUM")
    )

    incident_summary = {
        "severity": severity,
        "detection_mechanism": "MPS_Born_Machine_NLL_Exceedance",
        "anomaly_nll_score": round(float(anomaly_nll), 4),
        "calibrated_threshold": round(float(threshold), 4),
        "confidence_margin": round(confidence_margin, 4),
    }

    evidence_payload = {
        "report_id": alert_id,
        "incident_summary": incident_summary,
        "provenance_trace": provenance_info,
        "anomalous_feature_attributions": anomalous_features,
        "model_metadata": model_metadata,
    }

    sha256_hash = compute_evidence_sha256(evidence_payload)

    report_dict = {
        "forensic_incident_report": {
            "report_id": alert_id,
            "evidence_integrity_sha256": sha256_hash,
            "iso_sae_21434_clause": (
                "ISO/SAE 21434 Clause 13 - Conceptual Alignment "
                "(Cybersecurity Monitoring & Response)"
            ),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "incident_summary": incident_summary,
            "provenance_trace": provenance_info,
            "anomalous_feature_attributions": anomalous_features,
            "model_metadata": model_metadata,
            "investigative_guidance": investigative_guidance,
            "disclaimer": (
                "RESEARCH PROTOTYPE NOTICE: Statistical anomaly identified via Matrix Product "
                "State Born machine generative distribution. Marginal probability collapses "
                "indicate dimensions deviating from normal baseline, not confirmed causal "
                "attacker intent. Not certified for formal judicial legal admissibility "
                "without corroborating physical bus captures."
            ),
        }
    }

    return yaml.dump(report_dict, sort_keys=False, default_flow_style=False)


# Backward-compatible alias
build_iso_21434_incident_report = build_forensic_incident_report
