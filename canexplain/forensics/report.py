"""ISO/SAE 21434-aligned structured forensic incident report builder."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import yaml


def build_iso_21434_incident_report(
    alert_id: str,
    anomaly_nll: float,
    threshold: float,
    provenance_info: Dict[str, Any],
    anomalous_features: List[Dict[str, Any]],
    model_metadata: Dict[str, Any],
    investigative_guidance: Optional[List[str]] = None,
) -> str:
    """Builds an ISO/SAE 21434-compliant forensic incident report in YAML format.

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
            "Verify whether relevant Electronic Control Units (ECUs) logged Diagnostic Trouble Codes (DTCs).",
            "Review gateway firewall rules and inspect for buffer overflow or bus-off states."
        ]

    confidence_margin = float(anomaly_nll - threshold)
    severity = "CRITICAL" if confidence_margin > 20.0 else ("HIGH" if confidence_margin > 5.0 else "MEDIUM")

    report_dict = {
        "forensic_incident_report": {
            "report_id": alert_id,
            "iso_sae_21434_clause": "ISO/SAE 21434 Clause 13 - Cybersecurity Monitoring and Incident Response",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "incident_summary": {
                "severity": severity,
                "detection_mechanism": "MPS_Born_Machine_NLL_Exceedance",
                "anomaly_nll_score": round(float(anomaly_nll), 4),
                "calibrated_threshold": round(float(threshold), 4),
                "confidence_margin": round(confidence_margin, 4),
            },
            "provenance_trace": provenance_info,
            "anomalous_feature_attributions": anomalous_features,
            "model_metadata": model_metadata,
            "investigative_guidance": investigative_guidance,
            "disclaimer": (
                "Observed statistical anomaly under MPS generative normal distribution. "
                "Attributions indicate statistically deviant dimensions, not confirmed causal intent."
            )
        }
    }

    return yaml.dump(report_dict, sort_keys=False, default_flow_style=False)
